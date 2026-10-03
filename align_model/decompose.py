"""Split a lag curve into offset, drift, smooth residual, and jumps.

The fit does not move stretch markers. Callers pass the curve they already
measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats

WEAK_SCORE = 0.35


@dataclass
class Jump:
    time: float
    size_sec: float
    score: float

    def as_dict(self) -> dict:
        return {"time": self.time, "size_sec": self.size_sec, "score": self.score}


@dataclass
class AlignmentReport:
    offset_sec: float = 0.0
    drift: float = 0.0
    drift_ms_per_min: float = 0.0
    nonlinear_rms_sec: float = 0.0
    max_local_error_sec: float = 0.0
    median_error_sec: float = 0.0
    p95_error_sec: float = 0.0
    n_discontinuities: int = 0
    weak_frac: float = 0.0
    confidence: float = 0.0
    jumps: list = field(default_factory=list)
    times: list = field(default_factory=list)
    lags: list = field(default_factory=list)
    confidence_series: list = field(default_factory=list)
    low_conf_spans: list = field(default_factory=list)
    beats: list = field(default_factory=list)
    downbeats: list = field(default_factory=list)
    gap_inserts: list = field(default_factory=list)
    markers: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    profile: str = "default"
    rate_limit: dict = field(default_factory=dict)
    line_offset: float = 0.0
    line_slope: float = 0.0
    vocal: dict | None = None
    instrumental: dict | None = None
    mix: dict | None = None
    aca_curve: dict | None = None
    inst_curve: dict | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["jumps"] = [j.as_dict() if isinstance(j, Jump) else j for j in self.jumps]
        return data


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med))) * 1.4826


def _ransac_line(times: np.ndarray, lags: np.ndarray, *, thresh: float = 0.02) -> tuple[float, float, np.ndarray]:
    n = len(times)
    mask = np.ones(n, dtype=bool)
    if n == 0:
        return 0.0, 0.0, mask
    if n == 1:
        return float(lags[0]), 0.0, mask
    rng = np.random.default_rng(0)
    thresh = float(thresh)
    best_inliers = mask
    best_count = -1
    best = (float(np.median(lags)), 0.0)
    iters = 48 if n >= 2 else 0
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        dt = float(times[j] - times[i])
        if abs(dt) < 1e-6:
            continue
        slope = float(lags[j] - lags[i]) / dt
        offset = float(lags[i] - slope * times[i])
        resid = np.abs(lags - (offset + slope * times))
        inliers = resid <= thresh
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best = (offset, slope)
    if best_count < 2:
        design = np.vstack([np.ones(n), times]).T
        coef, *_ = np.linalg.lstsq(design, lags, rcond=None)
        resid = np.abs(lags - (coef[0] + coef[1] * times))
        return float(coef[0]), float(coef[1]), resid <= thresh
    chosen = best_inliers
    design = np.vstack([np.ones(int(np.count_nonzero(chosen))), times[chosen]]).T
    coef, *_ = np.linalg.lstsq(design, lags[chosen], rcond=None)
    return float(coef[0]), float(coef[1]), chosen


def robust_line(
    times: np.ndarray,
    lags: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    fit_tol_sec: float = 0.02,
) -> tuple[float, float, np.ndarray]:
    """Return intercept, slope, and an inlier mask. Tiny clouds use RANSAC."""
    t = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(lags, dtype=float).reshape(-1)
    n = len(t)
    if n == 0:
        return 0.0, 0.0, np.zeros(0, dtype=bool)
    if n < 6:
        return _ransac_line(t, y, thresh=fit_tol_sec)
    try:
        slope, intercept, _lo, _hi = stats.theilslopes(y, t)
        offset = float(intercept)
        slope = float(slope)
    except (ValueError, np.linalg.LinAlgError):
        return _ransac_line(t, y, thresh=fit_tol_sec)
    resid = y - (offset + slope * t)
    scale = _mad(resid)
    tol = max(3.0 * scale, float(fit_tol_sec))
    mask = np.abs(resid) <= tol
    if int(np.count_nonzero(mask)) < max(3, n // 5):
        mask = np.ones(n, dtype=bool)
    if scores is not None and len(scores) == n:
        # A very low score does not earn an inlier vote when the residual is large.
        weak = np.asarray(scores, dtype=float) < WEAK_SCORE
        mask = mask & ~(weak & (np.abs(resid) > tol * 0.5))
        if int(np.count_nonzero(mask)) < max(3, n // 5):
            mask = np.abs(resid) <= tol
    return offset, slope, mask


def find_jumps(
    times: np.ndarray,
    lags: np.ndarray,
    scores: np.ndarray | None,
    offset: float,
    slope: float,
) -> tuple[list[Jump], np.ndarray]:
    """Steps and spikes whose residual leaves both the previous sample and the local scale.

    A step is far from the previous sample and then holds. A spike is far from
    both immediate neighbors. Both are removed before the smooth-residual stats.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(lags, dtype=float).reshape(-1)
    n = len(t)
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        return [], mask
    resid = y - (offset + slope * t)
    scale = _mad(np.diff(resid))
    thr = max(3.0 * scale, 0.04)
    jumps: list[Jump] = []
    for i in range(1, n - 1):
        left = abs(float(resid[i] - resid[i - 1]))
        right = abs(float(resid[i] - resid[i + 1]))
        spike = left > thr and right > thr
        step = left > thr and right <= thr
        if not spike and not step:
            continue
        mask[i] = True
        score = 0.0
        if scores is not None and i < len(scores):
            score = float(scores[i])
        jumps.append(Jump(float(t[i]), float(resid[i] - resid[i - 1]), score))
    return jumps, mask


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def decompose(
    times: np.ndarray,
    lags: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    weak_score: float = WEAK_SCORE,
    fit_tol_sec: float = 0.02,
) -> AlignmentReport:
    """Measure offset, linear drift, nonlinear residual, and discontinuities."""
    t = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(lags, dtype=float).reshape(-1)
    if scores is None or len(scores) != len(y):
        sc = np.ones(len(y), dtype=float)
    else:
        sc = np.clip(np.asarray(scores, dtype=float).reshape(-1), 0.0, None)
    report = AlignmentReport()
    if len(t) == 0 or len(y) == 0 or len(t) != len(y):
        return report
    offset, slope, line_mask = robust_line(t, y, sc, fit_tol_sec=fit_tol_sec)
    jumps, jump_mask = find_jumps(t, y, sc, offset, slope)
    inlier = line_mask & ~jump_mask
    if int(np.count_nonzero(inlier)) < 2:
        inlier = line_mask if int(np.count_nonzero(line_mask)) else np.ones(len(t), dtype=bool)
    resid = y - (offset + slope * t)
    err = np.abs(resid[inlier])
    weight = sc[inlier]
    if float(sc.sum()) > 0:
        confidence = float(sc[inlier].sum() / sc.sum())
    else:
        confidence = float(np.mean(inlier))
    report.offset_sec = float(offset)
    report.drift = float(slope)
    report.drift_ms_per_min = float(slope * 60_000.0)
    report.nonlinear_rms_sec = float(np.sqrt(np.mean(resid[inlier] ** 2))) if err.size else 0.0
    report.max_local_error_sec = float(err.max()) if err.size else 0.0
    report.median_error_sec = float(np.median(err)) if err.size else 0.0
    report.p95_error_sec = _percentile(err, 95)
    report.n_discontinuities = len(jumps)
    report.weak_frac = float(np.mean(sc < weak_score))
    report.confidence = confidence
    report.jumps = jumps
    report.line_offset = float(offset)
    report.line_slope = float(slope)
    report.times = [float(v) for v in t]
    report.lags = [float(v) for v in y]
    report.confidence_series = [float(v) for v in sc]
    return report
