"""Certainty map, constrained DTW, and curvature-adaptive warp markers."""

from __future__ import annotations

import numpy as np

from align_model.beats import high_margin, safe_lag
from align_model.decompose import AlignmentReport
from align_model.evidence import HIGH_MARGIN, AlignmentEvidence
from align_model.params import TrackProfile

MAX_STRETCH = 0.08
MAX_ACCEL = 0.02
BEAT_WEIGHT = 0.35
BAND_HIGH_SEC = 0.50
BAND_MEDIUM_SEC = 0.15
BAND_LOW_SEC = 0.05


def _band_half(confidence: float, margin: float, *, high_conf: float, high_margin_min: float, medium_conf: float, medium_margin: float) -> float:
    if high_margin(margin, confidence, margin_min=high_margin_min, conf_min=high_conf):
        return BAND_HIGH_SEC
    if confidence >= medium_conf and margin >= medium_margin:
        return BAND_MEDIUM_SEC
    return BAND_LOW_SEC


def certainty_lags(
    times: np.ndarray,
    lags: np.ndarray,
    confidence: np.ndarray,
    margins: np.ndarray,
    offset: float,
    slope: float,
    *,
    high_conf: float = 0.45,
    high_margin_min: float = HIGH_MARGIN,
    medium_conf: float = 0.25,
    medium_margin: float = 0.04,
) -> tuple[np.ndarray, list[str]]:
    """High points keep the local lag. Low points sit on the robust line.

    Medium points are filled from the high neighbors around them.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(lags, dtype=float).copy()
    conf = np.asarray(confidence, dtype=float)
    margin = np.asarray(margins, dtype=float)
    labels: list[str] = []
    for i in range(len(y)):
        if i < len(conf) and i < len(margin) and high_margin(margin[i], conf[i], margin_min=high_margin_min, conf_min=high_conf):
            labels.append("high")
        elif i < len(conf) and conf[i] >= medium_conf and i < len(margin) and margin[i] >= medium_margin:
            labels.append("medium")
        else:
            labels.append("low")
            y[i] = float(offset + slope * t[i])
    high_idx = [i for i, label in enumerate(labels) if label == "high"]
    original = np.asarray(lags, dtype=float)
    for i, label in enumerate(labels):
        if label != "medium":
            continue
        prev = max((h for h in high_idx if h < i), default=None)
        nxt = min((h for h in high_idx if h > i), default=None)
        if prev is not None and nxt is not None and t[nxt] > t[prev]:
            w = (t[i] - t[prev]) / (t[nxt] - t[prev])
            y[i] = float(original[prev] * (1.0 - w) + original[nxt] * w)
        elif prev is not None:
            y[i] = float(original[prev])
        elif nxt is not None:
            y[i] = float(original[nxt])
        else:
            y[i] = float(offset + slope * t[i])
            labels[i] = "low"
    return y, labels


def spans_where(times: np.ndarray, labels: list[str], want: str) -> list[list[float]]:
    spans: list[list[float]] = []
    start = None
    t = np.asarray(times, dtype=float)
    for i, label in enumerate(labels):
        if label == want and start is None:
            start = float(t[i])
        elif label != want and start is not None:
            spans.append([start, float(t[i])])
            start = None
    if start is not None and len(t):
        spans.append([start, float(t[-1])])
    return spans


def solve_constrained_lag_path(
    times: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    margins: np.ndarray,
    offset: float,
    slope: float,
    *,
    beats: np.ndarray | None = None,
    downbeats: np.ndarray | None = None,
    beat_weight: float = 0.0,
    beat_scores: np.ndarray | None = None,
    query_events: np.ndarray | None = None,
    max_stretch: float = MAX_STRETCH,
    max_accel: float = MAX_ACCEL,
    high_conf: float = 0.45,
    high_margin_min: float = HIGH_MARGIN,
    medium_conf: float = 0.25,
    medium_margin: float = 0.04,
) -> np.ndarray:
    """Monotonic lag path inside a confidence-sized band around the robust line.

    A jump is allowed only when the margin is high and the step is large.
    Ambiguous points are pulled onto the beat-snapped line.
    """
    t = np.asarray(times, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    n = len(t)
    if n == 0:
        return target
    if n == 1:
        return target.copy()
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    margin = np.asarray(margins, dtype=float).reshape(-1)
    if len(conf) != n:
        conf = np.resize(conf, n)
    if len(margin) != n:
        margin = np.resize(margin, n)
    line = offset + slope * t
    beats = np.zeros(0) if beats is None else np.asarray(beats, dtype=float)
    downbeats = np.zeros(0) if downbeats is None else np.asarray(downbeats, dtype=float)
    events = np.zeros(0) if query_events is None else np.asarray(query_events, dtype=float)
    has_beats = beat_weight > 0 and beats.size >= 2
    scores = np.zeros(n) if beat_scores is None else np.asarray(beat_scores, dtype=float).reshape(-1)
    if len(scores) != n:
        scores = np.resize(scores, n)
    pull = target.copy()
    beat_w = np.zeros(n, dtype=float)
    for i in range(n):
        if high_margin(float(margin[i]), float(conf[i]), margin_min=high_margin_min, conf_min=high_conf):
            pull[i] = target[i]
            continue
        on_beat = float(scores[i]) >= 0.70
        if on_beat:
            pull[i] = target[i]
            beat_w[i] = beat_weight * 0.25
            continue
        if has_beats and float(margin[i]) < high_margin_min:
            snapped, mult = safe_lag(
                float(t[i]),
                float(line[i]),
                query_events=events,
                beats=beats,
                downbeats=downbeats,
            )
            pull[i] = snapped
            beat_w[i] = beat_weight * mult * (1.0 - 0.5 * float(scores[i]))
        else:
            pull[i] = float(line[i]) if float(conf[i]) < medium_conf else target[i]

    lo = float(min(pull.min(), line.min(), target.min()) - 0.05)
    hi = float(max(pull.max(), line.max(), target.max()) + 0.05)
    span = max(hi - lo, 0.05)
    step = max(0.005, span / 80.0)
    grid = np.arange(lo, hi + step * 0.5, step)
    nb = len(grid)
    inf = 1e9
    dp = np.full((n, nb), inf)
    prev = np.full((n, nb), -1, dtype=int)
    scale = 0.02

    def data_cost(i: int, lag: float) -> float:
        c = float(conf[i])
        cost = ((lag - float(pull[i])) / scale) ** 2 * (0.25 + c)
        if beat_w[i] > 0:
            cost += float(beat_w[i]) * ((lag - float(pull[i])) / scale) ** 2
        return float(cost)

    def band_mask(i: int) -> np.ndarray:
        half = _band_half(
            float(conf[i]),
            float(margin[i]),
            high_conf=high_conf,
            high_margin_min=high_margin_min,
            medium_conf=medium_conf,
            medium_margin=medium_margin,
        )
        lo_i = float(line[i]) - half
        hi_i = float(line[i]) + half
        if high_margin(float(margin[i]), float(conf[i]), margin_min=high_margin_min, conf_min=high_conf):
            lo_i = min(lo_i, float(target[i]) - 0.03)
            hi_i = max(hi_i, float(target[i]) + 0.03)
        return (grid >= lo_i) & (grid <= hi_i)

    first = data_cost_row(0, grid, pull[0], conf[0], beat_w[0], scale)
    inside = band_mask(0)
    if np.any(inside):
        first = np.where(inside, first, inf)
    dp[0] = first

    for i in range(1, n):
        dt = max(float(t[i] - t[i - 1]), 1e-3)
        max_dlag = max_stretch * dt
        lag_i = grid
        lag_prev = grid
        dlag = lag_i[:, None] - lag_prev[None, :]
        mono = dlag <= dt - 1e-3
        smooth = np.abs(dlag) <= max_dlag + 1e-9
        trans = np.where(mono & smooth, 0.0, inf)
        if float(margin[i]) >= HIGH_MARGIN:
            jump = mono & (np.abs(dlag) >= 0.04) & ~smooth
            trans = np.where(jump, 0.5 * np.abs(dlag) / scale, trans)
        costs = data_cost_row(i, grid, pull[i], conf[i], beat_w[i], scale)
        inside = band_mask(i)
        if np.any(inside):
            costs = np.where(inside, costs, inf)
        total = trans + dp[i - 1][None, :]
        best_p = np.argmin(total, axis=1)
        best = total[np.arange(nb), best_p]
        dead = best >= inf / 2
        if np.any(dead):
            # Keep the path monotonic even when the stretch band is empty.
            relaxed = np.where(mono, 0.15 * np.abs(dlag) / scale, inf)
            total2 = relaxed + dp[i - 1][None, :]
            best_p2 = np.argmin(total2, axis=1)
            best2 = total2[np.arange(nb), best_p2]
            best_p = np.where(dead, best_p2, best_p)
            best = np.where(dead, best2, best)
        dp[i] = best + costs
        prev[i] = best_p

    b = int(np.argmin(dp[-1]))
    out = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        out[i] = float(grid[b])
        if i == 0:
            break
        b = int(prev[i, b])
        if b < 0:
            b = int(np.argmin(np.abs(grid - pull[i - 1])))
    return _clamp_accel(t, out, max_accel)


def constrained_dtw(*args, **kwargs):
    """Older name. The solver is a constrained lag path, not feature DTW."""
    return solve_constrained_lag_path(*args, **kwargs)


def data_cost_row(i, grid, pull, conf, beat_w, scale) -> np.ndarray:
    cost = ((grid - float(pull)) / scale) ** 2 * (0.25 + float(conf))
    if beat_w > 0:
        cost = cost + float(beat_w) * ((grid - float(pull)) / scale) ** 2
    return cost


def _clamp_accel(times: np.ndarray, lags: np.ndarray, max_step: float) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    lag = np.asarray(lags, dtype=float).copy()
    if len(t) < 3:
        return _monotonic_lag(t, lag)
    src = t - lag
    rates = np.diff(src) / np.maximum(np.diff(t), 1e-6)
    dlag = np.diff(lag)
    for i in range(1, len(rates)):
        if abs(float(dlag[i])) >= 0.04 or abs(float(dlag[i - 1])) >= 0.04:
            continue
        delta = float(rates[i] - rates[i - 1])
        if abs(delta) > max_step:
            rates[i] = rates[i - 1] + np.sign(delta) * max_step
    src_out = np.empty_like(src)
    src_out[0] = src[0]
    for i, rate in enumerate(rates):
        src_out[i + 1] = src_out[i] + float(rate) * float(t[i + 1] - t[i])
    for i in range(1, len(src_out)):
        if abs(float(lag[i] - lag[i - 1])) >= 0.04:
            src_out[i] = src[i]
        src_out[i] = max(float(src_out[i]), float(src_out[i - 1]) + 1e-4)
    return t - src_out


def _monotonic_lag(times: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """src = t - lag must not run backwards."""
    t = np.asarray(times, dtype=float)
    lag = np.asarray(lags, dtype=float).copy()
    if len(t) < 2:
        return lag
    src = t - lag
    for i in range(1, len(src)):
        src[i] = max(float(src[i]), float(src[i - 1]) + 1e-4)
    return t - src


def adaptive_marker_times(
    times: np.ndarray,
    lags: np.ndarray,
    *,
    min_spacing: float = 2.0,
    max_spacing: float = 30.0,
    curvature_gain: float = 1.0,
    jump_times: list[float] | None = None,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    """Denser pins where confident curvature is large. A weak bend stays sparse."""
    t = np.asarray(times, dtype=float).reshape(-1)
    y = np.asarray(lags, dtype=float).reshape(-1)
    if len(t) < 2 or len(y) != len(t):
        return t.copy()
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    if scores is None or len(scores) != len(t):
        conf = np.ones(len(t), dtype=float)
    else:
        conf = np.clip(np.asarray(scores, dtype=float).reshape(-1)[order], 0.0, 1.0)
    d1 = np.gradient(y, t)
    d2 = np.abs(np.gradient(d1, t)) * float(curvature_gain)
    thresh = 0.0015 / max(float(curvature_gain), 0.15)
    marks = [float(t[0])]
    acc = 0.0
    last = float(t[0])
    jumps = [float(v) for v in (jump_times or [])]
    min_spacing = max(0.25, float(min_spacing))
    max_spacing = max(min_spacing, float(max_spacing))
    for i in range(1, len(t)):
        acc += float(d2[i]) * float(t[i] - t[i - 1]) * float(conf[i])
        span = float(t[i] - last)
        near_jump = any(abs(float(t[i]) - jt) <= 0.75 for jt in jumps)
        if span >= max_spacing or (span >= min_spacing and (acc >= thresh or near_jump)):
            marks.append(float(t[i]))
            last = float(t[i])
            acc = 0.0
    if float(t[-1]) - marks[-1] > 1e-3:
        marks.append(float(t[-1]))
    return np.asarray(marks, dtype=float)


def src_dst_from_lags(
    times: np.ndarray,
    lags: np.ndarray,
    *,
    target_sec: float,
    in_sec: float,
    min_spacing: float,
    max_spacing: float,
    curvature_gain: float,
    jump_times: list[float] | None = None,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """src = dst - lag, monotonic, with the 2% rate-step clamp."""
    t = np.asarray(times, dtype=float).reshape(-1)
    lag = np.asarray(lags, dtype=float).reshape(-1)
    target_sec = max(0.0, float(target_sec))
    in_sec = max(0.0, float(in_sec))
    if len(t) < 2 or target_sec <= 0:
        return np.array([0.0, in_sec]), np.array([0.0, target_sec])
    dst = adaptive_marker_times(
        t,
        lag,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
        curvature_gain=curvature_gain,
        jump_times=jump_times,
        scores=scores,
    )
    dst = dst[(dst >= -1e-6) & (dst <= target_sec + 1e-6)]
    if len(dst) == 0 or dst[0] > 1e-4:
        dst = np.insert(dst, 0, 0.0)
    dst = np.unique(np.clip(dst, 0.0, target_sec))
    if len(dst) < 2:
        dst = np.array([0.0, target_sec])
    lag_at = np.interp(dst, t, lag, left=float(lag[0]), right=float(lag[-1]))
    src = dst - lag_at
    src[0] = 0.0
    for i in range(1, len(src)):
        src[i] = max(float(src[i]), float(src[i - 1]) + 1e-3)
    rates = np.diff(src) / np.maximum(np.diff(dst), 1e-6)
    for i in range(1, len(rates)):
        delta = float(rates[i] - rates[i - 1])
        if abs(delta) > MAX_ACCEL:
            rates[i] = rates[i - 1] + np.sign(delta) * MAX_ACCEL
    src = np.empty_like(dst)
    src[0] = 0.0
    for i, rate in enumerate(rates):
        src[i + 1] = src[i] + float(rate) * float(dst[i + 1] - dst[i])
    for i in range(1, len(src)):
        src[i] = max(float(src[i]), float(src[i - 1]) + 1e-3)
    keep = [i for i in range(len(src)) if i == 0 or src[i] <= in_sec + 1e-3]
    if len(keep) < 2:
        return np.array([0.0, min(in_sec, target_sec)]), np.array([0.0, min(in_sec, target_sec)])
    return src[keep], dst[keep]


def solve_time_map(
    evidence: AlignmentEvidence,
    report: AlignmentReport,
    profile: TrackProfile,
    *,
    beats: np.ndarray | None = None,
    downbeats: np.ndarray | None = None,
    target_sec: float,
    in_sec: float,
    beat_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return times, mapped lags (full, including offset), src, dst.

    Warp callers subtract ``report.offset_sec`` and pass that residual with a
    front pad. The returned lags are the ones to draw.
    """
    times = evidence.times
    if len(times) < 2:
        src = np.array([0.0, max(in_sec, 0.0)])
        dst = np.array([0.0, max(target_sec, 0.0)])
        report.markers = [float(v) for v in dst]
        return times, evidence.lags, src, dst
    target, labels = certainty_lags(
        times,
        evidence.lags,
        evidence.scores,
        evidence.margins,
        report.offset_sec,
        report.drift,
        high_conf=profile.high_conf,
        high_margin_min=profile.high_margin,
        medium_conf=profile.medium_conf,
        medium_margin=profile.medium_margin,
    )
    weight = 0.0 if beat_weight is None else float(beat_weight)
    if weight <= 0 and beats is not None and len(np.asarray(beats)) >= 4:
        weight = BEAT_WEIGHT
    mapped = solve_constrained_lag_path(
        times,
        target,
        evidence.scores,
        evidence.margins,
        report.offset_sec,
        report.drift,
        beats=beats,
        downbeats=downbeats,
        beat_weight=weight,
        beat_scores=np.asarray([point.beat_score for point in evidence.points], dtype=float),
        query_events=np.asarray(getattr(evidence, "query_onsets", []), dtype=float),
        high_conf=profile.high_conf,
        high_margin_min=profile.high_margin,
        medium_conf=profile.medium_conf,
        medium_margin=profile.medium_margin,
    )
    mapped = _monotonic_lag(times, mapped)
    jump_times = [j.time if hasattr(j, "time") else float(j["time"]) for j in report.jumps]
    src, dst = src_dst_from_lags(
        times,
        mapped - report.offset_sec,
        target_sec=target_sec,
        in_sec=in_sec,
        min_spacing=profile.min_marker_sec,
        max_spacing=profile.max_marker_sec,
        curvature_gain=profile.curvature_gain,
        jump_times=jump_times,
        scores=evidence.scores,
    )
    report.times = [float(v) for v in times]
    report.lags = [float(v) for v in mapped]
    report.confidence_series = [float(v) for v in evidence.scores]
    report.low_conf_spans = spans_where(times, labels, "low")
    report.markers = [float(v) for v in dst]
    report.profile = profile.name
    if beats is not None and len(beats):
        report.beats = [float(v) for v in np.asarray(beats, dtype=float) if 0 <= v <= target_sec + 1]
    if downbeats is not None and len(downbeats):
        report.downbeats = [float(v) for v in np.asarray(downbeats, dtype=float) if 0 <= v <= target_sec + 1]
    return times, mapped, src, dst
