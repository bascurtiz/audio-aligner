"""Synthetic alignment cases with known lag, and the errors we report on them."""

from __future__ import annotations

import numpy as np

from align_model.decompose import decompose
from align_model.params import resolve_profile
from align_model.pipeline import map_stem, map_vocal_gaps

SR = 22050


def harmonic(sr: int, dur: float, f0: float, amp: float = 0.6) -> np.ndarray:
    n = max(8, int(dur * sr))
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float64)
    for k, gain in enumerate((1.0, 0.55, 0.28, 0.14), start=1):
        y += gain * np.sin(2 * np.pi * f0 * k * t)
    click = int(0.025 * sr)
    env = np.exp(-np.linspace(0.0, 5.0, click))
    step = int(0.5 * sr)
    for i in range(0, n - click, step):
        y[i : i + click] += 0.9 * env
    fade = min(int(0.02 * sr), n // 4)
    if fade > 1:
        y[:fade] *= np.linspace(0.0, 1.0, fade)
        y[-fade:] *= np.linspace(1.0, 0.0, fade)
    peak = float(np.max(np.abs(y))) or 1.0
    return (amp * y / peak).astype(np.float32)


def apply_lag(ref: np.ndarray, sr: int, lag_fn) -> np.ndarray:
    n = len(ref)
    t = np.arange(n) / sr
    src = np.clip(t - lag_fn(t), 0.0, (n - 1) / sr)
    return np.interp(src * sr, np.arange(n), ref).astype(np.float32)


def lag_errors(times: np.ndarray, lags: np.ndarray, truth_fn) -> dict:
    times = np.asarray(times, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if times.size == 0:
        return {"mae": 1e9, "rmse": 1e9, "p95": 1e9, "max": 1e9}
    truth = np.array([float(truth_fn(t)) for t in times], dtype=float)
    err = lags - truth
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p95": float(np.percentile(np.abs(err), 95)),
        "max": float(np.max(np.abs(err))),
    }


def discontinuity_counts(
    detected: list[float],
    truth: list[float],
    *,
    tol: float = 0.75,
) -> tuple[int, int]:
    missed = sum(1 for t in truth if not any(abs(t - d) <= tol for d in detected))
    false = sum(1 for d in detected if not any(abs(d - t) <= tol for t in truth))
    return false, missed


def warp_smoothness(times: np.ndarray, lags: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if times.size < 3:
        return 0.0
    order = np.argsort(times)
    times = times[order]
    lags = lags[order]
    slope = np.gradient(lags, times)
    curve = np.gradient(slope, times)
    return float(np.mean(np.abs(curve)))


def _profile(ref: np.ndarray, *, max_lag: float = 0.8):
    return resolve_profile(
        ref,
        SR,
        name="default",
        win_sec=3.0,
        step_sec=0.75,
        max_lag_sec=max_lag,
    )


def _fit(ref: np.ndarray, qry: np.ndarray, truth_fn, *, max_lag: float = 0.8) -> dict:
    stem = map_stem(
        ref,
        qry,
        SR,
        kind="instrumental",
        profile=_profile(ref, max_lag=max_lag),
        target_sec=len(ref) / SR,
        in_sec=len(qry) / SR,
    )
    errors = lag_errors(stem.times, stem.lags, truth_fn)
    jumps = [float(j.get("time", 0.0)) for j in stem.report.get("jumps") or []]
    src = stem.dst - np.interp(stem.dst, stem.times, stem.lags) if len(stem.times) else stem.src
    mono = bool(len(stem.src) < 2 or np.all(np.diff(stem.src) >= -1e-3))
    return {
        "errors": errors,
        "jumps": jumps,
        "smoothness": warp_smoothness(stem.times, stem.lags),
        "monotonic": mono,
        "offset": float(stem.report.get("offset_sec") or 0.0),
        "drift": float(stem.report.get("drift") or 0.0),
        "report": stem.report,
        "src": src,
    }


def case_perfect() -> dict:
    ref = harmonic(SR, 8.0, 220.0)
    out = _fit(ref, ref.copy(), lambda t: 0.0)
    out["name"] = "perfect"
    return out


def case_offset() -> dict:
    ref = harmonic(SR, 8.0, 196.0)
    delay = 0.16
    qry = apply_lag(ref, SR, lambda t: delay)
    # Positive model lag means the query is early. A delayed query is a negative lag.
    out = _fit(ref, qry, lambda t: -delay, max_lag=0.6)
    out["name"] = "constant_offset"
    out["true_offset"] = -delay
    return out


def case_drift() -> dict:
    ref = harmonic(SR, 8.0, 174.0)
    slope = 0.012
    qry = apply_lag(ref, SR, lambda t: slope * t)
    out = _fit(ref, qry, lambda t: -slope * t, max_lag=0.5)
    out["name"] = "linear_drift"
    out["true_slope"] = -slope
    return out


def case_nonlinear() -> dict:
    ref = harmonic(SR, 8.0, 246.0)

    def delay(t: float) -> float:
        return 0.06 * np.sin(2 * np.pi * t / 8.0)

    qry = apply_lag(ref, SR, delay)
    out = _fit(ref, qry, lambda t: -delay(t), max_lag=0.4)
    out["name"] = "nonlinear_drift"
    return out


def case_gaps() -> dict:
    a = harmonic(SR, 1.8, 196.0)
    b = harmonic(SR, 1.8, 330.0)
    rest = np.zeros(int(1.4 * SR), dtype=np.float32)
    head = np.zeros(int(0.4 * SR), dtype=np.float32)
    ref = np.concatenate([head, a, rest, b, head])
    qry = np.concatenate([a, b])
    stem = map_vocal_gaps(qry, ref, SR, profile=_profile(ref), target_sec=len(ref) / SR)
    inserts = stem.report.get("gap_inserts") or []
    durations = [float(span[1] - span[0]) for span in inserts]
    out = {
        "name": "missing_internal_gaps",
        "inserts": inserts,
        "durations": durations,
        "monotonic": _spans_monotonic(stem.spans),
        "errors": {"mae": 0.0, "rmse": 0.0, "p95": 0.0, "max": 0.0},
        "smoothness": 0.0,
        "jumps": [],
        "offset": float(stem.report.get("offset_sec") or 0.0),
        "drift": float(stem.report.get("drift") or 0.0),
    }
    return out


def _spans_monotonic(spans) -> bool:
    if not spans:
        return False
    for i, span in enumerate(spans):
        if span.dst_end + 1e-4 < span.dst_start:
            return False
        if span.kind == "speech" and span.src_end + 1e-4 < span.src_start:
            return False
        if i and span.dst_start + 1e-3 < spans[i - 1].dst_end:
            return False
    return True


def case_repeated() -> dict:
    phrase = harmonic(SR, 2.2, 220.0)
    gap = np.zeros(int(1.6 * SR), dtype=np.float32)
    ref = np.concatenate([phrase, gap, phrase, gap])
    out = _fit(ref, ref.copy(), lambda t: 0.0, max_lag=1.2)
    out["name"] = "repeated_section"
    return out


def case_tempo_change() -> dict:
    n = int(8.0 * SR)
    t = np.arange(n) / SR
    ref = np.sin(2 * np.pi * (90.0 * t + 12.0 * t * t)).astype(np.float32)

    def delay(t):
        t = np.asarray(t, dtype=float)
        return np.where(t < 4.0, 0.0, 0.18)

    qry = apply_lag(ref, SR, delay)
    out = _fit(ref, qry, lambda t: -delay(t), max_lag=0.5)
    out["name"] = "tempo_change"
    out["true_jumps"] = [4.0]
    return out


def case_noise() -> dict:
    rng = np.random.default_rng(2)
    ref = harmonic(SR, 8.0, 208.0)
    noisy = ref + (0.04 * rng.normal(size=len(ref))).astype(np.float32)
    delay = 0.14
    qry = apply_lag(noisy, SR, lambda t: delay)
    out = _fit(ref, qry, lambda t: -delay, max_lag=0.5)
    out["name"] = "noise"
    return out


def case_sparse() -> dict:
    sr = SR
    silence = np.zeros(int(2.2 * sr), dtype=np.float32)
    burst = harmonic(sr, 0.9, 392.0)
    ref = np.concatenate([silence, burst, silence, burst, silence[: int(1.0 * sr)]])
    delay = 0.12
    qry = apply_lag(ref, SR, lambda t: delay)
    out = _fit(ref, qry, lambda t: -delay, max_lag=0.5)
    out["name"] = "sparse_vocal"
    return out


CASES = (
    case_perfect,
    case_offset,
    case_drift,
    case_nonlinear,
    case_gaps,
    case_repeated,
    case_tempo_change,
    case_noise,
    case_sparse,
)


def run_all() -> list[dict]:
    return [fn() for fn in CASES]


def measure_curve(times: np.ndarray, lags: np.ndarray, scores: np.ndarray | None = None) -> dict:
    """Phase-1 measurement used by the math tests. Does not build a warp."""
    report = decompose(times, lags, scores)
    return report.as_dict()


def main() -> None:
    for row in run_all():
        err = row.get("errors") or {}
        print(
            f"{row['name']:<24} mae={err.get('mae', float('nan')):.3f} "
            f"rmse={err.get('rmse', float('nan')):.3f} "
            f"p95={err.get('p95', float('nan')):.3f} "
            f"max={err.get('max', float('nan')):.3f} "
            f"mono={row.get('monotonic')}"
        )


if __name__ == "__main__":
    main()
