"""Track profiles that choose windows, weights, and marker density."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = ("chroma", "onset", "spectral", "waveform")


@dataclass
class TrackProfile:
    name: str
    max_pad_sec: float
    win_sec: float
    step_sec: float
    max_lag_sec: float
    weights: dict[str, float]
    curvature_gain: float
    min_marker_sec: float = 2.0
    max_marker_sec: float = 30.0
    weak_score: float = 0.35
    fit_tol_sec: float = 0.02
    high_conf: float = 0.45
    high_margin: float = 0.12
    medium_conf: float = 0.25
    medium_margin: float = 0.04

    def weight_map(self) -> dict[str, float]:
        return {name: float(self.weights.get(name, 0.0)) for name in FEATURE_NAMES}


def _profiles() -> dict[str, TrackProfile]:
    return {
        "default": TrackProfile(
            "default",
            max_pad_sec=90.0,
            win_sec=8.0,
            step_sec=1.0,
            max_lag_sec=1.5,
            weights={"chroma": 0.30, "onset": 0.25, "spectral": 0.15, "waveform": 0.30},
            curvature_gain=1.0,
            fit_tol_sec=0.020,
        ),
        "edm": TrackProfile(
            "edm",
            max_pad_sec=30.0,
            win_sec=6.0,
            step_sec=0.5,
            max_lag_sec=1.0,
            weights={"chroma": 0.15, "onset": 0.40, "spectral": 0.15, "waveform": 0.30},
            curvature_gain=1.25,
            min_marker_sec=1.0,
            max_marker_sec=16.0,
            fit_tol_sec=0.020,
        ),
        "acoustic": TrackProfile(
            "acoustic",
            max_pad_sec=90.0,
            win_sec=10.0,
            step_sec=1.0,
            max_lag_sec=2.0,
            weights={"chroma": 0.35, "onset": 0.15, "spectral": 0.20, "waveform": 0.30},
            curvature_gain=0.85,
            max_marker_sec=40.0,
            fit_tol_sec=0.030,
        ),
        "sparse_vocal": TrackProfile(
            "sparse_vocal",
            max_pad_sec=90.0,
            win_sec=8.0,
            step_sec=1.0,
            max_lag_sec=2.0,
            weights={"chroma": 0.40, "onset": 0.05, "spectral": 0.15, "waveform": 0.40},
            curvature_gain=0.9,
            fit_tol_sec=0.018,
        ),
    }


PROFILES = _profiles()

# The GUI and CLI send this on every run. It matches the default profile,
# so it is not treated as an override. A value the user changed still wins.
# Correlation and drift gates are pass/fail policy, not model parameters.
APP_MAX_PAD_SEC = 90.0


def _explicit(value: float | None, stock: float) -> float | None:
    if value is None:
        return None
    if abs(float(value) - float(stock)) <= 1e-6:
        return None
    return float(value)


def classify_track(y: np.ndarray, sr: int) -> str:
    """Cheap onset density, duty cycle, and spectral flatness."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr <= 0 or y.size < int(sr * 1.5):
        return "default"
    if y.size > int(sr * 45):
        start = (y.size - int(sr * 45)) // 2
        y = y[start : start + int(sr * 45)]
    import librosa

    hop = 512
    flat = librosa.feature.spectral_flatness(y=y)
    flatness = float(np.mean(flat)) if flat.size else 0.0
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if onset.size < 3:
        return "default"
    med = float(np.median(onset))
    peaks = (
        (onset[1:-1] > onset[:-2])
        & (onset[1:-1] >= onset[2:])
        & (onset[1:-1] > max(med * 1.8, 1e-4))
    )
    density = float(np.count_nonzero(peaks) / (len(y) / sr))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    duty = float(np.mean(rms > max(float(np.median(rms)) * 0.85, 1e-5)))
    if density >= 2.8 and flatness >= 0.04:
        return "edm"
    if duty < 0.38:
        return "sparse_vocal"
    if density < 1.4 and flatness < 0.25:
        return "acoustic"
    return "default"


def resolve_profile(
    y: np.ndarray | None,
    sr: int,
    *,
    name: str | None = None,
    max_pad_sec: float | None = None,
    win_sec: float | None = None,
    step_sec: float | None = None,
    max_lag_sec: float | None = None,
    curvature_gain: float | None = None,
    weights: dict[str, float] | None = None,
) -> TrackProfile:
    """Pick a profile from the audio, then apply explicit overrides."""
    chosen = name or (classify_track(y, sr) if y is not None and sr > 0 else "default")
    if chosen not in PROFILES:
        chosen = "default"
    base = PROFILES[chosen]
    merged = dict(base.weights)
    if weights:
        merged.update({k: float(v) for k, v in weights.items() if k in FEATURE_NAMES})
    max_pad_sec = _explicit(max_pad_sec, APP_MAX_PAD_SEC)
    return TrackProfile(
        name=base.name if name is None else chosen,
        max_pad_sec=base.max_pad_sec if max_pad_sec is None else float(max_pad_sec),
        win_sec=base.win_sec if win_sec is None else float(win_sec),
        step_sec=base.step_sec if step_sec is None else float(step_sec),
        max_lag_sec=base.max_lag_sec if max_lag_sec is None else float(max_lag_sec),
        weights=merged,
        curvature_gain=base.curvature_gain if curvature_gain is None else float(curvature_gain),
        min_marker_sec=base.min_marker_sec,
        max_marker_sec=base.max_marker_sec,
        weak_score=base.weak_score,
        fit_tol_sec=base.fit_tol_sec,
        high_conf=base.high_conf,
        high_margin=base.high_margin,
        medium_conf=base.medium_conf,
        medium_margin=base.medium_margin,
    )
