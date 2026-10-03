#!/usr/bin/env python3
"""Check whether acapella+instrumental mixes align with the original song.

For each song subfolder: classify stems, mix acapella+instrumental in memory,
compare temporal alignment (volume-agnostic) to the original, then rename the
folder with _[pass] or _[fail].
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(
    r"T:\SDR30+\Pair-finder organized done\!2-files\with_original\to-review"
)
DEFAULT_SR = 22050
DEFAULT_MAX_SHIFT_SEC = 12.0
DEFAULT_WINDOW_SEC = 20.0
DEFAULT_CORR_MIN = 0.35
# Maximum |offset| at any checkpoint, and the largest step allowed between
# checkpoints. Past this the stem is flagged.
# <=3 ms excellent, <=5 ms good, <=10 ms acceptable, <=20 ms suspect, >20 ms bad.
# 10–20 ms is still reported as suspect and does not fail the folder.
DEFAULT_DRIFT_MS = 20.0
DEFAULT_WINDOW_CORR_MIN = 0.25
DEFAULT_WEAK_WINDOW_FRAC = 0.35
# Per-point gate. Chroma lag is quantized to ~23 ms, so this is applied to
# sample-accurate checkpoints, not to that coarse lag.
DEFAULT_STEM_LAG_MAX_SEC = DEFAULT_DRIFT_MS / 1000.0
# Coarse chroma search. One frame is coarser than the perceptual gate, so only
# an offset beyond the fine search window fails here.
DEFAULT_COARSE_LAG_MAX_SEC = 0.050
# Clock used when the original has no trustworthy 8-bar line. The pass test
# otherwise starts each window on the same downbeats as the stretch markers.
CHECKPOINT_EVERY_SEC = 30.0
CHECKPOINT_WIN_SEC = 8.0
CHECKPOINT_MAX_LAG_SEC = 0.040
# A drum groove always has a hit inside ±40 ms. Look out to 1 s, and keep
# that farther peak when it matches more strongly than the nearest hat.
CHECKPOINT_CONFIRM_LAG_SEC = 1.0
CHECKPOINT_CONFIRM_GAIN = 1.12
# Second look when a checkpoint sits on the ±40 ms edge. The pass test stays
# at 40 ms. The repair warp needs the real offset, still from onsets.
REPAIR_WIDE_LAG_SEC = 0.200
HOP_LENGTH = 512
TAG_RE = re.compile(r"_\[(pass|fail)\]$", re.IGNORECASE)
AUDIO_EXTS = {".flac", ".wav", ".mp3", ".m4a", ".aiff", ".aif", ".ogg"}

ORIGINAL_MARKERS = ("original song",)
ACA_MARKERS = (
    "acapella",
    "a cappella",
    "a-cappella",
    "vocals",
    "vocal",
    "vox",
    "lead vocal",
    "lead vocals",
    "backing vocal",
    "backing vocals",
    "bgv",
)
INST_MARKERS = ("instrumental", "inst", "karaoke")


@dataclass
class CheckResult:
    folder: str
    verdict: str  # pass | fail | skip | error
    corr: float = 0.0
    lag_sec: float = 0.0
    drift_ms: float = 0.0
    weak_window_frac: float = 0.0
    acapella: str = ""
    instrumental: str = ""
    original: str = ""
    notes: str = ""
    renamed_to: str = ""
    aca_verdict: str = ""
    inst_verdict: str = ""
    aca_corr: float = 0.0
    inst_corr: float = 0.0
    aca_check_drift_ms: float = 0.0
    inst_check_drift_ms: float = 0.0
    aca_lag_sec: float = 0.0
    inst_lag_sec: float = 0.0
    # [time_sec, lag_ms, search_edge]. None means this result never measured them.
    aca_checkpoints: list | None = None
    inst_checkpoints: list | None = None
    alignment_report: dict | None = None


def _peak_norm(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-9:
        return y.astype(np.float32, copy=False)
    return (y / peak).astype(np.float32)


def _match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(a), len(b))
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))
    return a, b


def classify_audio_file(path: Path) -> str | None:
    if path.suffix.lower() not in AUDIO_EXTS:
        return None
    stem = path.stem.lower()
    if "(original song)" in stem:
        return "original"
    # Prefer parenthetical role tags when present
    for m in re.finditer(r"\(([^)]+)\)", stem):
        role = m.group(1).strip().lower()
        if role in ORIGINAL_MARKERS or role == "original":
            return "original"
        if any(role == mk or role.endswith(f" {mk}") or mk in role for mk in ACA_MARKERS):
            # Avoid treating pure instrumental tags as vocals
            if any(ik in role for ik in ("instrumental", "karaoke")) and not any(
                ak in role for ak in ("acapella", "a cappella", "vocal", "vox")
            ):
                return "instrumental"
            return "acapella"
        if any(role == mk or role.endswith(f" {mk}") or mk in role for mk in INST_MARKERS):
            return "instrumental"
    lower = stem
    if any(f"({m})" in lower for m in ORIGINAL_MARKERS):
        return "original"
    if any(m in lower for m in ("acapella", "a cappella", "a-cappella")):
        return "acapella"
    if re.search(r"\b(vocals?|vox|lead vocal|backing vocal)\b", lower):
        return "acapella"
    if any(m in lower for m in ("instrumental", "karaoke")) or re.search(r"\binst\b", lower):
        return "instrumental"
    return None


def unclassified_audio(folder: Path) -> list[Path]:
    """Audio files in a folder with no acapella, instrumental, or original tag."""
    found: list[Path] = []
    if not folder.is_dir():
        return found
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        if classify_audio_file(path) is None:
            found.append(path)
    return found


def scan_folder(folder: Path) -> tuple[Path | None, Path | None, Path | None, str]:
    """Return (acapella, instrumental, original, notes)."""
    roles: dict[str, list[Path]] = {"acapella": [], "instrumental": [], "original": []}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        role = classify_audio_file(path)
        if role in roles:
            roles[role].append(path)

    notes_parts: list[str] = []
    picked: dict[str, Path | None] = {}
    for role, paths in roles.items():
        if not paths:
            picked[role] = None
        else:
            if len(paths) > 1:
                notes_parts.append(f"multi_{role}={len(paths)}")
            picked[role] = paths[0]

    if picked["original"] is None:
        loose = unclassified_audio(folder)
        if len(loose) == 1:
            picked["original"] = loose[0]
            notes_parts.append("original_assumed")

    return (
        picked["acapella"],
        picked["instrumental"],
        picked["original"],
        "; ".join(notes_parts),
    )


def _load_mono(path: Path, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _chroma(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    if y.size < sr // 4:
        return np.zeros((12, 1), dtype=np.float32)
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    # L2-normalize frames for volume invariance
    norms = np.linalg.norm(C, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return (C / norms).astype(np.float32)


def _onset_env(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    if y.size < sr // 4:
        return np.zeros(1, dtype=np.float32)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    env = env.astype(np.float32)
    env -= float(np.mean(env))
    std = float(np.std(env))
    if std > 1e-9:
        env /= std
    return env


def _xcorr_best_lag(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    max_lag_frames: int,
) -> tuple[int, float]:
    """Return (best_lag_frames, normalized_peak). Positive lag => query delayed vs ref."""
    from scipy.signal import correlate

    if reference.ndim == 1:
        ref = reference
        qry = query
    else:
        # chroma: flatten time by summing energy-weighted similarity via mean over bins
        # Use frame-wise mean of elementwise product via 1D summary (spectral flux proxy):
        # better: correlate mean-subtracted frame energy of cosine sim by treating as
        # multi-channel — use mean across chroma bins of each frame as 1D proxy, PLUS
        # full matrix via summing per-bin correlations.
        ref_1d = np.mean(reference, axis=0)
        qry_1d = np.mean(query, axis=0)
        # Also use chroma energy
        ref = ref_1d - float(np.mean(ref_1d))
        qry = qry_1d - float(np.mean(qry_1d))

    if ref.size < 4 or qry.size < 4:
        return 0, 0.0

    corr = correlate(ref, qry, mode="full", method="fft")
    lags = np.arange(-len(qry) + 1, len(ref))
    valid = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
    if not np.any(valid):
        return 0, 0.0

    corr_v = corr[valid]
    lags_v = lags[valid]
    best_i = int(np.argmax(corr_v))
    best_lag = int(lags_v[best_i])
    peak = float(corr_v[best_i])

    # Normalize by energy so score is roughly in [-1, 1]
    denom = float(np.linalg.norm(ref) * np.linalg.norm(qry))
    score = peak / denom if denom > 1e-12 else 0.0
    return best_lag, score


def _chroma_xcorr_best_lag(
    ref_chroma: np.ndarray,
    qry_chroma: np.ndarray,
    *,
    max_lag_frames: int,
) -> tuple[int, float]:
    """Cross-correlate chroma matrices (sum of per-bin correlations)."""
    from scipy.signal import correlate

    if ref_chroma.shape[0] != qry_chroma.shape[0]:
        return 0, 0.0
    n_bins = ref_chroma.shape[0]
    # Build full correlation by summing bins
    corr_sum = None
    ref_energy = 0.0
    qry_energy = 0.0
    for b in range(n_bins):
        ref = ref_chroma[b] - float(np.mean(ref_chroma[b]))
        qry = qry_chroma[b] - float(np.mean(qry_chroma[b]))
        ref_energy += float(np.dot(ref, ref))
        qry_energy += float(np.dot(qry, qry))
        c = correlate(ref, qry, mode="full", method="fft")
        corr_sum = c if corr_sum is None else corr_sum + c

    assert corr_sum is not None
    lags = np.arange(-qry_chroma.shape[1] + 1, ref_chroma.shape[1])
    valid = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
    if not np.any(valid):
        return 0, 0.0

    corr_v = corr_sum[valid]
    lags_v = lags[valid]
    best_i = int(np.argmax(corr_v))
    best_lag = int(lags_v[best_i])
    peak = float(corr_v[best_i])
    denom = float(np.sqrt(ref_energy * qry_energy))
    score = peak / denom if denom > 1e-12 else 0.0
    return best_lag, score


def _shift_chroma(chroma: np.ndarray, lag_frames: int) -> np.ndarray:
    """Shift query chroma so it aligns with reference (positive lag = delay query)."""
    n = chroma.shape[1]
    if lag_frames == 0:
        return chroma
    out = np.zeros_like(chroma)
    if lag_frames > 0:
        # query starts later: pad left
        src = min(n, n - lag_frames)
        if src > 0:
            out[:, lag_frames : lag_frames + src] = chroma[:, :src]
    else:
        # query starts earlier: trim left
        shift = -lag_frames
        src = min(n, n - shift)
        if src > 0:
            out[:, :src] = chroma[:, shift : shift + src]
    return out


def _window_lags(
    ref_chroma: np.ndarray,
    aligned_qry: np.ndarray,
    *,
    sr: int,
    window_sec: float,
    local_max_lag_sec: float = 0.5,
) -> tuple[list[float], list[float]]:
    """Return (lags_sec per window, corr per window) over overlapping timeline."""
    hop_sec = HOP_LENGTH / sr
    win_frames = max(8, int(window_sec / hop_sec))
    local_max = max(2, int(local_max_lag_sec / hop_sec))
    n = min(ref_chroma.shape[1], aligned_qry.shape[1])
    lags_sec: list[float] = []
    corrs: list[float] = []
    step = max(1, win_frames // 2)
    for start in range(0, max(1, n - win_frames + 1), step):
        end = min(n, start + win_frames)
        if end - start < max(8, win_frames // 3):
            continue
        lag_f, score = _chroma_xcorr_best_lag(
            ref_chroma[:, start:end],
            aligned_qry[:, start:end],
            max_lag_frames=local_max,
        )
        # Skip near-silent windows (very low energy)
        ref_e = float(np.mean(ref_chroma[:, start:end] ** 2))
        qry_e = float(np.mean(aligned_qry[:, start:end] ** 2))
        if ref_e < 1e-6 or qry_e < 1e-6:
            continue
        lags_sec.append(lag_f * hop_sec)
        corrs.append(score)
    return lags_sec, corrs


def score_alignment(
    reference: np.ndarray,
    query: np.ndarray,
    *,
    sr: int = DEFAULT_SR,
    max_shift_sec: float = DEFAULT_MAX_SHIFT_SEC,
    window_sec: float = DEFAULT_WINDOW_SEC,
    corr_min: float = DEFAULT_CORR_MIN,
    drift_ms: float = DEFAULT_DRIFT_MS,
    window_corr_min: float = DEFAULT_WINDOW_CORR_MIN,
    weak_window_frac: float = DEFAULT_WEAK_WINDOW_FRAC,
) -> tuple[str, float, float, float, float, str]:
    """Score query against reference. Return (verdict, corr, lag_sec, drift_ms, weak_frac, notes)."""
    ref = _peak_norm(reference)
    qry = _peak_norm(query)
    ref, qry = _match_length(ref, qry)

    mix_chroma = _chroma(qry, sr)
    orig_chroma = _chroma(ref, sr)
    mix_onset = _onset_env(qry, sr)
    orig_onset = _onset_env(ref, sr)

    hop_sec = HOP_LENGTH / sr
    max_lag_frames = int(max_shift_sec / hop_sec)

    lag_chroma, corr_chroma = _chroma_xcorr_best_lag(
        orig_chroma, mix_chroma, max_lag_frames=max_lag_frames
    )
    lag_onset, corr_onset = _xcorr_best_lag(
        orig_onset, mix_onset, max_lag_frames=max_lag_frames
    )

    # Prefer the feature with stronger normalized peak; break ties toward chroma
    if corr_onset > corr_chroma + 0.05:
        best_lag_frames, best_corr = lag_onset, corr_onset
        feature = "onset"
    else:
        best_lag_frames, best_corr = lag_chroma, corr_chroma
        feature = "chroma"

    lag_sec = best_lag_frames * hop_sec
    aligned = _shift_chroma(mix_chroma, best_lag_frames)
    # Also align onset-chosen lag onto chroma for drift check consistency
    if feature == "onset":
        aligned = _shift_chroma(mix_chroma, lag_onset)

    lags, win_corrs = _window_lags(
        orig_chroma, aligned, sr=sr, window_sec=window_sec
    )

    if lags:
        drift = (max(lags) - min(lags)) * 1000.0
        weak = sum(1 for c in win_corrs if c < window_corr_min) / len(win_corrs)
    else:
        drift = 0.0
        weak = 1.0

    notes = f"feature={feature}; windows={len(lags)}"
    reasons: list[str] = []

    if best_corr < corr_min:
        reasons.append(f"low_corr={best_corr:.3f}<{corr_min}")
    if drift > drift_ms:
        reasons.append(f"drift={drift:.1f}ms>{drift_ms}")
    if weak > weak_window_frac:
        reasons.append(f"weak_windows={weak:.2f}>{weak_window_frac}")

    verdict = "fail" if reasons else "pass"
    if reasons:
        notes = notes + "; " + "; ".join(reasons)
    return verdict, best_corr, lag_sec, drift, weak, notes


def offset_grade(abs_ms: float) -> str:
    """Perceptual grade for an instrumental/acapella offset."""
    if abs_ms <= 3.0:
        return "excellent"
    if abs_ms <= 5.0:
        return "good"
    if abs_ms <= 10.0:
        return "acceptable"
    if abs_ms <= 20.0:
        return "suspect"
    return "bad"


def _peak_lag_frames(corr: np.ndarray, mid: int, index: int) -> float:
    """Lag of one correlation peak, in frames relative to ``mid``."""
    delta = 0.0
    if 0 < index < len(corr) - 1:
        y0, y1, y2 = float(corr[index - 1]), float(corr[index]), float(corr[index + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(index - mid) + delta


def _onset_activity_floor(env: np.ndarray) -> float:
    """Level of a real phrase, as a fraction of the loud notes in this stem.

    A window that never reaches this is silence, a tail, or noise. A lag
    measured there is not a timing error.
    """
    if env.size == 0:
        return 1.0
    phrase = float(np.percentile(env, 95))
    return max(1e-3, 0.25 * phrase)


def _window_has_audio(env: np.ndarray, floor: float) -> bool:
    """True when a meaningful stretch of the window reaches the phrase level."""
    if env.size == 0:
        return False
    return float(np.percentile(env, 90)) >= floor


def checkpoint_sample_times(original: Path) -> tuple[np.ndarray | None, str]:
    """8-bar downbeats where each pass window starts.

    Same ruler as the stretch markers. None keeps the 30 s clock.
    """
    try:
        import soundfile as sf
        from beat_phase import track_file
        from warp_align_fail_all import bar_mark_times

        info = sf.info(str(original))
        duration = float(info.frames) / float(info.samplerate)
        if duration <= 0.0:
            return None, "checkpoints=30s"
        _beats, downbeats = track_file(Path(original))
        found = bar_mark_times(downbeats, duration)
    except Exception:  # noqa: BLE001 — a missing bar line keeps the 30 s clock
        return None, "checkpoints=30s"
    if found is None:
        return None, "checkpoints=30s"
    marks, span = found
    return np.asarray(marks, dtype=float), f"checkpoints=8bars; bar_span={span:.1f}s"


def checkpoint_offsets(
    reference: np.ndarray,
    query: np.ndarray,
    sr: int,
    *,
    every_sec: float = CHECKPOINT_EVERY_SEC,
    win_sec: float = CHECKPOINT_WIN_SEC,
    max_lag_sec: float = CHECKPOINT_MAX_LAG_SEC,
    at_times: np.ndarray | None = None,
    confirm_sec: float | None = None,
) -> list[tuple[float, float, bool]]:
    """Sample-accurate offset at points through the song.

    Returns (time_sec, lag_sec, railed). Positive lag means the query is early
    and needs a delay. Railed means the peak sat on the search edge, so the
    true offset is larger than the window that was searched. ``confirm_sec``
    looks past the near window and keeps a farther peak when it matches more
    strongly than the nearest hit. A slow walk that stays inside the limit is
    not the same as a sudden step; both are reported and the caller judges them.
    """
    from scipy.signal import correlate

    n = min(len(reference), len(query))
    if n < int(sr * win_sec):
        return []
    # Onset envelopes, not the raw waveform. A dense mix will lock a raw
    # correlation onto a nearby hat; the listener is hearing the attacks.
    import librosa

    hop = 128
    ref_env = librosa.onset.onset_strength(
        y=np.asarray(reference[:n], dtype=np.float32), sr=sr, hop_length=hop
    ).astype(np.float64)
    qry_env = librosa.onset.onset_strength(
        y=np.asarray(query[:n], dtype=np.float32), sr=sr, hop_length=hop
    ).astype(np.float64)
    hop_sec = hop / sr
    n_frames = min(len(ref_env), len(qry_env))
    ref_env = ref_env[:n_frames]
    qry_env = qry_env[:n_frames]
    win = max(8, int(win_sec / hop_sec))
    max_lag = max(2, int(max_lag_sec / hop_sec))
    every = max(5.0, float(every_sec))
    duration = n_frames * hop_sec
    if at_times is None:
        times = np.arange(every * 0.5, max(duration - win_sec * 0.5, every * 0.5), every)
        if len(times) == 0:
            times = np.array([duration * 0.5])
    else:
        times = np.asarray(at_times, dtype=float)
        times = times[(times >= 0.0) & (times < max(duration, 0.0))]
        if len(times) == 0:
            return []
    ref_floor = _onset_activity_floor(ref_env)
    qry_floor = _onset_activity_floor(qry_env)
    points: list[tuple[float, float, bool]] = []
    for t in times:
        start = int(t / hop_sec)
        end = min(n_frames, start + win)
        if end - start < win // 2:
            continue
        r = ref_env[start:end]
        q = qry_env[start:end]
        if not _window_has_audio(q, qry_floor) or not _window_has_audio(r, ref_floor):
            continue
        r = r - float(np.mean(r))
        q = q - float(np.mean(q))
        corr = correlate(r, q, mode="full", method="fft")
        mid = len(q) - 1
        lo = max(0, mid - max_lag)
        hi = min(len(corr), mid + max_lag + 1)
        window = corr[lo:hi]
        if window.size < 3:
            continue
        near_i = lo + int(np.argmax(window))
        chosen_i = near_i
        chosen_radius = max_lag
        if confirm_sec is not None and float(confirm_sec) > max_lag_sec:
            confirm_lag = max(max_lag + 1, int(float(confirm_sec) / hop_sec))
            clo = max(0, mid - confirm_lag)
            chi = min(len(corr), mid + confirm_lag + 1)
            wide_i = clo + int(np.argmax(corr[clo:chi]))
            near_c = float(corr[near_i])
            wide_c = float(corr[wide_i])
            outside = abs(wide_i - mid) > max_lag
            stronger = wide_c > 0.0 and (
                near_c <= 0.0 or wide_c >= CHECKPOINT_CONFIRM_GAIN * near_c
            )
            if outside and stronger:
                chosen_i = wide_i
                chosen_radius = confirm_lag
        lag_frames = _peak_lag_frames(corr, mid, chosen_i)
        lag_sec = float(lag_frames) * hop_sec
        railed = abs(chosen_i - mid) >= chosen_radius - 1
        center_t = (start + (end - start) / 2) * hop_sec
        points.append((float(center_t), lag_sec, railed))
    return points


def residual_checkpoints(
    reference: np.ndarray,
    query: np.ndarray,
    sr: int,
) -> list[tuple[float, float, bool]]:
    """Onset checkpoints, with a wider search where the ±40 ms window railed.

    A point that still sits on the wide edge stays railed. That lag is the
    search limit, not the true offset, so the repair must not follow it.
    """
    points = checkpoint_offsets(reference, query, sr)
    if not points:
        return []
    railed_at = [
        max(0.0, float(t) - CHECKPOINT_WIN_SEC / 2.0)
        for t, _lag, railed in points
        if railed
    ]
    wide: list[tuple[float, float, bool]] = []
    if railed_at:
        wide = checkpoint_offsets(
            reference,
            query,
            sr,
            at_times=np.array(railed_at, dtype=float),
            max_lag_sec=REPAIR_WIDE_LAG_SEC,
        )
    refined: list[tuple[float, float, bool]] = []
    for i, (t, lag, railed) in enumerate(points):
        if not railed:
            refined.append((float(t), float(lag), False))
            continue
        if not wide:
            refined.append((float(t), float(lag), True))
            continue
        nearest = min(wide, key=lambda item: abs(item[0] - float(t)))
        if abs(nearest[0] - float(t)) > 2.0:
            refined.append((float(t), float(lag), True))
            continue
        refined.append((float(t), float(nearest[1]), bool(nearest[2])))
    return refined


def repair_curve(
    points: list[tuple[float, float, bool]],
    limit_ms: float,
) -> tuple[str, np.ndarray | None, np.ndarray | None, str]:
    """Decide whether one residual warp should run.

    Returns status ``ok``, ``repair``, or ``unresolved``, plus times, lags,
    and a short note. Railed points are omitted: a lag stuck on the search
    edge is not a correction.
    """
    unresolved = sum(1 for _t, _lag, railed in points if railed)
    trusted = [(float(t), float(lag)) for t, lag, railed in points if not railed]
    trusted.sort()
    if not trusted:
        if unresolved:
            return "unresolved", None, None, f"unresolved={unresolved}"
        return "ok", None, None, ""
    lags = [lag for _t, lag in trusted]
    worst_ms = max(abs(x) for x in lags) * 1000.0
    jump_ms = 0.0
    if len(lags) >= 2:
        jump_ms = max(abs(b - a) for a, b in zip(lags, lags[1:])) * 1000.0
    if worst_ms <= float(limit_ms):
        return "ok", None, None, ""
    times = np.array([t for t, _lag in trusted], dtype=float)
    lag_a = np.array(lags, dtype=float)
    if len(times) == 1:
        times = np.array([float(times[0]), float(times[0]) + 30.0], dtype=float)
        lag_a = np.array([float(lag_a[0]), float(lag_a[0])], dtype=float)
    detail = f"worst={worst_ms:.0f}ms jump={jump_ms:.0f}ms"
    if unresolved:
        detail += f" unresolved={unresolved}"
    return "repair", times, lag_a, detail


def plan_residual_repair(
    reference: np.ndarray,
    query: np.ndarray,
    sr: int,
    limit_ms: float,
) -> tuple[str, np.ndarray | None, np.ndarray | None, str]:
    """Measure the rendered stem and plan one correction warp."""
    return repair_curve(residual_checkpoints(reference, query, sr), limit_ms)


def _apply_offset_gate(
    verdict: str,
    notes: str,
    coarse_lag_sec: float,
    points: list[tuple[float, float, bool]],
    limit_ms: float,
    *,
    coarse_max_sec: float = DEFAULT_COARSE_LAG_MAX_SEC,
) -> tuple[str, str, float]:
    """Pass or fail a stem from the onset checkpoints.

    Checkpoints inside ``limit_ms``, with no search-edge hit, pass even when
    the coarse chroma check failed. A point past the limit, or a checkpoint
    stuck on the search edge, fails the stem. A step between two points that
    are both inside the limit does not. With no checkpoints the coarse
    verdict is kept. Returns the signed offset with the largest magnitude.
    """
    reported = float(coarse_lag_sec)
    if abs(coarse_lag_sec) > coarse_max_sec:
        reason = f"lag_abs={abs(coarse_lag_sec):.3f}s>{coarse_max_sec:.3f}s"
        if reason not in notes:
            notes = f"{notes}; {reason}"
        verdict = "fail"
    if not points:
        return verdict, notes, reported

    lags = [p[1] for p in points]
    worst_i = int(np.argmax([abs(x) for x in lags]))
    reported = float(lags[worst_i])
    worst_ms = abs(reported) * 1000.0
    jump_ms = 0.0
    if len(lags) >= 2:
        jump_ms = max(abs(b - a) for a, b in zip(lags, lags[1:])) * 1000.0
    railed = any(p[2] for p in points)
    grade = "bad" if railed else offset_grade(worst_ms)
    shown = " ".join(f"{t:.0f}:{lag * 1000:+.1f}" for t, lag, _r in points)
    notes = (
        f"{notes}; offset={grade} worst={worst_ms:.1f}ms jump={jump_ms:.1f}ms [{shown}]"
    )
    if railed or worst_ms > limit_ms:
        verdict = "fail"
        if railed:
            notes += "; offset_past_search"
        elif worst_ms > limit_ms:
            notes += f"; offset={worst_ms:.1f}ms>{limit_ms:.0f}ms"
        return verdict, notes, reported
    if verdict != "pass":
        notes += f"; checkpoints_within={limit_ms:.0f}ms; coarse_fail_cleared"
    verdict = "pass"
    return verdict, notes, reported


_CHECK_BLOCK_RE = re.compile(
    r"(aca_check|inst_check)=(pass|fail)[^;]*\((.*?)\)",
    re.DOTALL,
)
_CHECKPOINT_RE = re.compile(r"(\d+(?:\.\d+)?):([+-]\d+(?:\.\d+)?)")
_OFFSET_GATE_RE = re.compile(r"(?:offset|jump)=[\d.]+ms>([\d.]+)ms")
# A checkpoint that lands on the ±40 ms search edge is stored near 35 ms.
_RAIL_ABS_MS = 34.0
_STEM_REVIEW_LABEL = {"aca_check": "acapella", "inst_check": "instrumental"}


def _review_clock(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _review_amount(points: list[tuple[float, float, bool]]) -> str:
    def token(ms: float, railed: bool) -> str:
        if railed:
            return f"past {'+' if ms >= 0 else '-'}40ms"
        return f"{ms:+.0f}ms"

    lo = min(points, key=lambda point: point[1])
    hi = max(points, key=lambda point: point[1])
    left = token(lo[1], lo[2])
    right = token(hi[1], hi[2])
    if left == right:
        return left
    return f"{left} to {right}"


def _checkpoint_parts(point: object) -> tuple[float, float, bool] | None:
    """[time_sec, lag_ms, search_edge], or the same fields on a dict."""
    if isinstance(point, dict):
        time_sec = point.get("time_sec", point.get("t"))
        lag_ms = point.get("lag_ms", point.get("lag"))
        edge = point.get("search_edge", point.get("railed", False))
        if not isinstance(time_sec, (int, float)) or not isinstance(lag_ms, (int, float)):
            return None
        return (float(time_sec), float(lag_ms), bool(edge))
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        if not isinstance(point[0], (int, float)) or not isinstance(point[1], (int, float)):
            return None
        edge = point[2] if len(point) > 2 else False
        return (float(point[0]), float(point[1]), bool(edge))
    return None


def drift_ranges(points: list, *, limit_ms: float = DEFAULT_DRIFT_MS) -> str:
    """One stem's 'drifts between' text. ``points`` are [time_sec, lag_ms, search_edge]."""
    half = CHECKPOINT_WIN_SEC / 2.0
    parsed: list[tuple[float, float, bool]] = []
    for point in points or []:
        parts = _checkpoint_parts(point)
        if parts is not None:
            parsed.append(parts)
    if not parsed:
        return ""

    def _over(point: tuple[float, float, bool]) -> bool:
        return point[2] or abs(point[1]) > limit_ms

    parts: list[str] = []
    group: list[tuple[float, float, bool]] = []

    def _flush() -> None:
        if not group:
            return
        start = _review_clock(group[0][0] - half)
        end = _review_clock(group[-1][0] + half)
        parts.append(f"drifts between {start} - {end} ({_review_amount(group)})")
        group.clear()

    for point in parsed:
        if not _over(point):
            _flush()
            continue
        if group and (point[1] >= 0) != (group[-1][1] >= 0):
            _flush()
        group.append(point)
    _flush()
    return ", ".join(parts)


def review_fail_parts(notes: str, *, limit_ms: float = DEFAULT_DRIFT_MS) -> tuple[str, str]:
    """Acapella drift notes, then instrumental drift notes.

    Each checkpoint is the center of an 8 s window. Points past ``limit_ms``
    in the same direction are one range. A pass stem is left out.
    """
    aca_parts: list[str] = []
    inst_parts: list[str] = []
    if not notes:
        return "", ""
    for kind, verdict, body in _CHECK_BLOCK_RE.findall(notes):
        if verdict != "fail" or kind not in _STEM_REVIEW_LABEL:
            continue
        gates = [float(value) for value in _OFFSET_GATE_RE.findall(body)]
        limit = min(gates) if gates else float(limit_ms)
        past_search = "offset_past_search" in body
        points: list[list] = []
        for time_text, lag_text in _CHECKPOINT_RE.findall(body):
            lag_ms = float(lag_text)
            railed = past_search and abs(lag_ms) >= _RAIL_ABS_MS
            points.append([float(time_text), lag_ms, railed])
        text = drift_ranges(points, limit_ms=limit)
        if not text:
            continue
        if kind == "aca_check":
            aca_parts.append(text)
        else:
            inst_parts.append(text)
    return ", ".join(aca_parts), ", ".join(inst_parts)


def review_fail_text(notes: str, *, limit_ms: float = DEFAULT_DRIFT_MS) -> str:
    """Both stems' drift notes, for the run log."""
    aca, inst = review_fail_parts(notes, limit_ms=limit_ms)
    clauses: list[str] = []
    if aca:
        clauses.append(f"acapella: {aca}")
    if inst:
        clauses.append(f"instrumental: {inst}")
    return " | ".join(clauses)


def analyze_alignment(
    acapella: Path,
    instrumental: Path,
    original: Path,
    *,
    sr: int = DEFAULT_SR,
    max_shift_sec: float = DEFAULT_MAX_SHIFT_SEC,
    window_sec: float = DEFAULT_WINDOW_SEC,
    corr_min: float = DEFAULT_CORR_MIN,
    drift_ms: float = DEFAULT_DRIFT_MS,
    window_corr_min: float = DEFAULT_WINDOW_CORR_MIN,
    weak_window_frac: float = DEFAULT_WEAK_WINDOW_FRAC,
) -> tuple[str, float, float, float, float, str]:
    """Return (verdict, corr, lag_sec, drift_ms, weak_frac, notes) for the mix vs the original."""
    aca = _peak_norm(_load_mono(acapella, sr))
    inst = _peak_norm(_load_mono(instrumental, sr))
    orig = _peak_norm(_load_mono(original, sr))
    aca, inst = _match_length(aca, inst)
    mix = _peak_norm(aca + inst)
    return score_alignment(
        orig,
        mix,
        sr=sr,
        max_shift_sec=max_shift_sec,
        window_sec=window_sec,
        corr_min=corr_min,
        drift_ms=drift_ms,
        window_corr_min=window_corr_min,
        weak_window_frac=weak_window_frac,
    )


def _combine_stem_verdicts(aca_verdict: str, inst_verdict: str) -> str:
    """Folder tag fails when either stem fails."""
    if aca_verdict == "fail" or inst_verdict == "fail":
        return "fail"
    if aca_verdict == "pass" and inst_verdict == "pass":
        return "pass"
    return "error"


def stem_verdicts(
    acapella: Path,
    instrumental: Path,
    original: Path,
    *,
    folder: Path | None = None,
    sr: int = DEFAULT_SR,
    max_shift_sec: float = DEFAULT_MAX_SHIFT_SEC,
    window_sec: float = DEFAULT_WINDOW_SEC,
    corr_min: float = DEFAULT_CORR_MIN,
    drift_ms: float = DEFAULT_DRIFT_MS,
    window_corr_min: float = DEFAULT_WINDOW_CORR_MIN,
    weak_window_frac: float = DEFAULT_WEAK_WINDOW_FRAC,
    lag_max_sec: float = DEFAULT_STEM_LAG_MAX_SEC,
) -> tuple[str, str, str, float, float, float, float, float, float, str]:
    """Score each stem against its Demucs reference.

    Pass/fail uses onset checkpoints on the original's 8-bar downbeats, or
    every 30 s when that bar line is missing. A stem passes when every
    checkpoint stays inside drift_ms, including when the coarse chroma check
    had failed. A nearer hat does not count when a stronger match sits further
    out, up to one second. A silent stretch is not a checkpoint. A point past
    drift_ms fails the stem. A step between two in-limit points does not.

    Returns (aca_verdict, inst_verdict, combined, aca_corr, inst_corr,
    aca_drift_ms, inst_drift_ms, aca_lag_sec, inst_lag_sec, notes,
    aca_checkpoints, inst_checkpoints). Checkpoints are
    [time_sec, lag_ms, search_edge].
    """
    from demucs_vocals import load_instrumental_mono, load_vocals_mono

    aca = _load_mono(acapella, sr)
    inst = _load_mono(instrumental, sr)
    vox = load_vocals_mono(original, sr, folder=folder)
    demucs_inst = load_instrumental_mono(original, sr, folder=folder)
    # Chroma drift is quantized to ~23 ms, so it cannot host a 10 ms gate.
    kwargs = dict(
        sr=sr,
        max_shift_sec=max_shift_sec,
        window_sec=window_sec,
        corr_min=corr_min,
        drift_ms=max(float(drift_ms), 100.0),
        window_corr_min=window_corr_min,
        weak_window_frac=weak_window_frac,
    )
    aca_verdict, aca_corr, aca_lag, aca_drift, _weak, aca_notes = score_alignment(vox, aca, **kwargs)
    inst_verdict, inst_corr, inst_lag, inst_drift, _weak, inst_notes = score_alignment(
        demucs_inst, inst, **kwargs
    )
    coarse_max = max(DEFAULT_COARSE_LAG_MAX_SEC, float(lag_max_sec))
    sample_times, sample_note = checkpoint_sample_times(original)
    aca_points = checkpoint_offsets(
        vox, aca, sr, at_times=sample_times, confirm_sec=CHECKPOINT_CONFIRM_LAG_SEC
    )
    inst_points = checkpoint_offsets(
        demucs_inst,
        inst,
        sr,
        at_times=sample_times,
        confirm_sec=CHECKPOINT_CONFIRM_LAG_SEC,
    )
    aca_verdict, aca_notes, aca_lag = _apply_offset_gate(
        aca_verdict, aca_notes, aca_lag, aca_points, drift_ms, coarse_max_sec=coarse_max
    )
    inst_verdict, inst_notes, inst_lag = _apply_offset_gate(
        inst_verdict, inst_notes, inst_lag, inst_points, drift_ms, coarse_max_sec=coarse_max
    )
    combined = _combine_stem_verdicts(aca_verdict, inst_verdict)
    notes = (
        f"aca_check={aca_verdict} corr={aca_corr:.3f} lag={aca_lag:+.3f}s"
        f" drift={aca_drift:.1f}ms ({aca_notes}); "
        f"inst_check={inst_verdict} corr={inst_corr:.3f} lag={inst_lag:+.3f}s"
        f" drift={inst_drift:.1f}ms ({inst_notes}); {sample_note}"
    )
    return (
        aca_verdict,
        inst_verdict,
        combined,
        aca_corr,
        inst_corr,
        aca_drift,
        inst_drift,
        aca_lag,
        inst_lag,
        notes,
        [
            [round(float(t), 3), round(float(lag) * 1000.0, 1), bool(railed)]
            for t, lag, railed in aca_points
        ],
        [
            [round(float(t), 3), round(float(lag) * 1000.0, 1), bool(railed)]
            for t, lag, railed in inst_points
        ],
    )


def apply_stem_check(result, acapella: Path, instrumental: Path, original: Path, **kwargs) -> None:
    """Write per-stem verdicts onto result and set result.verdict from them."""
    try:
        (
            aca_verdict,
            inst_verdict,
            combined,
            aca_corr,
            inst_corr,
            aca_drift,
            inst_drift,
            aca_lag,
            inst_lag,
            stem_notes,
            aca_points,
            inst_points,
        ) = stem_verdicts(acapella, instrumental, original, **kwargs)
    except Exception as exc:  # noqa: BLE001 — one bad stem check must not abort the batch
        result.aca_verdict = "error"
        result.inst_verdict = "error"
        result.verdict = "error"
        result.notes += f"; stem_check:{type(exc).__name__}:{exc}"
        return
    result.aca_verdict = aca_verdict
    result.inst_verdict = inst_verdict
    result.aca_corr = aca_corr
    result.inst_corr = inst_corr
    result.aca_check_drift_ms = aca_drift
    result.inst_check_drift_ms = inst_drift
    result.aca_lag_sec = aca_lag
    result.inst_lag_sec = inst_lag
    result.aca_checkpoints = aca_points
    result.inst_checkpoints = inst_points
    result.verdict = combined
    result.notes += f"; {stem_notes}"


def _emit_step(on_step, step_id: str) -> None:
    if on_step is None:
        return
    try:
        on_step(step_id)
    except Exception:  # noqa: BLE001 — UI callback must not abort processing
        pass


def process_folder(
    folder: Path,
    *,
    dry_run: bool,
    sr: int,
    max_shift_sec: float,
    window_sec: float,
    corr_min: float,
    drift_ms: float,
    window_corr_min: float,
    weak_window_frac: float,
    skip_tagged: bool = True,
    on_step=None,
) -> CheckResult:
    name = folder.name
    if skip_tagged and TAG_RE.search(name):
        return CheckResult(folder=name, verdict="skip", notes="already_tagged")

    _emit_step(on_step, "scan")
    aca, inst, orig, scan_notes = scan_folder(folder)
    if aca is None or inst is None or orig is None:
        missing = []
        if aca is None:
            missing.append("acapella")
        if inst is None:
            missing.append("instrumental")
        if orig is None:
            missing.append("original")
        return CheckResult(
            folder=name,
            verdict="skip",
            notes=f"missing:{','.join(missing)}" + (f"; {scan_notes}" if scan_notes else ""),
            acapella=aca.name if aca else "",
            instrumental=inst.name if inst else "",
            original=orig.name if orig else "",
        )

    try:
        _emit_step(on_step, "score")
        verdict, corr, lag, drift, weak, notes = analyze_alignment(
            aca,
            inst,
            orig,
            sr=sr,
            max_shift_sec=max_shift_sec,
            window_sec=window_sec,
            corr_min=corr_min,
            drift_ms=drift_ms,
            window_corr_min=window_corr_min,
            weak_window_frac=weak_window_frac,
        )
    except Exception as exc:  # noqa: BLE001 — per-folder isolation
        return CheckResult(
            folder=name,
            verdict="error",
            acapella=aca.name,
            instrumental=inst.name,
            original=orig.name,
            notes=f"{type(exc).__name__}: {exc}",
        )

    if scan_notes:
        notes = f"{scan_notes}; {notes}"

    result = CheckResult(
        folder=name,
        verdict=verdict,
        corr=corr,
        lag_sec=lag,
        drift_ms=drift,
        weak_window_frac=weak,
        acapella=aca.name,
        instrumental=inst.name,
        original=orig.name,
        notes=notes,
    )
    apply_stem_check(
        result,
        aca,
        inst,
        orig,
        folder=folder,
        sr=sr,
        max_shift_sec=max_shift_sec,
        window_sec=window_sec,
        corr_min=corr_min,
        drift_ms=drift_ms,
        window_corr_min=window_corr_min,
        weak_window_frac=weak_window_frac,
    )
    from align_model.quality import stamp_result

    stamp_result(result, mix_verdict=verdict, mix_corr=corr, mix_lag=lag, mix_drift=drift)
    verdict = result.verdict

    if verdict in ("pass", "fail") and not dry_run:
        base = TAG_RE.sub("", name).strip()
        new_name = f"{base}_[{verdict}]"
        if folder.name == new_name:
            return result
        dest = folder.with_name(new_name)
        if dest.exists():
            result.notes += "; rename_blocked_exists"
            result.verdict = "error"
            return result
        try:
            _emit_step(on_step, "tag")
            folder.rename(dest)
            result.renamed_to = new_name
        except OSError as exc:
            result.notes += f"; rename_failed:{exc}"
            result.verdict = "error"

    return result


def _worker(payload: dict) -> dict:
    folder = Path(payload["folder"])
    result = process_folder(
        folder,
        dry_run=payload["dry_run"],
        sr=payload["sr"],
        max_shift_sec=payload["max_shift_sec"],
        window_sec=payload["window_sec"],
        corr_min=payload["corr_min"],
        drift_ms=payload["drift_ms"],
        window_corr_min=payload["window_corr_min"],
        weak_window_frac=payload["weak_window_frac"],
    )
    return asdict(result)


def list_work_folders(root: Path, limit: int | None) -> list[Path]:
    folders = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name != "_backup_before_align"
    )
    # Skip already tagged at listing time for clearer --limit behavior
    folders = [p for p in folders if not TAG_RE.search(p.name)]
    if limit is not None:
        folders = folders[:limit]
    return folders


def csv_cells(row: dict) -> dict:
    """CSV cells. Checkpoint lists are stored as JSON text."""
    out: dict = {}
    for key, value in row.items():
        if isinstance(value, (list, tuple, dict)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def write_csv(path: Path, rows: list[CheckResult]) -> None:
    fields = list(asdict(rows[0]).keys()) if rows else list(asdict(CheckResult(folder="", verdict="")).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_cells(asdict(row)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rename song folders _[pass]/_[fail] based on acapella+instrumental vs original alignment."
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Folder containing song subfolders")
    p.add_argument("--dry-run", action="store_true", help="Judge only; do not rename")
    p.add_argument("--limit", type=int, default=None, help="Process only first N untagged folders")
    p.add_argument("--workers", type=int, default=2, help="Parallel worker processes (1=serial)")
    p.add_argument("--sr", type=int, default=DEFAULT_SR, help="Analysis sample rate")
    p.add_argument("--max-shift-sec", type=float, default=DEFAULT_MAX_SHIFT_SEC)
    p.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    p.add_argument("--corr-min", type=float, default=DEFAULT_CORR_MIN, help="Min global correlation to pass")
    p.add_argument("--drift-ms", type=float, default=DEFAULT_DRIFT_MS, help="Max allowed lag range across windows")
    p.add_argument("--window-corr-min", type=float, default=DEFAULT_WINDOW_CORR_MIN)
    p.add_argument("--weak-window-frac", type=float, default=DEFAULT_WEAK_WINDOW_FRAC)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "alignment_results.csv",
        help="Results CSV path",
    )
    p.add_argument(
        "--only",
        action="append",
        default=None,
        help="Process only folders whose name contains this substring (repeatable)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.root
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 2

    folders = list_work_folders(root, args.limit)
    if args.only:
        needles = [n.lower() for n in args.only]
        folders = [f for f in folders if any(n in f.name.lower() for n in needles)]

    if not folders:
        print("No folders to process.")
        return 0

    print(f"Root: {root}")
    print(f"Folders: {len(folders)}  dry_run={args.dry_run}  workers={args.workers}")
    print(
        f"Thresholds: corr_min={args.corr_min}  drift_ms={args.drift_ms}  "
        f"stem_lag_max_sec={DEFAULT_STEM_LAG_MAX_SEC:.3f}  "
        f"window_corr_min={args.window_corr_min}  weak_window_frac={args.weak_window_frac}"
    )

    results: list[CheckResult] = []
    payload_base = {
        "dry_run": args.dry_run,
        "sr": args.sr,
        "max_shift_sec": args.max_shift_sec,
        "window_sec": args.window_sec,
        "corr_min": args.corr_min,
        "drift_ms": args.drift_ms,
        "window_corr_min": args.window_corr_min,
        "weak_window_frac": args.weak_window_frac,
    }

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    def _emit(result: CheckResult) -> None:
        results.append(result)
        tag = result.verdict.upper()
        extra = f" corr={result.corr:.3f} lag={result.lag_sec:+.3f}s drift={result.drift_ms:.1f}ms"
        if result.aca_verdict or result.inst_verdict:
            extra += (
                f" aca_lag={result.aca_lag_sec:+.3f}s"
                f" inst_lag={result.inst_lag_sec:+.3f}s"
            )
        if result.verdict == "skip":
            extra = f" ({result.notes})"
        elif result.verdict == "error":
            extra = f" ({result.notes})"
        print(f"[{tag}] {result.folder}{extra}")

    if args.workers <= 1:
        iterator = folders
        if tqdm is not None:
            iterator = tqdm(folders, unit="folder")
        for folder in iterator:
            _emit(
                process_folder(
                    folder,
                    dry_run=args.dry_run,
                    sr=args.sr,
                    max_shift_sec=args.max_shift_sec,
                    window_sec=args.window_sec,
                    corr_min=args.corr_min,
                    drift_ms=args.drift_ms,
                    window_corr_min=args.window_corr_min,
                    weak_window_frac=args.weak_window_frac,
                )
            )
    else:
        payloads = [{**payload_base, "folder": str(f)} for f in folders]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, p): p["folder"] for p in payloads}
            done_iter = as_completed(futures)
            if tqdm is not None:
                done_iter = tqdm(done_iter, total=len(futures), unit="folder")
            for fut in done_iter:
                try:
                    _emit(CheckResult(**fut.result()))
                except Exception as exc:  # noqa: BLE001
                    folder_name = Path(futures[fut]).name
                    _emit(
                        CheckResult(
                            folder=folder_name,
                            verdict="error",
                            notes=f"worker:{type(exc).__name__}: {exc}",
                        )
                    )

    write_csv(args.csv, results)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    print("---")
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
