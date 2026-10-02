#!/usr/bin/env python3
"""Warp-align stems from _backup_before_align onto the original song.

Method:
  1. Load backup acapella/instrumental (pre-align originals)
  2. Find global front pad/trim via chroma cross-correlation vs original
     - acapella: compare vocal-band (150–5kHz) chroma
     - instrumental: full-band chroma
  3. Measure time-varying lag in sliding windows (drift curve)
  4. Pitch-preserving warp with Rubber Band timemap so lag(t) is cancelled
  5. Fit to original length/SR/channels and write FLAC
  6. Re-check mix alignment; rename the folder in place to _[pass] or _[fail]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from declick import resolve_declick
from check_alignment import (
    DEFAULT_CORR_MIN,
    DEFAULT_DRIFT_MS,
    DEFAULT_WEAK_WINDOW_FRAC,
    DEFAULT_WINDOW_CORR_MIN,
    analyze_alignment,
    apply_stem_check,
    scan_folder,
)

FAIL_ALL = Path(r"T:\SDR30+\Pair-finder organized done\!2-files\with_original\_fail_all")
PASS_ALL = Path(r"T:\SDR30+\Pair-finder organized done\!2-files\with_original\_pass_all")
BACKUP_NAME = "_backup_before_align"
OUT_DIR = Path(__file__).resolve().parent
RB_DIR = OUT_DIR / "tools" / "rubberband" / "rubberband-4.0.0-gpl-executable-windows"
RB_EXE = RB_DIR / "rubberband.exe"

SR_ANALYSIS = 22050
HOP = 512
DEFAULT_MAX_PAD_SEC = 90.0


def _ensure_rubberband_on_path() -> Path:
    if not RB_EXE.is_file():
        raise FileNotFoundError(f"rubberband.exe not found at {RB_EXE}")
    rb_dir = str(RB_DIR)
    if rb_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = rb_dir + os.pathsep + os.environ.get("PATH", "")
    return RB_EXE


@dataclass
class WarpResult:
    folder: str
    verdict: str = ""
    corr: float = 0.0
    lag_sec: float = 0.0
    drift_ms: float = 0.0
    aca_pad_sec: float = 0.0
    inst_pad_sec: float = 0.0
    aca_drift_range_ms: float = 0.0
    inst_drift_range_ms: float = 0.0
    moved_to: str = ""
    notes: str = ""
    aca_verdict: str = ""
    inst_verdict: str = ""
    aca_corr: float = 0.0
    inst_corr: float = 0.0
    aca_check_drift_ms: float = 0.0
    inst_check_drift_ms: float = 0.0
    aca_lag_sec: float = 0.0
    inst_lag_sec: float = 0.0
    aca_lag_sec: float = 0.0
    inst_lag_sec: float = 0.0


def peak_norm(y: np.ndarray) -> np.ndarray:
    p = float(np.max(np.abs(y))) if y.size else 0.0
    return (y / p).astype(np.float32) if p > 1e-9 else y.astype(np.float32)


def load_mono(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def bandpass(y: np.ndarray, sr: int, lo: float = 150.0, hi: float = 5000.0) -> np.ndarray:
    hi = min(hi, sr / 2 - 100)
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, y).astype(np.float32)


def pad_or_trim_front(y: np.ndarray, pad_sec: float, sr: int) -> np.ndarray:
    """Positive pad_sec = prepend silence; negative = trim from front."""
    n = int(round(pad_sec * sr))
    if n > 0:
        return np.concatenate([np.zeros(n, dtype=np.float32), y])
    if n < 0:
        cut = min(-n, len(y))
        return y[cut:]
    return y


def fit_len(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    return y[:n]


def chroma_norm(y: np.ndarray, sr: int) -> np.ndarray:
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    norms = np.linalg.norm(C, axis=0, keepdims=True)
    return (C / np.maximum(norms, 1e-9)).astype(np.float32)


def chroma_xcorr_pad(
    ref: np.ndarray,
    qry: np.ndarray,
    *,
    sr: int,
    max_pad_sec: float,
) -> tuple[float, float]:
    """Return (pad_sec, score). Positive pad_sec = delay query (prepend silence)."""
    rf = chroma_norm(ref, sr)
    qf = chroma_norm(qry, sr)
    hop_sec = HOP / sr
    max_lag = int(max_pad_sec / hop_sec)

    corr_sum = None
    e_r = e_q = 0.0
    for b in range(rf.shape[0]):
        rb = rf[b] - float(np.mean(rf[b]))
        qb = qf[b] - float(np.mean(qf[b]))
        e_r += float(np.dot(rb, rb))
        e_q += float(np.dot(qb, qb))
        c = signal.correlate(rb, qb, mode="full", method="fft")
        corr_sum = c if corr_sum is None else corr_sum + c

    lags = np.arange(-qf.shape[1] + 1, rf.shape[1])
    valid = (lags >= -max_lag) & (lags <= max_lag)
    cv = corr_sum[valid]
    lv = lags[valid]
    bi = int(np.argmax(cv))
    denom = float(np.sqrt(e_r * e_q)) or 1e-12
    score = float(cv[bi]) / denom
    pad_sec = float(lv[bi]) * hop_sec
    return pad_sec, score


def rms_frames(y: np.ndarray, hop: int = HOP, frame: int = 2048) -> np.ndarray:
    if len(y) < frame:
        return np.zeros(1, dtype=np.float32)
    n = 1 + (len(y) - frame) // hop
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        sl = y[i * hop : i * hop + frame]
        out[i] = float(np.sqrt(np.mean(sl**2)))
    return out


def lag_curve(
    ref: np.ndarray,
    qry: np.ndarray,
    *,
    sr: int,
    win_sec: float = 10.0,
    step_sec: float = 2.0,
    max_lag_sec: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hop_sec = HOP / sr
    rf = chroma_norm(ref, sr)
    qf = chroma_norm(qry, sr)
    q_rms = rms_frames(qry, hop=HOP)
    n = min(rf.shape[1], qf.shape[1], len(q_rms))
    rf, qf, q_rms = rf[:, :n], qf[:, :n], q_rms[:n]
    rms_thr = max(float(np.percentile(q_rms, 60)) * 0.35, 1e-4)

    win = max(8, int(win_sec / hop_sec))
    step = max(1, int(step_sec / hop_sec))
    max_lag = max(2, int(max_lag_sec / hop_sec))

    times, lags, scores = [], [], []
    for start in range(0, max(1, n - win), step):
        end = start + win
        if float(np.mean(q_rms[start:end])) < rms_thr:
            continue
        r = rf[:, start:end]
        q = qf[:, start:end]
        corr_sum = None
        e_r = e_q = 0.0
        for b in range(r.shape[0]):
            rb = r[b] - float(np.mean(r[b]))
            qb = q[b] - float(np.mean(q[b]))
            e_r += float(np.dot(rb, rb))
            e_q += float(np.dot(qb, qb))
            c = signal.correlate(rb, qb, mode="full", method="fft")
            corr_sum = c if corr_sum is None else corr_sum + c
        lags_f = np.arange(-q.shape[1] + 1, r.shape[1])
        valid = (lags_f >= -max_lag) & (lags_f <= max_lag)
        cv, lv = corr_sum[valid], lags_f[valid]
        bi = int(np.argmax(cv))
        denom = float(np.sqrt(e_r * e_q)) or 1e-12
        times.append((start + win / 2) * hop_sec)
        lags.append(float(lv[bi]) * hop_sec)
        scores.append(float(cv[bi]) / denom)
    return np.asarray(times), np.asarray(lags), np.asarray(scores)


def onset_lag_curve(
    ref: np.ndarray,
    qry: np.ndarray,
    *,
    sr: int,
    win_sec: float = 8.0,
    step_sec: float = 4.0,
    max_lag_sec: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attack lag, for the instrumental map. Chroma misses a kick that slips."""
    import librosa
    from scipy.signal import correlate

    hop = 128
    hop_sec = hop / sr
    rf = librosa.onset.onset_strength(
        y=np.asarray(ref, dtype=np.float32), sr=sr, hop_length=hop
    ).astype(np.float64)
    qf = librosa.onset.onset_strength(
        y=np.asarray(qry, dtype=np.float32), sr=sr, hop_length=hop
    ).astype(np.float64)
    n = min(len(rf), len(qf))
    rf, qf = rf[:n], qf[:n]
    win = max(8, int(win_sec / hop_sec))
    step = max(1, int(step_sec / hop_sec))
    max_lag = max(2, int(max_lag_sec / hop_sec))
    times, lags, scores = [], [], []
    for start in range(0, max(1, n - win), step):
        end = start + win
        r = rf[start:end]
        q = qf[start:end]
        if float(np.mean(q)) < 1e-4 or float(np.mean(r)) < 1e-4:
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
        center = mid - lo
        i = int(np.argmax(window))
        denom = float(np.linalg.norm(r) * np.linalg.norm(q)) + 1e-12
        times.append((start + win / 2.0) * hop_sec)
        lags.append((i - center) * hop_sec)
        scores.append(float(window[i]) / denom)
    return np.asarray(times), np.asarray(lags), np.asarray(scores)


def fit_linear(times: np.ndarray, lags: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(times) < 3:
        return 0.0, 0.0
    thr = max(0.1, float(np.percentile(scores, 40)))
    m = scores >= thr
    if np.count_nonzero(m) < 5:
        m = np.ones(len(times), dtype=bool)
    t, L, w = times[m], lags[m], np.clip(scores[m], 0.05, None)
    A = np.vstack([np.ones_like(t), t]).T
    coef, _, _, _ = np.linalg.lstsq(A * w[:, None], L * w, rcond=None)
    return float(coef[0]), float(coef[1])


def smooth_lags(times: np.ndarray, lags: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(lags) == 0:
        return times, lags
    out = lags.copy()
    for i in range(len(lags)):
        lo, hi = max(0, i - 2), min(len(lags), i + 3)
        w = np.clip(scores[lo:hi], 0.01, None)
        out[i] = float(np.average(lags[lo:hi], weights=w))
    return times, out


def time_stretch_pitch_preserving(y: np.ndarray, rate: float, sr: int = SR_ANALYSIS) -> np.ndarray:
    """Rubber Band tempo change; pitch unchanged. rate<1 slows down (longer)."""
    if abs(rate - 1.0) < 3e-4:
        return y.astype(np.float32, copy=False)
    _ensure_rubberband_on_path()
    import pyrubberband as pyrb

    rate = float(np.clip(rate, 0.90, 1.10))
    return pyrb.time_stretch(y.astype(np.float64), sr, rate).astype(np.float32)


def absorb_lag_offset(
    times: np.ndarray,
    lags: np.ndarray,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Split a constant offset out of a lag curve.

    The timemap uses one rate pivoted at t=0, so it only removes the slope.
    A flat offset has slope 0 and would be left in the rendered stem. That
    offset is returned for the caller to add to the front pad (positive
    delays the stem, same sign as chroma_xcorr_pad).
    """
    if len(times) < 2 or len(lags) < 2:
        return lags, 0.0
    t = np.asarray(times, dtype=float)
    lag = np.asarray(lags, dtype=float)
    if scores is not None and len(scores) == len(lag):
        w = np.clip(np.asarray(scores, dtype=float), 0.05, None)
        intercept = float(np.polyfit(t, lag, 1, w=w)[1])
    else:
        intercept = float(np.polyfit(t, lag, 1)[1])
    if abs(intercept) < 1e-4:
        return lag, 0.0
    return lag - intercept, intercept


def lag_line_rms(times: np.ndarray, lags: np.ndarray) -> float:
    """RMS leftover after the best straight tempo line, in seconds."""
    if len(times) < 6:
        return 0.0
    t = np.asarray(times, dtype=float)
    lag = np.asarray(lags, dtype=float)
    resid = lag - np.polyval(np.polyfit(t, lag, 1), t)
    return float(np.sqrt(np.mean(resid**2)))


def _despike_lags(lags: np.ndarray, scores: np.ndarray | None, limit_sec: float = 0.12) -> np.ndarray:
    """Replace a window that jumped to the wrong beat with the local median."""
    out = np.asarray(lags, dtype=float).copy()
    for i in range(len(out)):
        lo, hi = max(0, i - 3), min(len(out), i + 4)
        med = float(np.median(out[lo:hi]))
        low_score = scores is not None and i < len(scores) and float(scores[i]) < 0.35
        if abs(out[i] - med) > limit_sec or low_score:
            out[i] = med
    return out


def bar_mark_times(
    downbeats: np.ndarray,
    target_sec: float,
    bars: int = 8,
) -> tuple[np.ndarray, float] | None:
    """Downbeats every ``bars`` bars, inside the song.

    The times are a ruler for stretch markers. They do not change the lag.
    None when the bar line is too short to trust, and the caller keeps the
    30 s marks.
    """
    db = np.sort(np.asarray(downbeats, dtype=float))
    db = db[np.isfinite(db)]
    db = db[(db >= -1e-3) & (db < float(target_sec) - 1.0)]
    if len(db) < bars + 1:
        return None
    gaps = np.diff(db)
    gaps = gaps[gaps > 0.2]
    if len(gaps) < 4:
        return None
    span = float(np.median(gaps)) * float(bars)
    marks = db[bars::bars]
    marks = marks[(marks > 1.0) & (marks < float(target_sec) - span * 0.5)]
    if len(marks) == 0:
        return None
    return marks, span


def follow_lag_src_dst(
    times: np.ndarray,
    lags: np.ndarray,
    *,
    target_sec: float,
    in_sec: float,
    scores: np.ndarray | None = None,
    spacing_sec: float = 30.0,
    despike: bool = True,
    window_sec: float = 10.0,
    grid_sec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow the measured lag instead of one straight rate.

    src(t) = t - lag(t). The regular marks are ``grid_sec`` when that ruler
    is passed (8-bar downbeats). Otherwise they fall every 30 s. A marker
    every few seconds clicks; a marker only once a minute misses a slip
    inside that minute. A step of 15 ms or more gets its own marker, left
    where the lag jumped, so a section change is not ramped across the span.

    The first lag sample is the center of a window. When the vocal starts
    late, that window sits after a block of silence. A marker only at the
    first bar line ramps the correction across the intro, so the first
    phrases stay late. The silence before the window takes the whole shift.
    """
    in_sec = max(0.0, float(in_sec))
    target_sec = max(0.0, float(target_sec))
    if target_sec <= 0 or len(times) < 2:
        return stretch_src_dst(times, lags, target_sec=target_sec, in_sec=in_sec)

    t = np.asarray(times, dtype=float)
    lag = np.asarray(lags, dtype=float)
    jump_t = [
        float(t[i])
        for i in range(1, len(t))
        if abs(float(lag[i] - lag[i - 1])) >= 0.015
    ]
    if despike:
        lag = _despike_lags(lag, scores)
    # The intro error, before a slow bow is smoothed into the body.
    lead_lag = float(lag[0])
    # Smooth a slow bow. A real step (section change) is left intact so the
    # extra marker at that time can take it, instead of smearing it over 20 s.
    if not jump_t:
        win = max(3, int(round(20.0 / max(float(np.median(np.diff(t))), 0.5))))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=float) / win
        lag = np.convolve(lag, kernel, mode="same")

    grid = None if grid_sec is None else np.asarray(grid_sec, dtype=float)
    if grid is not None:
        grid = grid[np.isfinite(grid)]
        grid = grid[(grid > 1.0) & (grid < target_sec - 1.0)]
        if jump_t and len(grid):
            jumps = np.asarray(jump_t, dtype=float)
            grid = np.asarray(
                [float(g) for g in grid if float(np.min(np.abs(jumps - g))) >= 0.4],
                dtype=float,
            )
    if grid is not None and len(grid):
        dst_list = [0.0, *[float(x) for x in grid]]
    else:
        spacing = max(30.0, float(spacing_sec))
        dst_list = [0.0]
        cursor = spacing
        while cursor < target_sec - spacing * 0.5:
            dst_list.append(cursor)
            cursor += spacing
    # No marker on the end of the song. That pin is a hard stop: source
    # audio after it is never played. The last rate holds, and the item
    # runs until the source has been heard.
    # Start of the first measured window. The shift is finished there, in
    # the silence, instead of half-applied when the vocal has already begun.
    entrance = float(t[0]) - max(0.0, float(window_sec)) * 0.5
    if (
        abs(lead_lag) >= 0.012
        and 1.0 < entrance < target_sec - 1.0
    ):
        dst_list.append(entrance)
    else:
        entrance = None
    if jump_t:
        dst_list.extend(x for x in jump_t if 1.0 < x < target_sec - 1.0)
    dst = np.unique(np.asarray(dst_list, dtype=float))
    lag_at = np.interp(dst, t, lag, left=float(lag[0]), right=float(lag[-1]))
    if entrance is not None:
        at = int(np.argmin(np.abs(dst - entrance)))
        if abs(float(dst[at]) - entrance) < 1e-3:
            lag_at[at] = float(np.clip(lead_lag, -0.15, 0.15))
    src = dst - lag_at
    src[0] = 0.0
    for i in range(1, len(src)):
        src[i] = max(float(src[i]), float(src[i - 1]) + 1e-3)
    # Cap each rate step. A corner of a few tenths of a percent clicks;
    # the bow still accumulates, it just cannot turn in one marker.
    # A minute marker may change rate by a few tenths of a percent. De-click
    # cleans that corner; clamping harder would leave the bow in the file.
    max_step = 0.02
    rates = np.diff(src) / np.maximum(np.diff(dst), 1e-6)
    for i in range(1, len(rates)):
        delta = float(rates[i] - rates[i - 1])
        if abs(delta) > max_step:
            rates[i] = rates[i - 1] + np.sign(delta) * max_step
    src = np.empty_like(dst)
    src[0] = 0.0
    for i, rate in enumerate(rates):
        src[i + 1] = src[i] + float(rate) * float(dst[i + 1] - dst[i])
    for i in range(1, len(src)):
        src[i] = max(float(src[i]), float(src[i - 1]) + 1e-3)
    # Drop a marker that would read past the file. Do not replace it with
    # a marker on the last sample: that is the hard stop.
    keep = [0]
    for i in range(1, len(src)):
        if src[i] > in_sec + 1e-3:
            break
        keep.append(i)
    src = src[keep]
    dst = dst[keep]
    if len(src) < 2:
        return stretch_src_dst(times, lags, target_sec=target_sec, in_sec=in_sec)
    return src, dst


def stretch_src_dst(
    times: np.ndarray,
    lags: np.ndarray,
    *,
    target_sec: float,
    in_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One constant-rate map: two endpoints, no corners.

    Élastique clicks on every stretch marker, including a 0.15% rate step.
    The measured lag is a straight drift (a clock error), so one rate is the
    correction. When that rate would read past the end of the file, the output
    stops when the source runs out; the caller pads the tail instead of
    bending the rate.
    """
    in_sec = max(0.0, float(in_sec))
    target_sec = max(0.0, float(target_sec))
    if target_sec <= 0:
        return np.array([0.0, in_sec]), np.array([0.0, 0.0])
    if len(times) < 2:
        return np.array([0.0, in_sec]), np.array([0.0, target_sec])

    slope = float(np.polyfit(np.asarray(times, dtype=float), np.asarray(lags, dtype=float), 1)[0])
    rate = float(np.clip(1.0 - slope, 0.5, 1.5))
    if rate <= 1e-6:
        rate = 1.0
    # How long we can hold this rate before the source ends.
    run_sec = in_sec / rate
    out_sec = min(target_sec, run_sec)
    src_end = min(in_sec, out_sec * rate)
    if out_sec <= 1e-4:
        return np.array([0.0, in_sec]), np.array([0.0, target_sec])
    return np.array([0.0, src_end]), np.array([0.0, out_sec])


def _build_timemap(
    times: np.ndarray,
    lags: np.ndarray,
    *,
    sr: int,
    in_frames: int,
    out_frames: int,
) -> list[tuple[int, int]]:
    """Map padded-input frames -> output frames using lag(t) on the output timeline."""
    t_end = out_frames / sr
    in_sec = max(0.0, (in_frames - 1) / sr) if sr else 0.0
    if lag_line_rms(times, lags) > 0.02:
        src_s, dst_s = follow_lag_src_dst(
            times, lags, target_sec=t_end, in_sec=in_sec
        )
    else:
        src_s, dst_s = stretch_src_dst(times, lags, target_sec=t_end, in_sec=in_sec)
    pairs: list[tuple[int, int]] = []
    last_tgt = -1
    for src, tgt_s in zip(src_s, dst_s):
        tgt = int(round(tgt_s * sr))
        tgt = min(max(tgt, 0), max(0, out_frames - 1))
        if tgt <= last_tgt:
            continue
        pairs.append((int(round(src * sr)), tgt))
        last_tgt = tgt
    if not pairs:
        return [(0, 0), (max(0, in_frames - 1), max(0, out_frames - 1))]
    # The last returned marker is not the end of the item. Continue at its
    # rate until the source has been heard, and at least to the original length.
    last_src, last_out = pairs[-1]
    if len(pairs) >= 2:
        prev_src, prev_out = pairs[-2]
        span = last_out - prev_out
        rate = (last_src - prev_src) / span if span > 0 else 1.0
    else:
        rate = 1.0
    if rate <= 1e-6:
        rate = 1.0
    remaining = (in_frames - 1) - last_src
    end_out = last_out
    if remaining > 1:
        end_out = last_out + int(round(remaining / rate))
    end_out = max(end_out, max(0, out_frames - 1))
    if end_out > last_out:
        pairs.append((max(0, in_frames - 1), end_out))
    return pairs


def rubberband_timemap_stretch(
    y: np.ndarray,
    *,
    sr: int,
    times: np.ndarray,
    lags: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    """Pitch-preserving warp via Rubber Band timemap. y may be multi-channel (N, C)."""
    exe = _ensure_rubberband_on_path()
    if y.ndim == 1:
        y_out = y[:, None]
        squeeze = True
    else:
        y_out = y
        squeeze = False

    with tempfile.TemporaryDirectory(prefix="rb_warp_") as tmp:
        tmp_path = Path(tmp)
        infile = tmp_path / "in.wav"
        outfile = tmp_path / "out.wav"
        mapfile = tmp_path / "timemap.txt"

        sf.write(str(infile), y_out, sr, subtype="FLOAT")
        pairs = _build_timemap(
            times, lags, sr=sr, in_frames=len(y_out), out_frames=target_frames
        )
        mapfile.write_text(
            "\n".join(f"{src} {tgt}" for src, tgt in pairs) + "\n", encoding="utf-8"
        )
        # Play through the source tail. The last map point is that end,
        # which can sit past the original's length.
        duration = max(pairs[-1][1], 1) / sr
        cmd = [
            str(exe),
            "-D",
            f"{duration:.6f}",
            "-M",
            str(mapfile),
            "-c",
            "5",
            str(infile),
            str(outfile),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not outfile.is_file():
            raise RuntimeError(
                f"rubberband failed ({proc.returncode}): {proc.stderr[-500:]}"
            )
        out, out_sr = sf.read(str(outfile), always_2d=True, dtype="float32")
        if out_sr != sr:
            out = np.stack(
                [librosa.resample(out[:, c], orig_sr=out_sr, target_sr=sr) for c in range(out.shape[1])],
                axis=1,
            ).astype(np.float32)
        out = fit_len_2d(out, max(target_frames, pairs[-1][1] + 1), sr)
        return out[:, 0] if squeeze else out


def fit_len_2d(y: np.ndarray, n: int, sr: int | None = None) -> np.ndarray:
    """Trim or pad to ``n`` frames. A padded tail fades so the join is not a click."""
    if y.ndim == 1:
        y = y[:, None]
        squeeze = True
    else:
        squeeze = False
    if len(y) > n:
        y = y[:n]
    elif len(y) < n:
        gap = n - len(y)
        fade_n = int(0.015 * sr) if sr else 0
        if gap > fade_n > 1 and len(y) > fade_n:
            y = y.copy()
            y[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)[:, None]
        y = np.pad(y, ((0, gap), (0, 0)))
    return y[:, 0] if squeeze else y


def write_warped_stem(
    src: Path,
    dest: Path,
    *,
    pad_sec: float,
    times: np.ndarray,
    lags: np.ndarray,
    target_sr: int,
    target_frames: int,
    target_channels: int,
) -> None:
    """Pad/trim then Rubber Band timemap-warp to original length (pitch preserved)."""
    y, file_sr = sf.read(str(src), always_2d=True, dtype="float32")
    if file_sr != target_sr:
        y = np.stack(
            [librosa.resample(y[:, c], orig_sr=file_sr, target_sr=target_sr) for c in range(y.shape[1])],
            axis=1,
        ).astype(np.float32)

    n = int(round(pad_sec * target_sr))
    if n > 0:
        y = np.concatenate([np.zeros((n, y.shape[1]), dtype=np.float32), y], axis=0)
    elif n < 0:
        y = y[min(-n, len(y)) :]

    if len(times) >= 2 and len(lags) >= 2:
        y = rubberband_timemap_stretch(
            y, sr=target_sr, times=times, lags=lags, target_frames=target_frames
        )
    else:
        y = fit_len_2d(y, target_frames)

    if y.shape[1] < target_channels:
        y = np.pad(y, ((0, 0), (0, target_channels - y.shape[1])))
    elif y.shape[1] > target_channels:
        y = y[:, :target_channels]

    tmp = dest.with_name(dest.stem + "._warp_tmp" + dest.suffix)
    if tmp.exists():
        tmp.unlink()
    sf.write(str(tmp), y, target_sr, format="FLAC")
    tmp.replace(dest)


def find_backup_stems(folder: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Return (aca_backup, inst_backup, original) using role classification."""
    backup = folder / BACKUP_NAME
    if not backup.is_dir():
        return None, None, None

    # Classify using check_alignment on backup files + top-level original
    aca_b = inst_b = orig = None
    # Temporarily scan: put roles from backup files
    from check_alignment import classify_audio_file

    for p in sorted(backup.iterdir()):
        if not p.is_file():
            continue
        role = classify_audio_file(p)
        if role == "acapella" and aca_b is None:
            aca_b = p
        elif role == "instrumental" and inst_b is None:
            inst_b = p

    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if classify_audio_file(p) == "original":
            orig = p
            break
    if orig is None:
        from check_alignment import unclassified_audio

        loose = unclassified_audio(folder)
        if len(loose) == 1:
            orig = loose[0]
    return aca_b, inst_b, orig


def strip_fail_tag(name: str) -> str:
    if name.endswith("_[fail]"):
        return name[: -len("_[fail]")]
    if name.endswith("_[pass]"):
        return name[: -len("_[pass]")]
    return name


def tag_folder_verdict(folder: Path, verdict: str) -> Path | None:
    """Rename folder in place so the name ends with _[pass] or _[fail].

    Returns the new path when renamed. Returns None when the tag is already correct.
    """
    if verdict not in ("pass", "fail"):
        return None
    new_name = f"{strip_fail_tag(folder.name)}_[{verdict}]"
    if folder.name == new_name:
        return None
    dest = folder.with_name(new_name)
    if dest.exists():
        raise FileExistsError(dest)
    folder.rename(dest)
    return dest


def _emit_step(on_step, step_id: str) -> None:
    if on_step is None:
        return
    try:
        on_step(step_id)
    except Exception:  # noqa: BLE001 — UI callback must not abort processing
        pass


def warp_align_folder(
    folder: Path,
    *,
    max_pad_sec: float,
    corr_min: float,
    drift_ms: float,
    window_corr_min: float,
    weak_window_frac: float,
    move_on_pass: bool,
    dry_run: bool,
    gaps_cut: bool = True,
    declick: str = "rx",
    on_step=None,
) -> WarpResult:
    result = WarpResult(folder=folder.name)
    aca_src, inst_src, orig = find_backup_stems(folder)
    if not aca_src or not inst_src or not orig:
        missing = []
        if not aca_src:
            missing.append("aca_backup")
        if not inst_src:
            missing.append("inst_backup")
        if not orig:
            missing.append("original")
        result.verdict = "skip"
        result.notes = f"missing:{','.join(missing)}"
        return result

    try:
        _ensure_rubberband_on_path()
        orig_m = peak_norm(load_mono(orig, SR_ANALYSIS))
        aca_m = peak_norm(load_mono(aca_src, SR_ANALYSIS))
        inst_m = peak_norm(load_mono(inst_src, SR_ANALYSIS))

        orig_vox = peak_norm(bandpass(orig_m, SR_ANALYSIS))
        aca_vox = peak_norm(bandpass(aca_m, SR_ANALYSIS))

        aca_pad, aca_score = chroma_xcorr_pad(
            orig_vox, aca_vox, sr=SR_ANALYSIS, max_pad_sec=max_pad_sec
        )
        from demucs_vocals import load_instrumental_mono

        _emit_step(on_step, "demucs")
        inst_ref = peak_norm(load_instrumental_mono(orig, SR_ANALYSIS, folder=folder))
        inst_pad, inst_score = chroma_xcorr_pad(
            inst_ref, inst_m, sr=SR_ANALYSIS, max_pad_sec=max_pad_sec
        )
        result.aca_pad_sec = aca_pad
        result.inst_pad_sec = inst_pad

        aca_p = fit_len(pad_or_trim_front(aca_m, aca_pad, SR_ANALYSIS), len(orig_m))
        inst_p = fit_len(pad_or_trim_front(inst_m, inst_pad, SR_ANALYSIS), len(inst_ref))
        aca_pv = peak_norm(bandpass(peak_norm(aca_p), SR_ANALYSIS))

        t_a, lag_a, sc_a = lag_curve(orig_vox, aca_pv, sr=SR_ANALYSIS)
        t_i, lag_i, sc_i = lag_curve(inst_ref, peak_norm(inst_p), sr=SR_ANALYSIS)
        t_a, lag_a = smooth_lags(t_a, lag_a, sc_a)
        t_i, lag_i = smooth_lags(t_i, lag_i, sc_i)
        lag_i, inst_bias = absorb_lag_offset(t_i, lag_i, sc_i)
        inst_pad += inst_bias
        result.inst_pad_sec = inst_pad

        if len(lag_a):
            result.aca_drift_range_ms = float((lag_a.max() - lag_a.min()) * 1000)
        if len(lag_i):
            result.inst_drift_range_ms = float((lag_i.max() - lag_i.min()) * 1000)

        result.notes = (
            f"engine=rubberband{'+aca_gaps' if gaps_cut else ''}; aca_score={aca_score:.3f}; inst_score={inst_score:.3f}"
            "; inst_elastique_ref=demucs_instrumental"
        )
        result.notes += "; aca_gaps=1" if gaps_cut else "; aca_gaps=0"
        if abs(inst_bias) >= 1e-4:
            result.notes += f"; inst_lag_offset={inst_bias:+.3f}s"
        declick_plan = resolve_declick(declick, host_rx=False)
        result.notes += f"; {declick_plan.note}"
        result.notes += "; inst_declick=off"

        if dry_run:
            result.verdict = 'dry_run'
            return result

        info = sf.info(str(orig))
        aca_dest = folder / aca_src.name
        if gaps_cut:
            from aca_gap_align import write_gap_aligned_acapella

            _emit_step(on_step, "silence")
            gap_info = write_gap_aligned_acapella(
                aca_src,
                orig,
                aca_dest,
                folder=folder,
                use_demucs_vocals=True,
            )
            plan = gap_info.get("plan", {})
            result.aca_pad_sec = float(plan.get("front_pad_sec", aca_pad))
            gap_notes = gap_info.get("notes") or []
            ref_note = next((n for n in gap_notes if n.startswith("ref_src=")), "")
            result.notes += (
                f"; aca_front={plan.get('front_pad_sec', 0):.3f}s"
                f"; aca_segs={plan.get('n_segments', 0)}"
            )
            if ref_note:
                result.notes += f"; {ref_note}"
            gaps = plan.get("gaps") or []
            if gaps:
                inserts = ",".join(f"{g['insert_sec']:.3f}" for g in gaps[:6])
                result.notes += f"; aca_gap_inserts={inserts}"
        else:
            shutil.copy2(aca_src, aca_dest)

        _emit_step(on_step, "align")
        write_warped_stem(
            inst_src,
            folder / inst_src.name,
            pad_sec=inst_pad,
            times=t_i,
            lags=lag_i,
            target_sr=info.samplerate,
            target_frames=info.frames,
            target_channels=info.channels,
        )

        _emit_step(on_step, "loudness")
        try:
            from demucs_vocals import match_aligned_stems_to_demucs_isolated

            loud = match_aligned_stems_to_demucs_isolated(
                orig,
                folder / aca_src.name,
                folder / inst_src.name,
                folder=folder,
            )
            result.notes += (
                f"; loudness_match aca={loud['aca_gain_db']:+.1f}dB"
                f" inst={loud['inst_gain_db']:+.1f}dB"
            )
        except Exception as exc:  # noqa: BLE001
            result.notes += f"; loudness_match_skip:{type(exc).__name__}:{exc}"

        # Rubber Band cannot host RX 11, so this engine leaves the vocal as stretched.

        _emit_step(on_step, "score")
        aca, inst, orig2, scan_notes = scan_folder(folder)
        if not aca or not inst or not orig2:
            result.verdict = 'error'
            result.notes += f'; post_scan:{scan_notes}'
            return result

        _verdict, corr, lag, drift, _weak, notes = analyze_alignment(
            aca,
            inst,
            orig2,
            corr_min=corr_min,
            drift_ms=drift_ms,
            window_corr_min=window_corr_min,
            weak_window_frac=weak_window_frac,
        )
        result.corr = corr
        result.lag_sec = lag
        result.drift_ms = drift
        result.notes += f'; {notes}'
        apply_stem_check(
            result,
            aca,
            inst,
            orig2,
            folder=folder,
            corr_min=corr_min,
            drift_ms=drift_ms,
            window_corr_min=window_corr_min,
            weak_window_frac=weak_window_frac,
        )
        verdict = result.verdict

        if move_on_pass and verdict in ("pass", "fail"):
            _emit_step(on_step, "tag")
            try:
                dest = tag_folder_verdict(folder, verdict)
            except FileExistsError:
                result.notes += "; dest_exists"
            except OSError as exc:
                result.notes += f"; rename_failed:{exc}"
            else:
                if dest is not None:
                    result.moved_to = str(dest)

    except Exception as exc:  # noqa: BLE001
        result.verdict = 'error'
        result.notes += f'; {type(exc).__name__}:{exc}'
    return result



def _worker(payload: dict) -> dict:
    return asdict(
        warp_align_folder(
            Path(payload["folder"]),
            max_pad_sec=payload["max_pad_sec"],
            corr_min=payload["corr_min"],
            drift_ms=payload["drift_ms"],
            window_corr_min=payload["window_corr_min"],
            weak_window_frac=payload["weak_window_frac"],
            move_on_pass=payload["move_on_pass"],
            dry_run=payload["dry_run"],
            declick=payload.get("declick", "rx"),
        )
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=FAIL_ALL)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-move", action="store_true")
    ap.add_argument("--max-pad-sec", type=float, default=DEFAULT_MAX_PAD_SEC)
    ap.add_argument("--corr-min", type=float, default=DEFAULT_CORR_MIN)
    ap.add_argument("--drift-ms", type=float, default=DEFAULT_DRIFT_MS)
    ap.add_argument("--window-corr-min", type=float, default=DEFAULT_WINDOW_CORR_MIN)
    ap.add_argument("--weak-window-frac", type=float, default=DEFAULT_WEAK_WINDOW_FRAC)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument(
        "--declick",
        choices=("rx", "off"),
        default="rx",
        help="Acapella de-click. This engine cannot host RX 11, so rx and off "
        "both leave the acapella untouched. The instrumental is not de-clicked.",
    )
    ap.add_argument("--csv", type=Path, default=OUT_DIR / "fail_all_warp_align.csv")
    ap.add_argument(
        "--skip-existing-pass",
        action="store_true",
        help="Skip folders already named _[pass] (n/a in fail_all)",
    )
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        print(f"ERROR: {root}", file=sys.stderr)
        return 2

    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if args.only:
        needles = [n.lower() for n in args.only]
        folders = [f for f in folders if any(n in f.name.lower() for n in needles)]
    if args.limit is not None:
        folders = folders[: args.limit]

    declick_plan = resolve_declick(args.declick, host_rx=False)
    print(f"De-click: {declick_plan.summary}")
    print(f"Root: {root}")
    print(f"Folders: {len(folders)}  workers={args.workers}  dry_run={args.dry_run}")
    print(
        f"Thresholds: corr_min={args.corr_min} drift_ms={args.drift_ms} max_pad={args.max_pad_sec}s"
    )

    payloads = [
        {
            "folder": str(f),
            "max_pad_sec": args.max_pad_sec,
            "corr_min": args.corr_min,
            "drift_ms": args.drift_ms,
            "window_corr_min": args.window_corr_min,
            "weak_window_frac": args.weak_window_frac,
            "move_on_pass": not args.no_move and not args.dry_run,
            "dry_run": args.dry_run,
            "declick": args.declick,
        }
        for f in folders
    ]

    rows: list[dict] = []

    def emit(row: dict) -> None:
        rows.append(row)
        moved = f" -> {Path(row['moved_to']).name}" if row.get("moved_to") else ""
        print(
            f"[{row['verdict'].upper()}] {row['folder']} "
            f"corr={row['corr']:.3f} drift={row['drift_ms']:.1f}ms "
            f"pads=({row['aca_pad_sec']:+.3f},{row['inst_pad_sec']:+.3f}) "
            f"drange=({row['aca_drift_range_ms']:.0f},{row['inst_drift_range_ms']:.0f})ms"
            f"{moved}",
            flush=True,
        )

    if args.workers <= 1:
        for i, p in enumerate(payloads, 1):
            emit(_worker(p))
            if i % 10 == 0:
                print(f"  progress {i}/{len(payloads)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, p): p["folder"] for p in payloads}
            done = 0
            for fut in as_completed(futs):
                try:
                    emit(fut.result())
                except Exception as exc:  # noqa: BLE001
                    emit(
                        asdict(
                            WarpResult(
                                folder=Path(futs[fut]).name,
                                verdict="error",
                                notes=f"worker:{type(exc).__name__}:{exc}",
                            )
                        )
                    )
                done += 1
                if done % 25 == 0 or done == len(payloads):
                    print(f"  progress {done}/{len(payloads)}", flush=True)

    rows.sort(key=lambda r: r["folder"])
    write_csv(args.csv, rows)
    counts: dict[str, int] = {}
    moved = sum(1 for r in rows if r.get("moved_to"))
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("---")
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Renamed in place: {moved}")
    print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
