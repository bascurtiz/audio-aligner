#!/usr/bin/env python3
"""Gap-aware acapella alignment.

Acapellas often have *internal* silence removed (not just head/tail). A single
global pad + continuous time-stretch cannot restore those missing rests.

Approach:
  1. Split the backup acapella on short phrase rests (~0.45s), not just long gaps
  2. Place each phrase on the original timeline via bandpassed waveform NCC
     (dry vocal vs vocal band of the mix is usually enough; optional Demucs
     vocals ref remains available but is not required)
  3. Only insert extra silence when the match is confident; otherwise keep the
     file's own gap (avoid cascading wrong placements)
  4. Render by pasting chunks (rigid, no stretch) with soft edge pads/fades
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

SR_ANALYSIS = 22050
HOP = 512
# chroma scores are typically 0.08–0.50 against a full mix
MIN_CHROMA_SCORE = 0.10
MIN_CHROMA_SCORE_WEAK = 0.08
MIN_PEAK_RATIO = 1.25
# local waveform NCC against demixed vocals (same domain)
MIN_WAVE_SCORE = 0.22
MIN_WAVE_SCORE_WEAK = 0.15
MIN_WAVE_RATIO = 1.20
# Phrases may need several seconds of restored silence, but must not leap to a
# later chorus (Boy Is Mine false-matched ~230s ahead when search was unbounded).
MAX_FORWARD_JUMP_SEC = 22.0
# The first phrase can sit several seconds before its time in the acapella.
# A 1s look-behind missed Gimme Shelter's opening (file at 12.3s, Demucs at
# 8.2s); chroma then locked onto the next entrance and every later section
# inherited that slide to the right. Forward search stays capped.
FIRST_LOOKBACK_SEC = 12.0
# A confident waveform match may restore a longer cut rest (Love Me Two Times
# needs ~29s). Still far below an unbounded chorus-repeat lock.
MAX_LONG_JUMP_SEC = 60.0
MIN_LONG_WAVE_SCORE = 0.26
MIN_LONG_WAVE_RATIO = 1.60


def peak_norm(y: np.ndarray) -> np.ndarray:
    p = float(np.max(np.abs(y)) or 0.0)
    return (y / p).astype(np.float32) if p > 1e-9 else y.astype(np.float32)


def bandpass(y: np.ndarray, sr: int, lo: float = 150.0, hi: float = 5000.0) -> np.ndarray:
    hi = min(hi, sr / 2 - 100)
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, y).astype(np.float32)


def load_mono(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def rms_env(y: np.ndarray, hop: int = HOP, frame: int = 2048) -> np.ndarray:
    if len(y) < frame:
        return np.zeros(1, dtype=np.float32)
    n = 1 + (len(y) - frame) // hop
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        sl = y[i * hop : i * hop + frame]
        out[i] = float(np.sqrt(np.mean(sl * sl)))
    return out


def silence_run_containing(
    y: np.ndarray,
    sr: int,
    t: float,
    *,
    hop: int = HOP,
    silence_db: float = -36.0,
) -> tuple[float, float] | None:
    """Quiet run that contains t, or None when t falls on singing."""
    if sr <= 0 or len(y) < hop:
        return None
    env = rms_env(y, hop=hop)
    peak = float(np.max(env) or 0.0)
    if peak < 1e-8:
        return 0.0, len(y) / sr
    thr = peak * (10 ** (silence_db / 20.0))
    quiet = env < thr
    idx = int(t * sr / hop)
    idx = min(max(idx, 0), len(quiet) - 1)
    if not quiet[idx]:
        return None
    a = b = idx
    while a > 0 and quiet[a - 1]:
        a -= 1
    while b + 1 < len(quiet) and quiet[b + 1]:
        b += 1
    return a * hop / sr, (b + 1) * hop / sr


def longest_silence_sec(
    y: np.ndarray,
    sr: int,
    t0: float,
    t1: float,
    *,
    hop: int = HOP,
    silence_db: float = -36.0,
) -> float:
    """Longest quiet run between t0 and t1, relative to the whole signal peak."""
    if t1 <= t0 + 0.05 or sr <= 0:
        return 0.0
    i0 = max(0, int(t0 * sr))
    i1 = min(len(y), int(t1 * sr))
    if i1 - i0 < hop:
        return max(0.0, t1 - t0) if i1 <= i0 else 0.0
    env = rms_env(y[i0:i1], hop=hop)
    # Peak of the full reference, not this window, so a loud phrase does not
    # turn its own softer notes into "silence".
    full = rms_env(y, hop=hop) if len(y) >= hop else env
    peak = float(np.max(full) or 0.0)
    if peak < 1e-8:
        return t1 - t0
    thr = peak * (10 ** (silence_db / 20.0))
    quiet = env < thr
    best = run = 0
    for flag in quiet:
        if flag:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best * hop / sr


def find_active_segments(
    y: np.ndarray,
    *,
    sr: int,
    hop: int = HOP,
    min_silence_sec: float = 0.90,
    min_active_sec: float = 2.0,
    silence_db: float = -42.0,
) -> list[tuple[float, float]]:
    """Split only on longer silences (structural gaps), not short breaths."""
    env = rms_env(y, hop=hop)
    peak = float(np.max(env) or 0.0)
    if peak < 1e-9:
        return []
    thr = peak * (10 ** (silence_db / 20.0))
    thr = max(thr, float(np.percentile(env, 20)) * 1.5, 1e-4)
    active = env >= thr
    min_sil = max(1, int(min_silence_sec * sr / hop))
    min_act = max(1, int(min_active_sec * sr / hop))

    segs: list[tuple[float, float]] = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n and active[j]:
            j += 1
        k = j
        while k < n:
            s0 = k
            while k < n and not active[k]:
                k += 1
            sil = k - s0
            if sil <= min_sil and k < n and active[k]:
                while k < n and active[k]:
                    k += 1
                j = k
            else:
                break
        if (j - i) >= min_act:
            segs.append((i * hop / sr, j * hop / sr))
        i = max(j, i + 1)
    return segs


def waveform_ncc_offset(
    ref: np.ndarray,
    qry: np.ndarray,
    *,
    sr: int,
    max_lag_sec: float | None = None,
    finger_sec: float = 12.0,
) -> tuple[float, float, float]:
    """Local-normalized waveform xcorr. Returns (lag_sec, score, peak_ratio)."""
    max_q = int(finger_sec * sr)
    if len(qry) > max_q:
        qry = qry[:max_q]
    if len(qry) < sr // 4 or len(ref) < len(qry):
        return 0.0, 0.0, 0.0

    qry = qry.astype(np.float64, copy=False)
    ref = ref.astype(np.float64, copy=False)
    qry = qry - float(np.mean(qry))
    q_norm = float(np.linalg.norm(qry)) or 1e-12

    corr = signal.correlate(ref, qry, mode="full", method="fft")
    lags = np.arange(-len(qry) + 1, len(ref))
    max_lag = len(ref) if max_lag_sec is None else int(max_lag_sec * sr)
    m = len(qry)
    valid = (lags >= 0) & (lags <= max_lag) & (lags + m <= len(ref))
    if not np.any(valid):
        return 0.0, 0.0, 0.0

    lv = lags[valid]
    csum = np.concatenate([[0.0], np.cumsum(ref * ref)])
    energies = csum[lv + m] - csum[lv]
    denom = np.sqrt(np.maximum(energies, 1e-18)) * q_norm
    ncc = corr[lv + m - 1] / denom
    bi = int(np.argmax(ncc))
    score = float(ncc[bi])
    best_lag = float(lv[bi])

    # distinct 2nd peak (≥1.5s away)
    sep = int(1.5 * sr)
    mask2 = np.abs(lv - int(best_lag)) >= sep
    if np.any(mask2):
        ratio = score / (float(np.max(ncc[mask2])) + 1e-9)
    else:
        ratio = 99.0
    return best_lag / sr, score, ratio


def chroma_xcorr_offset(
    ref: np.ndarray,
    qry: np.ndarray,
    *,
    sr: int,
    hop: int = HOP,
    max_lag_sec: float | None = None,
) -> tuple[float, float, float]:
    """Chroma-bin xcorr. Returns (lag_sec, score, peak_ratio vs distinct 2nd peak)."""
    min_len = max(4096, hop * 8)
    if len(qry) < min_len:
        qry = np.pad(qry, (0, min_len - len(qry)))
    if len(ref) < min_len:
        return 0.0, 0.0, 0.0

    rf = librosa.feature.chroma_cqt(y=ref, sr=sr, hop_length=hop)
    qf = librosa.feature.chroma_cqt(y=qry, sr=sr, hop_length=hop)
    rf = rf / np.maximum(np.linalg.norm(rf, axis=0, keepdims=True), 1e-9)
    qf = qf / np.maximum(np.linalg.norm(qf, axis=0, keepdims=True), 1e-9)

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
    hop_sec = hop / sr
    if max_lag_sec is None:
        max_lag_frames = rf.shape[1]
    else:
        max_lag_frames = int(max_lag_sec / hop_sec)
    valid = (lags >= 0) & (lags <= max_lag_frames)
    if not np.any(valid):
        return 0.0, 0.0, 0.0

    cv = corr_sum[valid]
    lv = lags[valid]
    bi = int(np.argmax(cv))
    denom = float(np.sqrt(e_r * e_q)) or 1e-12
    score = float(cv[bi]) / denom
    best_lag = float(lv[bi])

    mask2 = np.abs(lv.astype(np.float64) - best_lag) * hop_sec >= 3.0
    if np.any(mask2):
        second = float(np.max(cv[mask2])) / denom
        ratio = score / (second + 1e-9)
    else:
        ratio = 99.0

    return best_lag * hop_sec, score, ratio


def _ncc_near(
    ref: np.ndarray,
    qry: np.ndarray,
    t: float,
    sr: int,
    *,
    look_sec: float = 0.15,
    finger_sec: float = 8.0,
) -> float:
    """Waveform correlation of qry against ref at time t.

    Returns 0 when the peak in that neighborhood sits somewhere else, so a
    nearby but different phrase is not counted as a match at t.
    """
    if sr <= 0 or t < -0.05 or len(qry) < sr // 4 or len(ref) < sr // 4:
        return 0.0
    start = max(0.0, t - look_sec)
    i0 = int(start * sr)
    i1 = int(min(len(ref), (max(t, 0.0) + finger_sec + look_sec + 0.25) * sr))
    if i1 - i0 < sr // 4:
        return 0.0
    lag, score, _ratio = waveform_ncc_offset(
        ref[i0:i1],
        qry,
        sr=sr,
        max_lag_sec=look_sec * 2,
        finger_sec=finger_sec,
    )
    expected = t - start
    if abs(lag - expected) > 0.20:
        return 0.0
    return score


def _accept_match(
    *,
    score: float,
    ratio: float,
    dur: float,
    candidate: float,
    prev_dst_b: float,
    kind: str,
    earliest: float | None = None,
) -> bool:
    # Later phrases may not start before the previous one ends. The first
    # phrase passes earliest=search_start so it can move back onto an
    # earlier Demucs entrance.
    floor = prev_dst_b - 0.1 if earliest is None else earliest
    if candidate < floor:
        return False
    if kind == "wave":
        if score >= MIN_WAVE_SCORE:
            return True
        if score >= MIN_WAVE_SCORE_WEAK and ratio >= MIN_WAVE_RATIO and dur >= 2.5:
            return True
        if dur >= 8.0 and score >= 0.12 and ratio >= 1.15:
            return True
        return False
    # chroma
    if score >= MIN_CHROMA_SCORE:
        return True
    if score >= MIN_CHROMA_SCORE_WEAK and ratio >= MIN_PEAK_RATIO and dur >= 2.5:
        return True
    if dur >= 8.0 and score >= 0.08 and ratio >= 1.15:
        return True
    return False


def _match_offset(
    ref: np.ndarray,
    qry: np.ndarray,
    *,
    sr: int,
    prefer_wave: bool,
    max_lag_sec: float | None = None,
) -> tuple[float, float, float, str]:
    """Return (lag, score, ratio, kind). Prefer waveform when ref is demixed vocals."""
    if prefer_wave:
        lag, score, ratio = waveform_ncc_offset(
            ref, qry, sr=sr, max_lag_sec=max_lag_sec
        )
        if score >= MIN_WAVE_SCORE_WEAK:
            return lag, score, ratio, "wave"
    lag, score, ratio = chroma_xcorr_offset(
        ref, qry, sr=sr, max_lag_sec=max_lag_sec
    )
    return lag, score, ratio, "chroma"


def merge_segments_across_loud_gaps(
    y: np.ndarray,
    segs: list[tuple[float, float]],
    *,
    sr: int,
    min_ratio: float = 0.35,
) -> list[tuple[float, float]]:
    """Re-join splits where the bridge is nearly as loud as the phrases.

    Fine RMS splits often cut through quiet singing (Boy Is Mine). True rests
    (MaMaSé / Everywhere) stay much quieter than neighboring phrases.
    """
    if len(segs) <= 1:
        return segs
    out: list[tuple[float, float]] = [segs[0]]
    for a, b in segs[1:]:
        prev_a, prev_b = out[-1]
        g0, g1 = int(prev_b * sr), int(a * sr)
        s0 = y[int(prev_a * sr) : g0]
        s1 = y[int(a * sr) : int(b * sr)]
        bridge = y[g0:g1] if g1 > g0 else np.zeros(1, dtype=np.float32)
        r0 = float(np.sqrt(np.mean(s0 * s0))) if len(s0) else 0.0
        r1 = float(np.sqrt(np.mean(s1 * s1))) if len(s1) else 0.0
        rb = float(np.sqrt(np.mean(bridge * bridge))) if len(bridge) else 0.0
        neigh = min(r0, r1) + 1e-9
        if rb / neigh >= min_ratio:
            out[-1] = (prev_a, b)
        else:
            out.append((a, b))
    return out


def place_segments(
    aca_mono: np.ndarray,
    ref_mono: np.ndarray,
    *,
    sr: int,
    prefer_wave: bool = False,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]], list[str]]:
    notes: list[str] = []
    aca_v = peak_norm(bandpass(aca_mono, sr))
    ref_v = peak_norm(bandpass(ref_mono, sr))
    # Demucs/mix wave path: split on short rests so missing silence can be restored.
    if prefer_wave:
        segs = find_active_segments(
            aca_v,
            sr=sr,
            min_silence_sec=0.45,
            min_active_sec=0.7,
            silence_db=-48.0,
        )
        before = len(segs)
        segs = merge_segments_across_loud_gaps(aca_mono, segs, sr=sr)
        if len(segs) != before:
            notes.append(f"merged_loud_gaps {before}->{len(segs)}")
    else:
        segs = find_active_segments(aca_v, sr=sr)
    notes.append(f"ref={'vocals_wave' if prefer_wave else 'mix_chroma'}")

    if not segs:
        lag, score, _ratio, kind = _match_offset(
            ref_v, aca_v, sr=sr, prefer_wave=prefer_wave
        )
        dur = len(aca_v) / sr
        return (
            [(0.0, dur)],
            [(max(0.0, lag), max(0.0, lag) + dur, score)],
            notes + [f"single_segment_fallback_{kind}"],
        )

    if prefer_wave:
        # Keep phrase-level splits; only glue tiny crumbs separated by a breath.
        merged: list[tuple[float, float]] = []
        for a, b in segs:
            if merged and (a - merged[-1][1]) < 0.25 and (b - a) < 0.8:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        segs = merged
    else:
        merged = []
        for a, b in segs:
            if merged and (a - merged[-1][1]) < 1.2 and (b - a) < 2.5:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        segs = merged

    placements: list[tuple[float, float, float]] = []
    for i, (a, b) in enumerate(segs):
        seg = aca_v[int(a * sr) : int(b * sr)]
        dur = b - a

        if i == 0 and (not prefer_wave) and dur < 8.0 and len(segs) > 1:
            finger_end = min(segs[-1][1], a + 25.0)
            for _aa, bb in segs:
                if bb <= a + 25.0:
                    finger_end = bb
            finger = aca_v[int(a * sr) : int(finger_end * sr)]
            lag, score, ratio, kind = _match_offset(
                ref_v, finger, sr=sr, prefer_wave=prefer_wave
            )
            notes.append(
                f"seg0_finger={finger_end-a:.1f}s_{kind}={score:.3f}_ratio={ratio:.2f}"
            )
            place = max(0.0, lag)
            if (kind == "wave" and score < MIN_WAVE_SCORE_WEAK) or (
                kind == "chroma" and score < MIN_CHROMA_SCORE_WEAK
            ):
                notes.append(f"seg0_weak={score:.3f}")
            placements.append((place, place + dur, score))
            continue

        if i == 0:
            # The phrase already has a time in the acapella. Searching the
            # whole song forward locks a short hook onto a later repeat
            # (Be All Mine: the 1:23 hook scored 0.81 at 2:59, ratio only
            # 1.18) and every later phrase stays glued to that shift.
            # Look behind far enough to reach an earlier entrance.
            prev_dst_b = a
            file_gap = 0.0
            chained = a
            search_pad = FIRST_LOOKBACK_SEC
        else:
            prev_src_a, prev_src_b = segs[i - 1]
            prev_dst_a, prev_dst_b, _ = placements[-1]
            file_gap = a - prev_src_b
            chained = prev_dst_b + max(0.0, file_gap)
            search_pad = 2.0 if not prefer_wave else 1.0

        # Search only near the chain point — unbounded search locks onto
        # later repeats in songs with similar verses/choruses.
        search_start = max(0.0, chained - search_pad)
        earliest = search_start if i == 0 else None
        max_lag = MAX_FORWARD_JUMP_SEC + search_pad
        ref_end = min(len(ref_v) / sr, search_start + max_lag + dur + 1.0)
        ref = ref_v[int(search_start * sr) : int(ref_end * sr)]
        lag, score, ratio, kind = _match_offset(
            ref,
            seg,
            sr=sr,
            prefer_wave=prefer_wave,
            max_lag_sec=max_lag,
        )
        candidate = search_start + lag
        jump = candidate - prev_dst_b
        accepted = jump <= MAX_FORWARD_JUMP_SEC and _accept_match(
            score=score,
            ratio=ratio,
            dur=dur,
            candidate=candidate,
            prev_dst_b=prev_dst_b,
            kind=kind,
            earliest=earliest,
        )
        # A weak waveform peak can outrank chroma inside _match_offset and
        # then fail the accept test, hiding a chroma hit on the real phrase.
        if not accepted and kind == "wave":
            clag, cscore, cratio = chroma_xcorr_offset(
                ref, seg, sr=sr, max_lag_sec=max_lag
            )
            ccand = search_start + clag
            cjump = ccand - prev_dst_b
            if cjump <= MAX_FORWARD_JUMP_SEC and _accept_match(
                score=cscore,
                ratio=cratio,
                dur=dur,
                candidate=ccand,
                prev_dst_b=prev_dst_b,
                kind="chroma",
                earliest=earliest,
            ):
                candidate, score, ratio, kind = ccand, cscore, cratio, "chroma"
                jump = cjump
                accepted = True
        long_jump = False
        if not accepted:
            long_max = MAX_LONG_JUMP_SEC + search_pad
            long_end = min(len(ref_v) / sr, search_start + long_max + dur + 1.0)
            ref_long = ref_v[int(search_start * sr) : int(long_end * sr)]
            lag_l, score_l, ratio_l, kind_l = _match_offset(
                ref_long,
                seg,
                sr=sr,
                prefer_wave=prefer_wave,
                max_lag_sec=long_max,
            )
            cand_l = search_start + lag_l
            jump_l = cand_l - prev_dst_b
            if (
                kind_l == "wave"
                and score_l >= MIN_LONG_WAVE_SCORE
                and ratio_l >= MIN_LONG_WAVE_RATIO
                and MAX_FORWARD_JUMP_SEC < jump_l <= MAX_LONG_JUMP_SEC
                and cand_l >= prev_dst_b - 0.1
            ):
                candidate, score, ratio, kind = cand_l, score_l, ratio_l, kind_l
                accepted = True
                long_jump = True

        # Chaining into a long Demucs rest puts singing where the vocal
        # is silent. The real entrance is often just past the 22s window
        # (Crystal Ship: 1:08 in the output, 1:30 in Demucs).
        if not accepted:
            rest_run = silence_run_containing(ref_v, sr, chained)
            if rest_run is not None and (rest_run[1] - rest_run[0]) >= 2.5:
                rescue_max = MAX_LONG_JUMP_SEC + search_pad
                rescue_end = min(
                    len(ref_v) / sr, search_start + rescue_max + dur + 1.0
                )
                ref_rescue = ref_v[int(search_start * sr) : int(rescue_end * sr)]
                rlag, rscore, rratio, rkind = _match_offset(
                    ref_rescue,
                    seg,
                    sr=sr,
                    prefer_wave=prefer_wave,
                    max_lag_sec=rescue_max,
                )
                rcand = search_start + rlag
                rjump = rcand - prev_dst_b
                wave_ok = rjump <= MAX_LONG_JUMP_SEC and _accept_match(
                    score=rscore,
                    ratio=rratio,
                    dur=dur,
                    candidate=rcand,
                    prev_dst_b=prev_dst_b,
                    kind=rkind,
                    earliest=earliest,
                )
                if rkind == "wave" and not wave_ok:
                    clag, cscore, cratio = chroma_xcorr_offset(
                        ref_rescue, seg, sr=sr, max_lag_sec=rescue_max
                    )
                    rcand = search_start + clag
                    rjump = rcand - prev_dst_b
                    rscore, rratio, rkind = cscore, cratio, "chroma"
                if (
                    rest_run[1] - 0.4 <= rcand
                    and rjump <= MAX_LONG_JUMP_SEC
                    and _accept_match(
                        score=rscore,
                        ratio=rratio,
                        dur=dur,
                        candidate=rcand,
                        prev_dst_b=prev_dst_b,
                        kind=rkind,
                        earliest=earliest,
                    )
                ):
                    candidate, score, ratio, kind = rcand, rscore, rratio, rkind
                    jump = rjump
                    accepted = True
                    notes.append(
                        f"seg{i}_silence_resume={rcand:.2f}s_{rkind}={rscore:.3f}"
                    )

        # Don't open a rest the reference vocal does not have. A phrase
        # match across continuous singing is a later repeat, not a cut gap.
        span_ok = True
        if accepted and i > 0:
            insert_try = candidate - prev_dst_b - file_gap
            if insert_try > 0.8:
                rest = longest_silence_sec(ref_v, sr, prev_dst_b, candidate)
                # A short dip must not open a multi-second hole. The insert
                # has to fit the silence Demucs actually has (Brass In Pocket
                # added 4s and 16s across a 1s dip).
                if rest < 1.0 or insert_try > rest + 0.5:
                    # A phrase that already matches at the chain is a later
                    # repeat (Brass In Pocket: 4s and 16s across a 1s dip).
                    # A phrase that misses the chain and hits later is the
                    # vocal this acapella was missing, even when Demucs is
                    # not silent there (MaMaSé).
                    chain_ncc = (
                        _ncc_near(ref_v, seg, chained, sr) if kind == "wave" else 1.0
                    )
                    if (
                        kind == "wave"
                        and score >= 0.55
                        and chain_ncc <= 0.25
                        and score >= chain_ncc + 0.40
                    ):
                        notes.append(
                            f"seg{i}_span_keep={insert_try:.2f}s_rest={rest:.2f}s"
                            f"_wave={score:.3f}_chain={chain_ncc:.3f}"
                        )
                    else:
                        span_ok = False
                        notes.append(
                            f"seg{i}_span_reject={insert_try:.2f}s_rest={rest:.2f}s"
                            f"_{kind}={score:.3f}_chain={chain_ncc:.3f}"
                        )

        # Hard cap: past the long-jump ceiling, keep the file's own gap.
        if jump > MAX_LONG_JUMP_SEC and not accepted:
            notes.append(
                f"seg{i}_jump_reject={jump:.1f}s>"
                f"{MAX_LONG_JUMP_SEC:.0f}s_{kind}={score:.3f}"
            )
            place = chained
            score = 0.0
        elif accepted and span_ok:
            place = candidate
            insert = place - prev_dst_b - file_gap
            tag = "long_insert" if long_jump else "insert"
            if insert > 0.15:
                notes.append(
                    f"seg{i}_{tag}={insert:.3f}s_{kind}={score:.3f}_ratio={ratio:.2f}"
                )
            else:
                notes.append(f"seg{i}_match_{kind}={score:.3f}")
        else:
            place = chained
            if jump > MAX_FORWARD_JUMP_SEC:
                notes.append(
                    f"seg{i}_jump_reject={jump:.1f}s>"
                    f"{MAX_FORWARD_JUMP_SEC:.0f}s_{kind}={score:.3f}"
                )
            else:
                notes.append(
                    f"seg{i}_chain_{kind}={score:.3f}_ratio={ratio:.2f}"
                )
            score = 0.0

        placements.append((place, place + dur, score))

    return segs, placements, notes


def snap_tiny_inserts(
    segs: list[tuple[float, float]],
    placements: list[tuple[float, float, float]],
    *,
    snap_sec: float = 0.06,
) -> tuple[list[tuple[float, float, float]], int]:
    """Drop sub-60ms join corrections so the source waveform stays intact.

    A cut through a sung note is a pop. Placement errors this small are inside
    the drift budget, so the phrase stays sample-contiguous with the one before
    it. Real rests (inserts larger than snap_sec) are left alone.
    """
    if len(placements) < 2:
        return placements, 0
    out = list(placements)
    snapped = 0
    for i in range(1, len(segs)):
        prev_p1 = out[i - 1][1]
        p0 = out[i][0]
        file_gap = segs[i][0] - segs[i - 1][1]
        insert = (p0 - prev_p1) - file_gap
        if abs(insert) <= snap_sec:
            # Move this phrase and everything after it, so the correction
            # does not pile up as a bigger cut at the next join.
            delta = (prev_p1 + file_gap) - p0
            for j in range(i, len(out)):
                a, b, s = out[j]
                out[j] = (a + delta, b + delta, s)
            snapped += 1
    return out, snapped


def gap_plan(
    segs: list[tuple[float, float]],
    placements: list[tuple[float, float, float]],
) -> dict:
    front = placements[0][0] - segs[0][0]
    gaps = []
    for i in range(1, len(segs)):
        file_gap = segs[i][0] - segs[i - 1][1]
        desired = placements[i][0] - placements[i - 1][1]
        gaps.append(
            {
                "after_seg": i - 1,
                "file_gap_sec": file_gap,
                "desired_gap_sec": desired,
                "insert_sec": desired - file_gap,
            }
        )
    return {"front_pad_sec": front, "gaps": gaps, "n_segments": len(segs)}


def expand_segments_with_edges(
    segs: list[tuple[float, float]],
    placements: list[tuple[float, float, float]],
    *,
    aca_dur_sec: float,
    edge_pad_sec: float = 0.28,
    release_pad_sec: float = 0.35,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]]]:
    """Pull pre/post silence from the source so phrase edges aren't gated.

    Releases get a longer pad than attacks (chopped tails are what sounds gated).
    Core onset placement stays fixed: left-pad shifts the paste start earlier.
    Pads are clamped so neighboring pastes don't overlap.
    """
    if not segs:
        return segs, placements
    n = len(segs)
    pad_l = [0.0] * n
    pad_r = [0.0] * n

    for i, (a, b) in enumerate(segs):
        prev_b = segs[i - 1][1] if i > 0 else 0.0
        next_a = segs[i + 1][0] if i + 1 < n else aca_dur_sec
        left_room = max(0.0, a - prev_b)
        right_room = max(0.0, next_a - b)
        # Prefer taking most of the available file quiet — leave a tiny seam
        pad_l[i] = min(edge_pad_sec, 0.85 * left_room) if i > 0 else min(edge_pad_sec, left_room)
        pad_r[i] = (
            min(release_pad_sec, 0.85 * right_room)
            if i + 1 < n
            else min(release_pad_sec, right_room)
        )

    for i in range(n - 1):
        _p0_i, p1_i, _ = placements[i]
        p0_n, _p1_n, _ = placements[i + 1]
        budget = max(0.0, p0_n - p1_i)
        need = pad_r[i] + pad_l[i + 1]
        if need > budget and need > 1e-9:
            # Favor keeping release pad when squeezing into a tight dest gap
            release = min(pad_r[i], budget * 0.65)
            attack = min(pad_l[i + 1], max(0.0, budget - release))
            # if still over, scale both
            if release + attack > budget:
                scale = budget / (release + attack)
                release *= scale
                attack *= scale
            pad_r[i] = release
            pad_l[i + 1] = attack

    out_segs: list[tuple[float, float]] = []
    out_places: list[tuple[float, float, float]] = []
    for i, ((a, b), (p0, _p1, sc)) in enumerate(zip(segs, placements)):
        a2 = max(0.0, a - pad_l[i])
        b2 = min(aca_dur_sec, b + pad_r[i])
        p0_2 = p0 - pad_l[i]
        out_segs.append((a2, b2))
        out_places.append((p0_2, p0_2 + (b2 - a2), sc))
    return out_segs, out_places


def _edge_fade(
    seg: np.ndarray,
    sr: int,
    *,
    fade_in_sec: float = 0.035,
    fade_out_sec: float = 0.12,
) -> np.ndarray:
    """Asymmetric cosine fades — short attack, longer release (anti-gate)."""
    n = len(seg)
    if n < 8:
        return seg
    fade_in = min(int(round(fade_in_sec * sr)), n // 4) if fade_in_sec > 0 else 0
    fade_out = min(int(round(fade_out_sec * sr)), n // 3) if fade_out_sec > 0 else 0
    out = seg.astype(np.float32, copy=True)
    if fade_in >= 2:
        w = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_in, dtype=np.float32))
        if out.ndim == 2:
            out[:fade_in] *= w[:, None]
        else:
            out[:fade_in] *= w
    if fade_out >= 2:
        w = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_out, dtype=np.float32))
        if out.ndim == 2:
            out[-fade_out:] *= w[::-1, None]
        else:
            out[-fade_out:] *= w[::-1]
    return out


def render_gap_aligned(
    aca_audio: np.ndarray,
    *,
    sr: int,
    segs: list[tuple[float, float]],
    placements: list[tuple[float, float, float]],
    target_frames: int,
    channels: int,
    fade_in_sec: float = 0.035,
    fade_out_sec: float = 0.12,
) -> np.ndarray:
    """Paste *all* source audio; only add silence where placement inserts it.

    Between phrases, the file's own bridge audio is kept. Extra destination
    silence (restored missing rests) sits after that bridge.
    """
    if aca_audio.ndim == 1:
        aca_audio = aca_audio[:, None]
    n_ch = aca_audio.shape[1]
    out = np.zeros((target_frames, max(n_ch, channels)), dtype=np.float32)
    written = np.zeros(target_frames, dtype=bool)
    n_src = len(aca_audio)
    # Overlaps used to be summed, which doubles the waveform (crackle/pops).
    xfade = max(8, int(round(0.012 * sr)))
    hole_close = max(2, int(round(0.002 * sr)))

    def _paste(
        dst0: int,
        seg: np.ndarray,
        *,
        fade_in: bool,
        fade_out: bool,
        fade_in_override: float | None = None,
        fade_out_override: float | None = None,
    ) -> None:
        if seg.size == 0:
            return
        if seg.ndim == 1:
            seg = seg[:, None]
        if fade_in or fade_out:
            seg = _edge_fade(
                seg,
                sr,
                fade_in_sec=(fade_in_override if fade_in_override is not None else fade_in_sec)
                if fade_in
                else 0.0,
                fade_out_sec=(fade_out_override if fade_out_override is not None else fade_out_sec)
                if fade_out
                else 0.0,
            )
        if dst0 >= target_frames:
            return
        if dst0 < 0:
            seg = seg[-dst0:]
            dst0 = 0
        if len(seg) <= 0:
            return
        if dst0 + len(seg) > target_frames:
            seg = seg[: target_frames - dst0]
        n = len(seg)
        if n <= 0:
            return
        # A 1–2ms hole is a click. Abut the next piece. Pulling it back by the
        # full crossfade mixes a continuous vocal with itself a few ms off,
        # which is another pop.
        if dst0 > 0 and written[:dst0].any():
            last = int(np.flatnonzero(written[:dst0])[-1])
            gap = dst0 - last - 1
            if 0 < gap <= hole_close:
                dst0 = last + 1
                if dst0 + n > target_frames:
                    seg = seg[: target_frames - dst0]
                    n = len(seg)
        n_ov = 0
        while n_ov < n and dst0 + n_ov < target_frames and written[dst0 + n_ov]:
            n_ov += 1
        fade = min(xfade, n_ov, n)
        if fade >= 2:
            # Linear blend. These joins are the same vocal, so equal-power
            # would peak at +3 dB and clip.
            w = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
            out[dst0 : dst0 + fade, :n_ch] = (
                out[dst0 : dst0 + fade, :n_ch] * (1.0 - w) + seg[:fade, :n_ch] * w
            )
        if n_ov > fade:
            out[dst0 + fade : dst0 + n_ov, :n_ch] = seg[fade:n_ov, :n_ch]
        if n_ov < n:
            out[dst0 + n_ov : dst0 + n, :n_ch] = seg[n_ov:, :n_ch]
        written[dst0 : dst0 + n] = True

    if not segs:
        return out[:, :channels] if out.shape[1] >= channels else out

    # Leading source audio before first active seg
    a0, b0 = segs[0]
    p0, _p1, _ = placements[0]
    if a0 > 0:
        lead = aca_audio[0 : int(round(a0 * sr))]
        _paste(int(round((p0 - a0) * sr)), lead, fade_in=True, fade_out=False)

    for i, ((a, b), (p0, _p1, _sc)) in enumerate(zip(segs, placements)):
        src0 = int(round(a * sr))
        src1 = int(round(b * sr))
        dst0 = int(round(p0 * sr))
        # Fade into a phrase only when the previous piece does not connect.
        # Fading a continuous join (lead straight into phrase 0) is itself a click.
        fade_in = False
        short_edge = False
        if i == 0:
            lead_end = int(round((p0 - a0) * sr)) + int(round(a0 * sr))
            fade_in = abs(lead_end - dst0) > hole_close
        else:
            prev_b = segs[i - 1][1]
            prev_p0, prev_p1, _ = placements[i - 1]
            file_gap = a - prev_b
            desired = p0 - prev_p1
            insert_before = desired - file_gap
            fade_in = insert_before > 0.004
            short_edge = fade_in and insert_before <= 0.08

        _paste(
            dst0,
            aca_audio[src0:src1],
            fade_in=fade_in,
            fade_out=False,
            fade_in_override=0.008 if short_edge else None,
        )

        # File bridge audio between this seg and the next (never drop it)
        if i + 1 < len(segs):
            next_a, _next_b = segs[i + 1]
            next_p0 = placements[i + 1][0]
            g0 = int(round(b * sr))
            g1 = int(round(next_a * sr))
            bridge = aca_audio[g0:g1]
            bridge_dst = dst0 + (src1 - src0)
            file_gap = next_a - b
            desired = next_p0 - (p0 + (b - a))
            insert = desired - file_gap
            # Continuous into the bridge. Fade the bridge out only when extra
            # silence follows, including short inserts that would otherwise click.
            _paste(
                bridge_dst,
                bridge,
                fade_in=False,
                fade_out=insert > 0.004,
                fade_out_override=0.008 if 0.004 < insert <= 0.08 else None,
            )

    # Trailing source after last seg
    _aL, bL = segs[-1]
    p0L, p1L, _ = placements[-1]
    t0 = int(round(bL * sr))
    if t0 < n_src:
        trail = aca_audio[t0:]
        _paste(int(round(p1L * sr)), trail, fade_in=False, fade_out=True)

    if out.shape[1] < channels:
        out = np.pad(out, ((0, 0), (0, channels - out.shape[1])))
    elif out.shape[1] > channels:
        out = out[:, :channels]
    return out


def write_gap_aligned_acapella(
    aca_src: Path,
    orig_path: Path,
    dest: Path,
    *,
    analysis_sr: int = SR_ANALYSIS,
    use_demucs_vocals: bool = False,
    folder: Path | None = None,
    fade_in_sec: float = 0.035,
    fade_out_sec: float = 0.12,
) -> dict:
    info = sf.info(str(orig_path))
    target_sr = info.samplerate
    target_frames = info.frames
    target_ch = info.channels

    aca_m = load_mono(aca_src, analysis_sr)
    prefer_wave = True
    ref_note = "mix"
    if use_demucs_vocals:
        try:
            from demucs_vocals import load_vocals_mono

            ref_m = load_vocals_mono(
                orig_path, analysis_sr, folder=folder or dest.parent
            )
            ref_note = "demucs_vocals"
        except Exception as exc:  # noqa: BLE001
            ref_m = load_mono(orig_path, analysis_sr)
            ref_note = f"mix_fallback:{type(exc).__name__}"
    else:
        ref_m = load_mono(orig_path, analysis_sr)

    segs, placements, notes = place_segments(
        aca_m, ref_m, sr=analysis_sr, prefer_wave=prefer_wave
    )
    placements, n_snap = snap_tiny_inserts(segs, placements)
    if n_snap:
        notes.append(f"snapped_joins={n_snap}")
    notes = [f"ref_src={ref_note}"] + notes
    plan = gap_plan(segs, placements)
    notes.append(f"fade={fade_in_sec:.3f}/{fade_out_sec:.3f}s_preserve_bridges")

    y, file_sr = sf.read(str(aca_src), always_2d=True, dtype="float32")
    if file_sr != target_sr:
        y = np.stack(
            [
                librosa.resample(y[:, c], orig_sr=file_sr, target_sr=target_sr)
                for c in range(y.shape[1])
            ],
            axis=1,
        ).astype(np.float32)

    out = render_gap_aligned(
        y,
        sr=target_sr,
        segs=segs,
        placements=placements,
        target_frames=target_frames,
        channels=target_ch,
        fade_in_sec=fade_in_sec,
        fade_out_sec=fade_out_sec,
    )

    tmp = dest.with_name(dest.stem + "._gap_tmp" + dest.suffix)
    if tmp.exists():
        tmp.unlink()
    sf.write(str(tmp), out, target_sr, format="FLAC")
    tmp.replace(dest)

    return {
        "plan": plan,
        "placements": [
            {"src": [a, b], "dst": [p0, p1], "score": sc}
            for (a, b), (p0, p1, sc) in zip(segs, placements)
        ],
        "notes": notes,
    }
