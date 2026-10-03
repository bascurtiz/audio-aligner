"""One monotonic map: local phrase stretch, then silence for the missing rest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from align_model.quality import GAP_CONFIDENCE


@dataclass
class PhraseUnit:
    src_start: float
    src_end: float
    dst_start: float
    dst_end: float
    start_offset: float
    end_offset: float
    duration_delta: float
    confidence: float
    match_margin: float


@dataclass
class MapSpan:
    kind: str
    src_start: float
    src_end: float
    dst_start: float
    dst_end: float


def _margin_from_ratio(score: float, ratio: float) -> float:
    if ratio <= 1.0:
        return 0.0
    return float(score) * (1.0 - 1.0 / float(ratio))


def _end_offset(
    ref: np.ndarray,
    aca: np.ndarray,
    sr: int,
    src_a: float,
    src_b: float,
    dst_a: float,
    start_offset: float,
) -> float:
    """Lag at the end of the phrase. Falls back to the start offset when the end is weak."""
    dur = src_b - src_a
    finger = min(2.0, max(0.35, dur * 0.4))
    if dur < 0.8:
        return start_offset
    src0 = src_b - finger
    i0 = int(max(0.0, src0) * sr)
    i1 = int(min(len(aca), src_b * sr))
    chunk = aca[i0:i1]
    if chunk.size < sr // 5:
        return start_offset
    expected = dst_a + (src0 - src_a)
    search0 = max(0.0, expected - 0.45)
    search1 = min(len(ref) / sr, expected + finger + 0.45)
    j0 = int(search0 * sr)
    j1 = int(search1 * sr)
    if j1 - j0 < chunk.size:
        return start_offset
    from aca_gap_align import waveform_ncc_offset

    lag, score, _ratio = waveform_ncc_offset(
        ref[j0:j1],
        chunk,
        sr=sr,
        max_lag_sec=0.9,
        finger_sec=finger + 0.1,
    )
    if score < 0.12:
        return start_offset
    matched = search0 + lag
    return float((matched + finger) - src_b)


def phrase_units(aca: np.ndarray, ref: np.ndarray, sr: int) -> tuple[list[PhraseUnit], list[str]]:
    """Place phrases, then measure a separate lag at each end."""
    from aca_gap_align import place_segments, snap_tiny_inserts, waveform_ncc_offset

    segs, placements, notes = place_segments(aca, ref, sr=sr, prefer_wave=True)
    placements, n_snap = snap_tiny_inserts(segs, placements)
    if n_snap:
        notes.append(f"snapped_joins={n_snap}")
    units: list[PhraseUnit] = []
    for (src_a, src_b), (dst_a, dst_b, score) in zip(segs, placements):
        dur = max(0.05, src_b - src_a)
        i0 = int(max(0.0, src_a) * sr)
        i1 = int(min(len(aca), src_b * sr))
        chunk = aca[i0:i1]
        search0 = max(0.0, dst_a - 0.3)
        search1 = min(len(ref) / sr, dst_b + 0.3)
        ratio = 99.0
        if chunk.size > sr // 5 and search1 > search0:
            _lag, wave_score, ratio = waveform_ncc_offset(
                ref[int(search0 * sr) : int(search1 * sr)],
                chunk,
                sr=sr,
                max_lag_sec=max(0.6, search1 - search0),
                finger_sec=min(8.0, dur),
            )
            if wave_score > 0:
                score = float(wave_score)
        start_offset = float(dst_a - src_a)
        end_offset = _end_offset(ref, aca, sr, src_a, src_b, dst_a, start_offset)
        dst_start = float(src_a + start_offset)
        dst_end = float(src_b + end_offset)
        if dst_end < dst_start + 0.05:
            dst_end = dst_start + dur
            end_offset = start_offset
        units.append(
            PhraseUnit(
                src_start=float(src_a),
                src_end=float(src_b),
                dst_start=dst_start,
                dst_end=dst_end,
                start_offset=start_offset,
                end_offset=float(end_offset),
                duration_delta=float((dst_end - dst_start) - (src_b - src_a)),
                confidence=float(score),
                match_margin=_margin_from_ratio(float(score), float(ratio)),
            )
        )
    return units, notes


def spans_from_units(units: list[PhraseUnit], duration: float) -> list[MapSpan]:
    """Speech runs stretch. A stall between them is silence, not smeared audio."""
    spans: list[MapSpan] = []
    prev_dst = 0.0
    ordered = sorted(units, key=lambda unit: unit.dst_start)
    for unit in ordered:
        dst0 = max(float(unit.dst_start), prev_dst)
        dst1 = max(float(unit.dst_end), dst0 + 0.05)
        src0 = float(unit.src_start)
        src1 = max(float(unit.src_end), src0 + 0.05)
        if dst0 > prev_dst + 0.02:
            spans.append(MapSpan("silence", src0, src0, prev_dst, dst0))
        spans.append(MapSpan("speech", src0, src1, dst0, dst1))
        prev_dst = dst1
    duration = max(float(duration), prev_dst)
    if duration > prev_dst + 0.05:
        tail_src = ordered[-1].src_end if ordered else 0.0
        spans.append(MapSpan("silence", tail_src, tail_src, prev_dst, duration))
    return spans


def series_from_spans(
    spans: list[MapSpan],
    *,
    step: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Lag samples on speech only. Silence is returned as gap intervals."""
    times: list[float] = []
    lags: list[float] = []
    gaps: list[list[float]] = []
    for span in spans:
        if span.kind == "silence":
            if span.dst_end - span.dst_start >= 0.02:
                gaps.append([float(span.dst_start), float(span.dst_end)])
            continue
        dur = span.dst_end - span.dst_start
        src_dur = max(1e-3, span.src_end - span.src_start)
        n = max(2, int(round(dur / max(step, 0.1))) + 1)
        for i in range(n):
            u = i / (n - 1)
            dst = span.dst_start + u * dur
            src = span.src_start + u * src_dur
            times.append(float(dst))
            lags.append(float(dst - src))
    return np.asarray(times, dtype=float), np.asarray(lags, dtype=float), gaps


def gap_confidence(units: list[PhraseUnit]) -> float:
    if not units:
        return 0.0
    weights = np.array([max(unit.dst_end - unit.dst_start, 0.05) for unit in units])
    scores = np.array([unit.confidence for unit in units])
    return float(np.average(scores, weights=weights))


def uncertain_units(units: list[PhraseUnit]) -> list[PhraseUnit]:
    return [
        unit
        for unit in units
        if unit.confidence < GAP_CONFIDENCE or unit.match_margin < 0.05
    ]


def _fit_length(chunk: np.ndarray, n_out: int) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    if chunk.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    if chunk.size == n_out:
        return chunk
    src_x = np.linspace(0.0, 1.0, chunk.size)
    dst_x = np.linspace(0.0, 1.0, n_out)
    return np.interp(dst_x, src_x, chunk).astype(np.float32)


def _stretch_channel(chunk: np.ndarray, n_out: int, sr: int, engine: str) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    if chunk.size < 8 or abs(chunk.size - n_out) <= 2 or engine == "linear":
        return _fit_length(chunk, n_out)
    rate = float(np.clip(chunk.size / max(n_out, 1), 0.25, 4.0))
    try:
        import pyrubberband as pyrb

        stretched = pyrb.time_stretch(chunk.astype(np.float64), sr, rate).astype(np.float32)
    except Exception:
        return _fit_length(chunk, n_out)
    return _fit_length(stretched, n_out)


def _reaper_stretch_many(
    chunks: list[np.ndarray],
    n_outs: list[int],
    sr: int,
    reaper_exe,
) -> list[np.ndarray]:
    """One REAPER pass stretches every speech span to its destination length."""
    import tempfile
    from pathlib import Path

    import soundfile as sf

    from warp_align_reaper import run_reaper_jobs

    rendered: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="gap_phrase_") as tmp:
        tmp_path = Path(tmp)
        jobs = []
        outs = []
        for i, (chunk, n_out) in enumerate(zip(chunks, n_outs)):
            src = tmp_path / f"in_{i}.wav"
            dest = tmp_path / f"out_{i}.wav"
            audio = chunk.astype(np.float32)
            if audio.ndim == 1:
                audio = audio[:, None]
            sf.write(str(src), audio, sr, subtype="FLOAT")
            in_sec = len(chunk) / sr if chunk.ndim == 1 else len(chunk) / sr
            out_sec = n_out / sr
            jobs.append(
                {
                    "input": str(src).replace("\\", "/"),
                    "output": str(dest).replace("\\", "/"),
                    "target_length_sec": out_sec,
                    "sample_rate": sr,
                    "markers": [
                        {"src": 0.0, "dst": 0.0},
                        {"src": float(in_sec), "dst": float(out_sec)},
                    ],
                }
            )
            outs.append(dest)
        run_reaper_jobs(reaper_exe, jobs, timeout_sec=240.0, expected_outputs=outs)
        for dest, n_out in zip(outs, n_outs):
            y, file_sr = sf.read(str(dest), always_2d=True, dtype="float32")
            if file_sr != sr and len(y):
                import librosa

                y = np.stack(
                    [librosa.resample(y[:, c], orig_sr=file_sr, target_sr=sr) for c in range(y.shape[1])],
                    axis=1,
                ).astype(np.float32)
            rendered.append(y[:n_out] if len(y) >= n_out else np.pad(y, ((0, n_out - len(y)), (0, 0))))
    return rendered


def render_gap_aware(
    y: np.ndarray,
    sr: int,
    spans: list[MapSpan],
    target_frames: int,
    *,
    engine: str = "rubberband",
    reaper_exe=None,
    notes: list[str] | None = None,
) -> np.ndarray:
    """Place stretched phrases on the reference timeline and write silence in the stalls."""
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    channels = audio.shape[1]
    out = np.zeros((max(1, int(target_frames)), channels), dtype=np.float32)
    speech = [span for span in spans if span.kind == "speech"]
    reaper_chunks = None
    if engine == "reaper" and reaper_exe is not None and speech:
        pieces = []
        n_outs = []
        for span in speech:
            s0 = int(np.clip(round(span.src_start * sr), 0, len(audio)))
            s1 = int(np.clip(round(span.src_end * sr), s0, len(audio)))
            n_out = max(1, int(round((span.dst_end - span.dst_start) * sr)))
            pieces.append(audio[s0:s1] if s1 > s0 else np.zeros((1, channels), dtype=np.float32))
            n_outs.append(n_out)
        try:
            reaper_chunks = _reaper_stretch_many(pieces, n_outs, sr, reaper_exe)
        except Exception as exc:
            reaper_chunks = None
            if notes is not None:
                notes.append(f"phrase_stretch=rubberband:{type(exc).__name__}")
    elif engine == "reaper" and notes is not None and speech:
        notes.append("phrase_stretch=rubberband")
    fade = max(1, int(0.008 * sr))
    speech_i = 0
    for span in spans:
        if span.kind != "speech":
            continue
        i0 = int(round(span.dst_start * sr))
        n_out = max(1, int(round((span.dst_end - span.dst_start) * sr)))
        if reaper_chunks is not None:
            stretched = reaper_chunks[speech_i]
            speech_i += 1
        else:
            s0 = int(np.clip(round(span.src_start * sr), 0, len(audio)))
            s1 = int(np.clip(round(span.src_end * sr), s0, len(audio)))
            chunk = audio[s0:s1]
            columns = [
                _stretch_channel(chunk[:, c] if chunk.size else np.zeros(0), n_out, sr, engine)
                for c in range(channels)
            ]
            stretched = np.stack(columns, axis=1) if columns else np.zeros((n_out, channels), dtype=np.float32)
        if stretched.shape[0] != n_out:
            stretched = np.stack([_fit_length(stretched[:, c], n_out) for c in range(stretched.shape[1])], axis=1)
        if stretched.shape[1] != channels:
            if stretched.shape[1] < channels:
                stretched = np.pad(stretched, ((0, 0), (0, channels - stretched.shape[1])))
            else:
                stretched = stretched[:, :channels]
        n_fade = min(fade, max(1, n_out // 8))
        if n_out > n_fade * 2:
            ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
            stretched[:n_fade] *= ramp[:, None]
            stretched[-n_fade:] *= ramp[::-1, None]
        j0 = max(0, i0)
        j1 = min(len(out), i0 + n_out)
        if j1 <= j0:
            speech_i += 0
            continue
        a0 = j0 - i0
        out[j0:j1] = stretched[a0 : a0 + (j1 - j0)]
        if reaper_chunks is None:
            speech_i += 1
    return out


def write_gap_file(
    src,
    dest,
    spans: list[MapSpan],
    *,
    target_sr: int,
    target_frames: int,
    target_channels: int,
    engine: str = "rubberband",
    reaper_exe=None,
) -> list[str]:
    """Render the gap-aware map into ``dest`` at the original's rate and length."""
    import soundfile as sf

    y, file_sr = sf.read(str(src), always_2d=True, dtype="float32")
    if file_sr != target_sr:
        import librosa

        y = np.stack(
            [librosa.resample(y[:, c], orig_sr=file_sr, target_sr=target_sr) for c in range(y.shape[1])],
            axis=1,
        ).astype(np.float32)
    notes: list[str] = []
    out = render_gap_aware(
        y,
        target_sr,
        spans,
        target_frames,
        engine=engine,
        reaper_exe=reaper_exe,
        notes=notes,
    )
    if out.shape[1] < target_channels:
        out = np.pad(out, ((0, 0), (0, target_channels - out.shape[1])))
    elif out.shape[1] > target_channels:
        out = out[:, :target_channels]
    if len(out) < target_frames:
        out = np.pad(out, ((0, target_frames - len(out)), (0, 0)))
    elif len(out) > target_frames:
        out = out[:target_frames]
    dest = dest if hasattr(dest, "with_suffix") else dest
    tmp = dest.with_name(dest.stem + "._gap_tmp" + dest.suffix)
    if tmp.exists():
        tmp.unlink()
    if dest.suffix.lower() == ".flac":
        sf.write(str(tmp), out, target_sr, format="FLAC")
    else:
        sf.write(str(tmp), out, target_sr, subtype="FLOAT")
    tmp.replace(dest)
    return notes
