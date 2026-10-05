"""Build the stem time map the warp engines consume."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from align_model.beats import attach_beat_scores
from align_model.decompose import decompose
from align_model.evidence import AlignmentEvidence, EvidencePoint, extract_evidence
from align_model.gaps import (
    MapSpan,
    PhraseUnit,
    gap_confidence,
    phrase_units,
    series_from_spans,
    trim_span_to_file,
)
from align_model.params import TrackProfile, resolve_profile
from align_model.quality import apply_codes, notes_fragment
from align_model.time_map import solve_time_map


@dataclass
class StemMap:
    """Solved map for one stem.

    ``src`` is an original-stem time. Time 0 is the first sample of the
    file before padding. ``dst`` is a reference time. ``pad_sec`` is
    pre-roll in front of source time 0 (negative trims the front). The
    renderer reads the padded wav at ``src + pad_sec``.
    """

    times: np.ndarray
    lags: np.ndarray
    scores: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    pad_sec: float
    report: dict
    kind: str
    gap_units: list[PhraseUnit] = field(default_factory=list)
    spans: list = field(default_factory=list)
    # ``pad_sec`` is what a non-gap renderer adds to ``src``. On a gap vocal
    # that value is the first phrase's placement, not the robust offset.
    # These three names keep the two meanings apart.
    global_offset_sec: float = 0.0
    render_pad_sec: float = 0.0
    phrase_start_offset: float | None = None


def _empty_map(kind: str, profile: TrackProfile, target_sec: float, in_sec: float) -> StemMap:
    report = decompose(np.zeros(0), np.zeros(0)).as_dict()
    report["profile"] = profile.name
    apply_codes(report)
    return StemMap(
        times=np.zeros(0),
        lags=np.zeros(0),
        scores=np.zeros(0),
        src=np.array([0.0, max(in_sec, 0.0)]),
        dst=np.array([0.0, max(target_sec, 0.0)]),
        pad_sec=0.0,
        report=report,
        kind=kind,
        global_offset_sec=0.0,
        render_pad_sec=0.0,
    )


def _activity_start_sec(y: np.ndarray, sr: int, frac: float = 0.02) -> float:
    """First moment the waveform leaves the noise floor."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr <= 0 or y.size < max(1, sr // 5):
        return 0.0
    win = max(1, int(0.05 * sr))
    n = len(y) // win
    if n < 1:
        return 0.0
    peak = np.max(np.abs(y[: n * win].reshape(n, win)), axis=1)
    body = float(np.percentile(peak, 95)) if peak.size else 0.0
    if body < 1e-4:
        return 0.0
    idx = np.flatnonzero(peak >= max(body * frac, 1e-4))
    if len(idx) == 0:
        return 0.0
    return float(idx[0] * win / sr)


_BODY_MIN_SEC = 60.0
_BODY_HOP_SEC = 0.25


def _rms_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(_BODY_HOP_SEC * sr))
    n = len(y) // hop
    if n < 4:
        return np.zeros(0, dtype=np.float64)
    block = np.asarray(y[: n * hop], dtype=np.float64).reshape(n, hop)
    return np.sqrt(np.mean(block * block, axis=1))


def _envelope_ncc(ref_e: np.ndarray, qry_e: np.ndarray, shift: int) -> float:
    """Positive ``shift`` delays the query so its start meets a later reference frame."""
    if shift >= 0:
        a = ref_e[shift:]
        b = qry_e[: a.size]
    else:
        b = qry_e[-shift:]
        a = ref_e[: b.size]
    n = min(int(a.size), int(b.size))
    if n < 32:
        return -1.0
    a = a[:n] - float(np.mean(a[:n]))
    b = b[:n] - float(np.mean(b[:n]))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-8:
        return -1.0
    return float(np.dot(a, b) / den)


def _body_offset(ref: np.ndarray, qry: np.ndarray, sr: int, profile: TrackProfile) -> float | None:
    """Whole-file lag when a later copy matches the body and the first onset does not.

    A repeating section makes the first energy land on the wrong copy. Local
    evidence then stays inside ``max_lag_sec`` of that onset and never sees
    the copy the rest of the file agrees with.
    """
    if sr <= 0 or min(len(ref), len(qry)) < _BODY_MIN_SEC * sr:
        return None
    ref_e = _rms_envelope(ref, sr)
    qry_e = _rms_envelope(qry, sr)
    if ref_e.size < 32 or qry_e.size < 32:
        return None
    activity = _activity_start_sec(ref, sr) - _activity_start_sec(qry, sr)
    hop = _BODY_HOP_SEC
    limit = float(profile.max_pad_sec)
    best_shift = 0
    best_score = -2.0
    for shift in range(int(-limit / hop), int(limit / hop) + 1):
        score = _envelope_ncc(ref_e, qry_e, shift)
        if score > best_score:
            best_score = score
            best_shift = shift
    best = best_shift * hop
    act_score = _envelope_ncc(ref_e, qry_e, int(round(activity / hop)))
    if best_score < 0.45 or best_score < act_score + 0.15:
        return None
    if abs(best - activity) <= float(profile.max_lag_sec) + 0.25:
        return None
    if abs(best) > limit:
        return None
    return float(best)


def _front_silence_pad(ref: np.ndarray, qry: np.ndarray, sr: int, profile: TrackProfile) -> float:
    """Offset the local search cannot see: a long lead-in, or the matching copy of a loop.

    Local evidence only looks ``max_lag_sec`` either way. A lead-in longer than
    that never becomes a pad, so the stem starts at time 0 and the spare length
    is left as silence at the end.
    """
    if sr <= 0:
        return 0.0
    coarse = _activity_start_sec(ref, sr) - _activity_start_sec(qry, sr)
    body = _body_offset(ref, qry, sr, profile)
    if body is not None:
        coarse = body
    if abs(coarse) <= float(profile.max_lag_sec) + 0.25:
        return 0.0
    if abs(coarse) > float(profile.max_pad_sec):
        return 0.0
    return float(coarse)


def _shift_audio(y: np.ndarray, sr: int, pad_sec: float) -> np.ndarray:
    """Positive pad prepends silence. Negative pad trims the front."""
    n = int(round(float(pad_sec) * sr))
    if n > 0:
        return np.concatenate([np.zeros(n, dtype=np.float32), np.asarray(y, dtype=np.float32)])
    if n < 0:
        y = np.asarray(y, dtype=np.float32)
        return y[min(-n, len(y)) :]
    return np.asarray(y, dtype=np.float32)


def _rebase_front_pad(stem: StemMap, coarse: float) -> StemMap:
    """Move a map built on a pre-shifted stem back to the original file."""
    if abs(coarse) < 1e-6:
        return stem
    stem.src = np.asarray(stem.src, dtype=float) - float(coarse)
    stem.pad_sec = float(stem.pad_sec) + float(coarse)
    stem.render_pad_sec = float(stem.render_pad_sec) + float(coarse)
    stem.global_offset_sec = float(stem.global_offset_sec) + float(coarse)
    stem.report["coarse_pad_sec"] = float(coarse)
    stem.report["render_pad_sec"] = stem.render_pad_sec
    stem.report["global_offset_sec"] = stem.global_offset_sec
    return stem


def map_stem(
    ref: np.ndarray,
    qry: np.ndarray,
    sr: int,
    *,
    kind: str,
    profile: TrackProfile | None = None,
    beats: np.ndarray | None = None,
    downbeats: np.ndarray | None = None,
    target_sec: float | None = None,
    in_sec: float | None = None,
    evidence: AlignmentEvidence | None = None,
    allow_front_pad: bool = True,
) -> StemMap:
    """Feature match, robust model, constrained map. ``pad_sec`` is the robust offset."""
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    qry = np.asarray(qry, dtype=np.float32).reshape(-1)
    if profile is None:
        profile = resolve_profile(ref, sr)
    coarse = 0.0
    if evidence is None and allow_front_pad:
        coarse = _front_silence_pad(ref, qry, sr, profile)
    if abs(coarse) >= 1e-4:
        qry = _shift_audio(qry, sr, coarse)
    duration = len(ref) / sr if sr else 0.0
    target = duration if target_sec is None else float(target_sec)
    incoming = len(qry) / sr if sr else 0.0
    source_sec = incoming if in_sec is None else float(in_sec) + coarse
    if evidence is None:
        evidence = extract_evidence(ref, qry, sr, profile=profile, kind=kind, beats=beats, downbeats=downbeats)
    if beats is not None and len(beats) >= 4:
        attach_beat_scores(evidence, beats, np.asarray(downbeats if downbeats is not None else []))
    if len(evidence.points) < 2:
        return _rebase_front_pad(_empty_map(kind, profile, target, source_sec), coarse)
    measured = decompose(
        evidence.times,
        evidence.lags,
        evidence.scores,
        weak_score=profile.weak_score,
        fit_tol_sec=profile.fit_tol_sec,
    )
    if evidence.low_signal_frac > measured.weak_frac:
        measured.weak_frac = float(evidence.low_signal_frac)
    _times, mapped, src, dst = solve_time_map(
        evidence,
        measured,
        profile,
        beats=beats,
        downbeats=downbeats,
        target_sec=target,
        in_sec=source_sec,
        envelope_ref=ref if kind == "instrumental" else None,
        envelope_qry=qry if kind == "instrumental" else None,
        envelope_sr=sr if kind == "instrumental" else 0,
    )
    measured_pad = float(measured.offset_sec)
    # Clipping the pad and still subtracting it would leave src short of a
    # real source time by the amount that was clipped. An offset past the
    # pad limit is a bad match, so the warp falls back to an identity map.
    if abs(measured_pad) > float(profile.max_pad_sec):
        report = apply_codes(measured, margins=evidence.margins)
        report["kind"] = kind
        report["pad_rejected"] = True
        report["map_rejected"] = "pad_limit"
        report["identity_fallback"] = True
        report["map_type"] = "identity_fallback"
        report["measured_pad_sec"] = measured_pad
        report["global_offset_sec"] = measured_pad
        report["render_pad_sec"] = 0.0
        rejected = StemMap(
            times=np.asarray(_times, dtype=float),
            lags=np.asarray(mapped, dtype=float),
            scores=np.asarray(evidence.scores, dtype=float),
            src=np.array([0.0, max(source_sec, 0.0)]),
            dst=np.array([0.0, max(target, 0.0)]),
            pad_sec=0.0,
            report=report,
            kind=kind,
            global_offset_sec=measured_pad,
            render_pad_sec=0.0,
        )
        return _rebase_front_pad(rejected, coarse)
    pad = measured_pad
    # The solver's src is a position in the padded wav. Store the original
    # stem time. Renderers add pad back when they address that wav.
    src = np.asarray(src, dtype=float) - pad
    report = apply_codes(measured, margins=evidence.margins)
    report["kind"] = kind
    report["map_type"] = "time_map"
    report["global_offset_sec"] = pad
    report["render_pad_sec"] = pad
    fitted = StemMap(
        times=_times,
        lags=mapped,
        scores=evidence.scores,
        src=src,
        dst=dst,
        pad_sec=pad,
        report=report,
        kind=kind,
        global_offset_sec=pad,
        render_pad_sec=pad,
    )
    return _rebase_front_pad(fitted, coarse)


def map_from_points(
    points: list[EvidencePoint],
    *,
    kind: str = "vocal",
    profile: TrackProfile | None = None,
    beats: np.ndarray | None = None,
    downbeats: np.ndarray | None = None,
    target_sec: float = 10.0,
    in_sec: float = 10.0,
) -> StemMap:
    """Time map from evidence that was built without reading audio."""
    if profile is None:
        profile = resolve_profile(None, 0, name="default")
    evidence = AlignmentEvidence(points=list(points), profile=profile.name, kind=kind)
    return map_stem(
        np.zeros(8, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        1,
        kind=kind,
        profile=profile,
        beats=beats,
        downbeats=downbeats,
        target_sec=target_sec,
        in_sec=in_sec,
        evidence=evidence,
    )


def _model_phrase(
    unit: PhraseUnit,
    aca: np.ndarray,
    ref: np.ndarray,
    sr: int,
    profile: TrackProfile,
    beats: np.ndarray | None,
    downbeats: np.ndarray | None,
) -> list[MapSpan] | None:
    """Run the shared evidence map inside one placed phrase.

    A weak or wandering fit is discarded so the phrase placement keeps the rest.
    """
    dur = float(unit.src_end - unit.src_start)
    if dur < 0.9 or sr <= 0:
        return None
    q0 = int(max(0.0, unit.src_start) * sr)
    q1 = int(min(len(aca), unit.src_end * sr))
    query = np.asarray(aca[q0:q1], dtype=np.float32)
    if query.size < int(0.8 * sr):
        return None
    d0 = int(max(0.0, unit.dst_start) * sr)
    local_lag = min(float(profile.max_lag_sec), max(0.45, dur * 0.35))
    # Five percent of a long phrase is a couple of seconds. That lookbehind
    # sits past the lag search, so the placed entrance is invisible and the
    # map pins the attack onto whatever vocal Demucs still has before it.
    lead_sec = min(0.05 * dur, 0.25, max(0.0, local_lag - 0.05))
    lead = int(round(lead_sec * sr))
    extra = int(0.30 * query.size)
    r0 = max(0, d0 - lead)
    r1 = min(len(ref), d0 + query.size + extra)
    reference = np.asarray(ref[r0:r1], dtype=np.float32)
    if reference.size < int(0.8 * sr):
        return None
    local = replace(
        profile,
        win_sec=min(profile.win_sec, max(1.2, dur / 2.0)),
        step_sec=min(profile.step_sec, max(0.3, min(profile.win_sec, max(1.2, dur / 2.0)) / 3.0)),
        max_lag_sec=local_lag,
    )
    origin = r0 / float(sr)
    end = origin + len(reference) / float(sr)
    beat_arr = np.asarray([] if beats is None else beats, dtype=float)
    down_arr = np.asarray([] if downbeats is None else downbeats, dtype=float)
    beat_arr = beat_arr[(beat_arr >= origin) & (beat_arr <= end)] - origin
    down_arr = down_arr[(down_arr >= origin) & (down_arr <= end)] - origin
    stem = map_stem(
        reference,
        query,
        sr,
        kind="vocal",
        profile=local,
        beats=beat_arr if beat_arr.size >= 4 else None,
        downbeats=down_arr if down_arr.size else None,
        target_sec=len(reference) / float(sr),
        in_sec=len(query) / float(sr),
        allow_front_pad=False,
    )
    if len(stem.src) < 2:
        return None
    # No evidence inside the phrase. The empty map then stretches the whole
    # source across the lookahead window, so a one-second hit starts early
    # and runs long. The placement already has the right rate.
    if stem.report.get("map_type") != "time_map" or len(stem.times) < 2:
        return None
    confidence = float(stem.report.get("confidence") or 0.0)
    if abs(float(stem.pad_sec)) > 0.12 and confidence < 0.45:
        return None
    # A multi-second pad or source far before the phrase pastes earlier
    # acapella into a Demucs rest or a different part. A modest lookbehind
    # refine (under ~lead_sec) is fine; keep the detector placement otherwise.
    if abs(float(stem.pad_sec)) > 0.50:
        return None
    if float(np.min(np.asarray(stem.src, dtype=float))) < -0.35:
        return None
    spans: list[MapSpan] = []
    for src, dst, nxt_src, nxt_dst in zip(stem.src[:-1], stem.dst[:-1], stem.src[1:], stem.dst[1:]):
        # stem.src is already the unpadded query-slice time. stem.dst is
        # already the reference-window time. The pad is a source-file
        # pre-roll and was removed when src was stored. Adding it to the
        # source reads the wrong sample. Adding it to the destination
        # moves the phrase in the reference. Neither side gets it again.
        src0 = float(unit.src_start) + float(src)
        src1 = float(unit.src_start) + float(nxt_src)
        dst0 = origin + float(dst)
        dst1 = origin + float(nxt_dst)
        if src1 - src0 < 0.03 or dst1 - dst0 < 0.03:
            continue
        if src0 < float(unit.src_start) - 0.35:
            return None
        trimmed = trim_span_to_file(
            MapSpan("speech", src0, src1, dst0, dst1),
            len(aca) / float(sr),
        )
        if trimmed is not None:
            spans.append(trimmed)
    return _hold_placed_entrance(spans, unit, map_confidence=confidence)


def _cover_placed_edges(spans: list[MapSpan], unit: PhraseUnit) -> list[MapSpan]:
    """Fill an edge the local map did not reach.

    The detector placement owns either end. A gap on the source side or on
    the destination side is enough. Both axes do not have to be late.
    """
    if not spans:
        return spans
    out = list(spans)
    first = out[0]
    src_gap = float(first.src_start - unit.src_start)
    dst_gap = float(first.dst_start - unit.dst_start)
    if (src_gap > 0.03 or dst_gap > 0.03) and src_gap > -1e-3 and dst_gap > -1e-3:
        out.insert(
            0,
            MapSpan(
                "speech",
                float(unit.src_start),
                float(first.src_start),
                float(unit.dst_start),
                float(first.dst_start),
            ),
        )
    last = out[-1]
    src_tail = float(unit.src_end - last.src_end)
    dst_tail = float(unit.dst_end - last.dst_end)
    if (src_tail > 0.03 or dst_tail > 0.03) and src_tail > -1e-3 and dst_tail > -1e-3:
        out.append(
            MapSpan(
                "speech",
                float(last.src_end),
                float(unit.src_end),
                float(last.dst_end),
                float(unit.dst_end),
            )
        )
    return out


def _dst_at_src(spans: list[MapSpan], src: float) -> float | None:
    """Where ``src`` lands. A source before the first span uses that span's start."""
    if not spans:
        return None
    src = float(src)
    first = spans[0]
    if src < float(first.src_start) - 1e-3:
        return float(first.dst_start)
    for span in spans:
        width = float(span.src_end - span.src_start)
        if width <= 1e-6:
            continue
        if float(span.src_start) - 1e-3 <= src <= float(span.src_end) + 1e-3:
            share = (src - float(span.src_start)) / width
            return float(span.dst_start) + share * float(span.dst_end - span.dst_start)
    return None


def _hold_placed_entrance(
    spans: list[MapSpan],
    unit: PhraseUnit,
    *,
    map_confidence: float | None = None,
) -> list[MapSpan] | None:
    """The detector owns the entrance when the placement itself is a real match.

    A local map may refine the phrase. It may not open on Demucs audio from
    before a confident placement. That audio is the previous phrase, or a
    harmony the lead acapella does not contain.

    A weak placement can sit a fraction of a second late. The waveform is
    then obvious and the interior map is the correction, so a confident map
    that is only modestly earlier is kept.
    """
    if not spans:
        return None
    attack = _dst_at_src(spans, float(unit.src_start))
    if attack is not None and attack < float(unit.dst_start) - 0.12:
        early = float(unit.dst_start) - attack
        placed = float(unit.confidence)
        modeled = 1.0 if map_confidence is None else float(map_confidence)
        modest = early <= 0.40
        if not (modest and placed < 0.45 and modeled >= 0.55):
            return None
    return _cover_placed_edges(spans, unit) or None


def _clip_dst_end(span: MapSpan, new_end: float) -> MapSpan | None:
    """Shorten a speech span so it ends where the next one starts.

    The source end moves in proportion, so the kept prefix keeps its timing.
    A span that would be left with nothing is dropped.
    """
    if new_end <= span.dst_start + 1e-3:
        return None
    solved = float(span.dst_end - span.dst_start)
    if solved <= 1e-6:
        return None
    kept = (new_end - float(span.dst_start)) / solved
    src_end = float(span.src_start) + kept * (float(span.src_end) - float(span.src_start))
    return MapSpan("speech", float(span.src_start), src_end, float(span.dst_start), float(new_end))


def _with_silence(pieces: list[MapSpan], duration: float) -> list[MapSpan]:
    """Speech pieces stay in order. A stall between them is silence.

    An overlap clips the earlier span's end to the next span's start.
    The later span keeps the destination the solver gave it. A zero-length
    span gets a 1 ms stand-in. A few milliseconds of overlap is the same
    clip, so float noise does not move a phrase.
    """
    ordered = sorted(pieces, key=lambda span: (span.dst_start, span.src_start))
    speech: list[MapSpan] = []
    for span in ordered:
        src0 = float(span.src_start)
        src1 = float(span.src_end)
        start = float(span.dst_start)
        end = float(span.dst_end)
        if end - start <= 1e-4:
            end = start + 1e-3
        if speech:
            prev = speech[-1]
            overlap = float(prev.dst_end) - start
            if overlap > 0.0:
                clipped = _clip_dst_end(prev, start)
                if clipped is None:
                    speech.pop()
                else:
                    speech[-1] = clipped
        speech.append(MapSpan("speech", src0, src1, start, end))
    spans: list[MapSpan] = []
    prev = 0.0
    for span in speech:
        if span.dst_start > prev + 0.02:
            spans.append(MapSpan("silence", float(span.src_start), float(span.src_start), prev, span.dst_start))
        spans.append(span)
        prev = float(span.dst_end)
    duration = max(float(duration), prev)
    if duration > prev + 0.05 and ordered:
        tail = float(ordered[-1].src_end)
        spans.append(MapSpan("silence", tail, tail, prev, duration))
    return spans


def map_vocal_gaps(
    aca: np.ndarray,
    ref: np.ndarray,
    sr: int,
    *,
    profile: TrackProfile,
    target_sec: float | None = None,
    beats: np.ndarray | None = None,
    downbeats: np.ndarray | None = None,
) -> StemMap:
    """Phrase placement, then the shared time map inside each phrase.

    Rests stay digital silence. A phrase the model cannot read keeps its placement.
    """
    aca = np.asarray(aca, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    duration = len(ref) / sr if target_sec is None else float(target_sec)
    units, notes = phrase_units(aca, ref, sr)
    if not units:
        stem = map_stem(
            ref,
            aca,
            sr,
            kind="vocal",
            profile=profile,
            beats=beats,
            downbeats=downbeats,
            target_sec=duration,
            in_sec=len(aca) / sr,
        )
        stem.report["gap_notes"] = notes
        stem.report["global_pad_sec"] = float(stem.pad_sec)
        stem.report["phrase_start_offset"] = None
        return stem
    pieces: list[MapSpan] = []
    used_model = False
    for unit in units:
        modeled = _model_phrase(unit, aca, ref, sr, profile, beats, downbeats)
        if modeled:
            pieces.extend(modeled)
            used_model = True
        else:
            pieces.append(
                MapSpan("speech", unit.src_start, unit.src_end, unit.dst_start, unit.dst_end)
            )
    if used_model:
        notes.append("phrase_model=1")
    spans = _with_silence(pieces, duration)
    times, lags, gaps = series_from_spans(spans)
    scores = np.zeros(len(times), dtype=float)
    cursor = 0
    for unit in units:
        while cursor < len(times) and times[cursor] <= unit.dst_end + 1e-6:
            if times[cursor] >= unit.dst_start - 1e-6:
                scores[cursor] = unit.confidence
            cursor += 1
    measured = decompose(times, lags, scores, weak_score=profile.weak_score, fit_tol_sec=profile.fit_tol_sec)
    measured.profile = profile.name
    measured.gap_inserts = gaps
    measured.confidence = gap_confidence(units)
    measured.times = [float(v) for v in times]
    measured.lags = [float(v) for v in lags]
    measured.confidence_series = [float(v) for v in scores]
    report = apply_codes(measured, margins=np.array([unit.match_margin for unit in units]), gap_units=units)
    report["gap_notes"] = notes
    report["kind"] = "vocal"
    report["profile"] = profile.name
    # Where phrase 1 was placed, and the robust offset of the speech lag
    # series. The first is a detector result. The second is a fit across
    # every phrase, so a bad first phrase must not be treated as the
    # render pad. The renderer still uses the phrase offset until a
    # benchmark shows that choice moves the vocal.
    phrase_start_offset = float(units[0].start_offset)
    global_pad_sec = float(measured.offset_sec)
    report["phrase_start_offset"] = phrase_start_offset
    report["global_pad_sec"] = global_pad_sec
    report["global_offset_sec"] = global_pad_sec
    report["global_drift"] = float(measured.drift)
    report["render_pad_sec"] = phrase_start_offset
    report["map_type"] = "gap_spans"
    # The phrase renderer consumes spans directly. src/dst here describe speech pins.
    src = []
    dst = []
    for span in spans:
        if span.kind != "speech":
            continue
        src.extend([span.src_start, span.src_end])
        dst.extend([span.dst_start, span.dst_end])
    if len(dst) < 2:
        src = [0.0, len(aca) / sr]
        dst = [0.0, duration]
    report["markers"] = [float(v) for v in dst]
    return StemMap(
        times=times,
        lags=lags,
        scores=scores,
        src=np.asarray(src, dtype=float),
        dst=np.asarray(dst, dtype=float),
        pad_sec=phrase_start_offset,
        report=report,
        kind="vocal",
        gap_units=units,
        spans=spans,
        global_offset_sec=global_pad_sec,
        render_pad_sec=phrase_start_offset,
        phrase_start_offset=phrase_start_offset,
    )


def combine_reports(aca: dict | None, inst: dict | None, *, profile: str) -> dict:
    """One row payload. The drawn curve is the vocal map when it exists."""
    primary = dict(aca or inst or {})
    primary.setdefault("failures", [])
    primary["profile"] = profile
    if aca:
        primary["vocal_confidence"] = float(aca.get("confidence") or 0.0)
        primary["aca_curve"] = {"times": list(aca.get("times") or []), "lags": list(aca.get("lags") or [])}
        primary["times"] = list(aca.get("times") or [])
        primary["lags"] = list(aca.get("lags") or [])
        primary["gap_inserts"] = list(aca.get("gap_inserts") or [])
        primary["low_conf_spans"] = list(aca.get("low_conf_spans") or [])
        primary["markers"] = list(aca.get("markers") or [])
        primary["jumps"] = list(aca.get("jumps") or [])
    if inst:
        primary["inst_confidence"] = float(inst.get("confidence") or 0.0)
        primary["inst_curve"] = {"times": list(inst.get("times") or []), "lags": list(inst.get("lags") or [])}
        primary["inst_offset_sec"] = float(inst.get("offset_sec") or 0.0)
        primary["inst_drift"] = float(inst.get("drift") or 0.0)
        if not aca:
            primary["times"] = list(inst.get("times") or [])
            primary["lags"] = list(inst.get("lags") or [])
            primary["markers"] = list(inst.get("markers") or [])
    if aca and inst:
        primary["confidence"] = min(float(aca.get("confidence") or 0.0), float(inst.get("confidence") or 0.0))
        primary["n_discontinuities"] = int(aca.get("n_discontinuities") or 0) + int(inst.get("n_discontinuities") or 0)
        primary["max_local_error_sec"] = max(
            float(aca.get("max_local_error_sec") or 0.0),
            float(inst.get("max_local_error_sec") or 0.0),
        )
        primary["p95_error_sec"] = max(
            float(aca.get("p95_error_sec") or 0.0),
            float(inst.get("p95_error_sec") or 0.0),
        )
        primary["weak_frac"] = max(float(aca.get("weak_frac") or 0.0), float(inst.get("weak_frac") or 0.0))
        primary["failures"] = list(aca.get("failures") or []) + list(inst.get("failures") or [])
        primary["offset_sec"] = float(aca.get("offset_sec") or 0.0)
        primary["drift"] = float(aca.get("drift") or 0.0)
        primary["drift_ms_per_min"] = float(aca.get("drift_ms_per_min") or 0.0)
        primary["line_offset"] = float(aca.get("line_offset") or primary["offset_sec"])
        primary["line_slope"] = float(aca.get("line_slope") or primary["drift"])
    if aca and aca.get("map_type") == "gap_spans":
        # The drawn offset is the speech-series fit. The vocal renderer
        # follows the phrase spans, not that one curve.
        primary["map_type"] = "gap_spans"
        primary["global_offset_sec"] = float(aca.get("global_offset_sec") or 0.0)
        primary["global_drift"] = float(aca.get("global_drift") or 0.0)
        primary["render_pad_sec"] = float(aca.get("render_pad_sec") or 0.0)
        primary["phrase_start_offset"] = aca.get("phrase_start_offset")
        beats = list(aca.get("beats") or []) or list(inst.get("beats") or [])
        downbeats = list(aca.get("downbeats") or []) or list(inst.get("downbeats") or [])
        primary["beats"] = beats
        primary["downbeats"] = downbeats
    primary["summary"] = notes_fragment(primary)
    return primary
