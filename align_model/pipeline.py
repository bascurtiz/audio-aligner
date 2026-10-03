"""Build the stem time map the warp engines consume."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from align_model.beats import attach_beat_scores
from align_model.decompose import decompose
from align_model.evidence import AlignmentEvidence, EvidencePoint, extract_evidence
from align_model.gaps import MapSpan, PhraseUnit, gap_confidence, phrase_units, series_from_spans
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
    )


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
) -> StemMap:
    """Feature match, robust model, constrained map. ``pad_sec`` is the robust offset."""
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    qry = np.asarray(qry, dtype=np.float32).reshape(-1)
    if profile is None:
        profile = resolve_profile(ref, sr)
    duration = len(ref) / sr if sr else 0.0
    target = duration if target_sec is None else float(target_sec)
    incoming = len(qry) / sr if sr else 0.0
    source_sec = incoming if in_sec is None else float(in_sec)
    if evidence is None:
        evidence = extract_evidence(ref, qry, sr, profile=profile, kind=kind, beats=beats, downbeats=downbeats)
    if beats is not None and len(beats) >= 4:
        attach_beat_scores(evidence, beats, np.asarray(downbeats if downbeats is not None else []))
    if len(evidence.points) < 2:
        return _empty_map(kind, profile, target, source_sec)
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
    )
    pad = float(np.clip(measured.offset_sec, -profile.max_pad_sec, profile.max_pad_sec))
    # The solver's src is a position in the padded wav. Store the original
    # stem time. Renderers add pad back when they address that wav.
    src = np.asarray(src, dtype=float) - pad
    report = apply_codes(measured, margins=evidence.margins)
    report["kind"] = kind
    return StemMap(
        times=_times,
        lags=mapped,
        scores=evidence.scores,
        src=src,
        dst=dst,
        pad_sec=pad,
        report=report,
        kind=kind,
    )


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
    lead = int(0.05 * query.size)
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
        max_lag_sec=min(profile.max_lag_sec, max(0.45, dur * 0.35)),
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
    )
    if len(stem.src) < 2:
        return None
    confidence = float(stem.report.get("confidence") or 0.0)
    if abs(float(stem.pad_sec)) > 0.12 and confidence < 0.45:
        return None
    spans: list[MapSpan] = []
    shift = float(stem.pad_sec)
    for src, dst, nxt_src, nxt_dst in zip(stem.src[:-1], stem.dst[:-1], stem.src[1:], stem.dst[1:]):
        # stem.src has the local pad removed. The phrase file is not padded,
        # so the read adds that pad back once. stem.dst is a time in the
        # reference window, and the same pad is the query-to-reference
        # offset, so the destination adds it once. A phrase that is 100 ms
        # early then reads query 9.900 and writes reference 10.000.
        src0 = float(unit.src_start) + float(src) + shift
        src1 = float(unit.src_start) + float(nxt_src) + shift
        dst0 = origin + shift + float(dst)
        dst1 = origin + shift + float(nxt_dst)
        if src1 - src0 < 0.03 or dst1 - dst0 < 0.03:
            continue
        spans.append(MapSpan("speech", src0, src1, dst0, dst1))
    return _cover_placed_edges(spans, unit) or None


def _cover_placed_edges(spans: list[MapSpan], unit: PhraseUnit) -> list[MapSpan]:
    """Keep the placed phrase where the local map has no markers.

    Evidence windows sit inside the phrase, so the solved map often stops
    before the end the detector already measured. That uncovered tail is
    still the phrase. Leaving it out turns the end of the line into the rest.
    """
    if not spans:
        return spans
    out = list(spans)
    first = out[0]
    if first.src_start > unit.src_start + 0.03 and first.dst_start > unit.dst_start + 0.03:
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
    if unit.src_end > last.src_end + 0.03 and unit.dst_end > last.dst_end + 0.03:
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


def _with_silence(pieces: list[MapSpan], duration: float) -> list[MapSpan]:
    """Speech pieces stay in order. A stall between them is silence."""
    ordered = sorted(pieces, key=lambda span: (span.dst_start, span.src_start))
    spans: list[MapSpan] = []
    prev = 0.0
    for span in ordered:
        dur = max(0.04, float(span.dst_end - span.dst_start))
        dst0 = max(float(span.dst_start), prev)
        dst1 = dst0 + dur if dst0 > span.dst_start + 1e-4 else max(float(span.dst_end), dst0 + 0.04)
        if dst0 > prev + 0.02:
            spans.append(MapSpan("silence", float(span.src_start), float(span.src_start), prev, dst0))
        spans.append(MapSpan("speech", float(span.src_start), float(span.src_end), dst0, dst1))
        prev = dst1
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
        beats = list(aca.get("beats") or []) or list(inst.get("beats") or [])
        downbeats = list(aca.get("downbeats") or []) or list(inst.get("downbeats") or [])
        primary["beats"] = beats
        primary["downbeats"] = downbeats
    primary["summary"] = notes_fragment(primary)
    return primary
