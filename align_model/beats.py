"""Beat and downbeat scores. High-margin audio matches are left alone."""

from __future__ import annotations

import numpy as np

from align_model.evidence import AlignmentEvidence, HIGH_MARGIN


def beat_period(beats: np.ndarray) -> float:
    beats = np.asarray(beats, dtype=float).reshape(-1)
    beats = beats[np.isfinite(beats)]
    if beats.size < 2:
        return 0.0
    gaps = np.diff(np.sort(beats))
    gaps = gaps[(gaps > 0.18) & (gaps < 2.0)]
    if gaps.size == 0:
        return 0.0
    return float(np.median(gaps))


def _nearest(time: float, events: np.ndarray) -> float | None:
    events = np.asarray(events, dtype=float).reshape(-1)
    events = events[np.isfinite(events)]
    if events.size == 0:
        return None
    return float(events[int(np.argmin(np.abs(events - time)))])


def local_period(beats: np.ndarray, time: float) -> float:
    """Beat spacing around ``time``. A changing tempo keeps its local gap."""
    beats = np.asarray(beats, dtype=float).reshape(-1)
    beats = beats[np.isfinite(beats)]
    if beats.size < 2:
        return 0.0
    near = np.sort(beats[np.abs(beats - float(time)) <= 8.0])
    if near.size >= 3:
        gaps = np.diff(near)
        gaps = gaps[(gaps > 0.18) & (gaps < 2.0)]
        if gaps.size:
            return float(np.median(gaps))
    return beat_period(beats)


def onset_event_times(onset: np.ndarray, hop_sec: float) -> np.ndarray:
    """Query onset times, in seconds, from an onset-strength envelope."""
    onset = np.asarray(onset, dtype=float).reshape(-1)
    if onset.size < 3 or hop_sec <= 0:
        return np.zeros(0)
    floor = max(float(np.median(onset)) * 1.8, 1e-4)
    peaks = np.flatnonzero((onset[1:-1] > onset[:-2]) & (onset[1:-1] >= onset[2:]) & (onset[1:-1] > floor)) + 1
    return peaks.astype(float) * float(hop_sec)


def beat_alignment_score(
    lag: float,
    query_events: np.ndarray,
    beats: np.ndarray,
    downbeats: np.ndarray,
) -> float:
    """1 when query events shifted by ``lag`` land on reference beats.

    Positive lag delays an early query: the reference position is ``event + lag``.
    Downbeats count more than beats. The period is the local gap, not one song median.
    """
    events = np.asarray(query_events, dtype=float).reshape(-1)
    events = events[np.isfinite(events)]
    beats = np.asarray(beats, dtype=float).reshape(-1)
    if events.size == 0 or beats.size == 0:
        return 0.0
    downs = np.asarray(downbeats, dtype=float).reshape(-1)
    scores: list[float] = []
    weights: list[float] = []
    for event in events:
        mapped = float(event) + float(lag)
        period = local_period(beats, mapped)
        if period <= 0.05:
            continue
        beat = _nearest(mapped, beats)
        if beat is None:
            continue
        score = max(0.0, 1.0 - abs(mapped - beat) / (0.35 * period))
        weight = 1.0
        down = _nearest(mapped, downs)
        if down is not None and abs(mapped - down) <= 0.5 * period:
            bar = max(period * 4.0, period)
            down_score = max(0.0, 1.0 - abs(mapped - down) / (0.35 * bar))
            score = 0.40 * score + 0.60 * down_score
            weight = 2.0
        scores.append(score)
        weights.append(weight)
    if not scores:
        return 0.0
    return float(np.average(np.asarray(scores), weights=np.asarray(weights)))


def attach_beat_scores(
    evidence: AlignmentEvidence,
    beats: np.ndarray,
    downbeats: np.ndarray,
) -> float:
    """Write beat_score on each point from its own lag. Does not replace that lag."""
    events = np.asarray(getattr(evidence, "query_onsets", []), dtype=float)
    period = beat_period(beats)
    if period <= 0 or events.size == 0:
        return 0.0
    for point in evidence.points:
        near = events[np.abs(events - float(point.time)) <= 2.0]
        chosen = near if near.size else events
        point.beat_score = beat_alignment_score(point.lag, chosen, beats, downbeats)
    return period


def safe_lag(
    time: float,
    line_lag: float,
    *,
    query_events: np.ndarray,
    beats: np.ndarray,
    downbeats: np.ndarray,
) -> tuple[float, float]:
    """Lag near the line that places a query onset on a reference beat."""
    events = np.asarray(query_events, dtype=float).reshape(-1)
    period = local_period(beats, time)
    if period <= 0.05 or events.size == 0:
        return float(line_lag), 0.0
    query_time = float(time) - float(line_lag)
    near = events[np.abs(events - query_time) <= max(0.5, period)]
    if near.size == 0:
        return float(line_lag), 0.0
    event = float(near[int(np.argmin(np.abs(near - query_time)))])
    mapped = event + float(line_lag)
    downs = np.asarray(downbeats, dtype=float).reshape(-1)
    down = _nearest(mapped, downs)
    near_downbeat = down is not None and abs(down - mapped) <= 0.5 * period
    grid = downs if near_downbeat and downs.size else beats
    beat = _nearest(mapped, grid)
    if beat is None:
        return float(line_lag), 0.0
    snapped = float(beat) - event
    limit = 0.5 * (period * 4.0 if near_downbeat else period)
    if abs(snapped - float(line_lag)) > limit:
        return float(line_lag), (2.0 if near_downbeat else 1.0)
    return float(snapped), (2.0 if near_downbeat else 1.0)


def load_beats(path) -> tuple[np.ndarray, np.ndarray]:
    """Beats from the original. An empty pair when the tracker is unavailable."""
    try:
        from beat_phase import track_file

        beats, downbeats = track_file(path)
    except Exception:
        return np.zeros(0), np.zeros(0)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def high_margin(
    margin: float,
    confidence: float,
    *,
    margin_min: float = HIGH_MARGIN,
    conf_min: float = 0.45,
) -> bool:
    return float(margin) >= float(margin_min) and float(confidence) >= float(conf_min)
