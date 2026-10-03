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


def beat_score_at(
    time: float,
    lag: float,
    beats: np.ndarray,
    downbeats: np.ndarray,
    period: float,
) -> float:
    """1 when this lag lands the window on a real beat of the reference.

    Near a downbeat, distance to that downbeat counts more than the beat.
    """
    if period <= 0.05:
        return 0.0
    query_time = float(time) - float(lag)
    beat = _nearest(query_time, beats)
    if beat is None:
        return 0.0
    err = abs(query_time - beat)
    score = max(0.0, 1.0 - err / (0.35 * period))
    downbeats = np.asarray(downbeats, dtype=float).reshape(-1)
    nearest_down = _nearest(float(time), downbeats)
    if nearest_down is not None and abs(nearest_down - float(time)) <= 0.5 * period:
        bar = max(period * 4.0, period)
        down = _nearest(query_time, downbeats)
        if down is not None:
            bar_score = max(0.0, 1.0 - abs(query_time - down) / (0.35 * bar))
            score = 0.40 * score + 0.60 * bar_score
    return float(score)


def attach_beat_scores(
    evidence: AlignmentEvidence,
    beats: np.ndarray,
    downbeats: np.ndarray,
) -> float:
    """Write beat_score on each point. Does not replace the audio lag."""
    period = beat_period(beats)
    if period <= 0:
        return 0.0
    for point in evidence.points:
        point.beat_score = beat_score_at(point.time, point.lag, beats, downbeats, period)
    return period


def safe_lag(
    time: float,
    line_lag: float,
    *,
    beats: np.ndarray,
    downbeats: np.ndarray,
    period: float,
) -> tuple[float, float]:
    """Lag that lands this time on the nearest real beat, plus a downbeat weight."""
    if period <= 0.05:
        return float(line_lag), 0.0
    query_time = float(time) - float(line_lag)
    downbeats = np.asarray(downbeats, dtype=float).reshape(-1)
    nearest_down = _nearest(float(time), downbeats)
    near_downbeat = nearest_down is not None and abs(nearest_down - float(time)) <= 0.5 * period
    events = downbeats if near_downbeat and downbeats.size else beats
    nearest = _nearest(query_time, events)
    if nearest is None:
        return float(line_lag), 0.0
    snapped = float(time) - nearest
    per = period * 4.0 if near_downbeat else period
    if abs(snapped - float(line_lag)) > 0.5 * per:
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


def high_margin(margin: float, confidence: float) -> bool:
    return float(margin) >= HIGH_MARGIN and float(confidence) >= 0.45
