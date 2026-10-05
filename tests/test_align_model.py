"""Math and synthetic-audio checks for the alignment model."""

from __future__ import annotations

import unittest

import numpy as np

from aca_gap_align import _keep_later_phrase, place_segments, snap_tiny_inserts, waveform_ncc_offset
from align_model.benchmark import (
    SR,
    case_continuous_vocal,
    harmonic,
    case_drift,
    case_gaps,
    case_noise,
    case_nonlinear,
    case_offset,
    case_perfect,
    case_repeated,
    case_sparse,
    case_stripped_vocal,
    case_tempo_change,
    discontinuity_counts,
    measure_curve,
)
from align_model.beats import beat_alignment_score
from align_model.evidence import (
    AlignmentEvidence,
    EvidencePoint,
    context_weights,
    extract_evidence,
    fuse_candidates,
)
from align_model.gaps import MapSpan, PhraseUnit, render_gap_aware, spans_from_units, trim_span_to_file, _needs_elastique
from align_model.params import resolve_profile
from align_model.pipeline import (
    _cover_placed_edges,
    _dst_at_src,
    _front_silence_pad,
    _hold_placed_entrance,
    _model_phrase,
    _with_silence,
    combine_reports,
    map_from_points,
    map_stem,
)
from align_model.quality import apply_codes, is_compensating, stamp_result
from align_model.time_map import (
    _clamp_accel,
    adaptive_marker_times,
    fit_reference_gaps,
    pull_unmatched_phrase,
    refine_lags_with_envelope,
    solve_constrained_lag_path,
    source_on_padded_wav,
    src_dst_from_lags,
)


def _one_speech_run(speech: list[tuple[float, float]]) -> bool:
    """Speech spans that meet, with no restored rest between them."""
    if not speech:
        return False
    return all(nxt[0] - prev[1] <= 0.05 for prev, nxt in zip(speech, speech[1:]))


class DecomposeTests(unittest.TestCase):
    def test_offset_and_drift(self) -> None:
        times = np.arange(0.5, 12.0, 0.5)
        lags = 0.25 + 0.002 * times
        report = measure_curve(times, lags, np.full(len(times), 0.8))
        self.assertAlmostEqual(report["offset_sec"], 0.25, delta=0.02)
        self.assertAlmostEqual(report["drift"], 0.002, delta=0.0008)
        self.assertGreater(report["drift_ms_per_min"], 30.0)
        self.assertLess(report["max_local_error_sec"], 0.02)

    def test_jump_is_removed_from_residual(self) -> None:
        times = np.arange(0.0, 10.0, 0.5)
        lags = np.zeros_like(times)
        lags[times >= 5.0] = 0.22
        report = measure_curve(times, lags, np.full(len(times), 0.7))
        self.assertGreaterEqual(report["n_discontinuities"], 1)
        false_pos, missed = discontinuity_counts(
            [jump["time"] for jump in report["jumps"]],
            [5.0],
            tol=1.0,
        )
        self.assertEqual(missed, 0)
        self.assertLess(report["median_error_sec"], 0.08)


class EvidenceTests(unittest.TestCase):
    def test_margin_and_vocal_weights(self) -> None:
        weights = context_weights(
            {"chroma": 0.3, "onset": 0.3, "spectral": 0.2, "waveform": 0.2},
            chroma=0.6,
            onset=0.1,
            kind="vocal",
        )
        self.assertGreater(weights["chroma"], weights["onset"])
        self.assertAlmostEqual(sum(weights.values()), 1.0, delta=1e-6)
        fused = fuse_candidates(
            [
                ("chroma", 0.10, 0.80, 0.0, 0.20),
                ("waveform", 0.11, 0.70, 0.0, 0.15),
                ("onset", 0.90, 0.30, 0.0, 0.10),
            ],
            weights,
        )
        self.assertIsNotNone(fused)
        lag, confidence, second, margin = fused
        self.assertAlmostEqual(lag, 0.105, delta=0.03)
        self.assertGreater(margin, 0.05)
        self.assertGreater(confidence, second)
        ambiguous = fuse_candidates(
            [
                ("chroma", 0.10, 0.80, 0.40, 0.72),
                ("waveform", 0.11, 0.70, 0.0, 0.10),
            ],
            weights,
        )
        clear = fuse_candidates(
            [
                ("chroma", 0.10, 0.80, 0.0, 0.05),
                ("waveform", 0.11, 0.70, 0.0, 0.05),
            ],
            weights,
        )
        self.assertLess(ambiguous[3], clear[3])
        beats = np.array([98.0, 98.5, 99.0, 99.5, 100.0, 100.5])
        event = np.array([99.94])
        landed = beat_alignment_score(0.06, event, beats, np.zeros(0))
        off_beat = beat_alignment_score(0.25, event, beats, np.zeros(0))
        self.assertGreater(landed, 0.85)
        self.assertGreater(landed, off_beat)

    def test_query_longer_than_the_reference_still_scores(self) -> None:
        sr = 22050
        rng = np.random.default_rng(0)
        ref = (rng.standard_normal(sr * 12) * 0.05).astype(np.float32)
        qry = (rng.standard_normal(sr * 20) * 0.05).astype(np.float32)
        burst = np.hanning(sr).astype(np.float32)
        ref[sr : 2 * sr] += burst
        qry[sr : 2 * sr] += burst
        profile = resolve_profile(ref, sr)
        evidence = extract_evidence(ref, qry, sr, profile=profile, kind="instrumental")
        self.assertGreater(len(evidence.points), 0)


class TimeMapTests(unittest.TestCase):
    def _points(self, lag_at) -> list[EvidencePoint]:
        points = []
        for t in np.arange(0.5, 8.0, 0.5):
            lag, conf, margin = lag_at(float(t))
            points.append(
                EvidencePoint(
                    time=float(t),
                    lag=lag,
                    confidence=conf,
                    best_score=conf,
                    second_score=max(0.0, conf - margin),
                    match_margin=margin,
                )
            )
        return points

    def test_map_is_monotonic(self) -> None:
        stem = map_from_points(self._points(lambda t: (0.02 * t, 0.7, 0.3)), target_sec=8.0, in_sec=8.0)
        self.assertGreaterEqual(len(stem.src), 2)
        self.assertTrue(np.all(np.diff(stem.src) >= -1e-3))
        self.assertTrue(np.all(np.diff(stem.dst) >= -1e-3))

    def test_high_margin_keeps_audio_lag(self) -> None:
        beats = np.arange(0.0, 8.0, 0.5)
        downbeats = np.arange(0.0, 8.0, 2.0)

        def lag_at(t: float):
            if abs(t - 4.0) < 0.2:
                return 0.18, 0.85, 0.45
            return 0.0, 0.75, 0.35

        stem = map_from_points(
            self._points(lag_at),
            beats=beats,
            downbeats=downbeats,
            target_sec=8.0,
            in_sec=8.0,
        )
        near = np.argmin(np.abs(stem.times - 4.0))
        self.assertGreater(abs(float(stem.lags[near])), 0.08)

    def test_beat_snap_does_not_walk_off_a_steady_offset(self) -> None:
        """A half-beat snap must not make a constant instrumental early, then late."""
        points = []
        for t in np.arange(1.0, 17.0, 1.0):
            margin = 0.2 if 8.0 <= t <= 12.0 else 0.0
            points.append(
                EvidencePoint(
                    time=float(t),
                    lag=-0.33,
                    confidence=0.7,
                    best_score=0.7,
                    second_score=0.5 if margin else 0.7,
                    match_margin=margin,
                )
            )
        evidence = AlignmentEvidence(points=points, profile="default", kind="instrumental")
        evidence.query_onsets = [float(t) + 0.33 + 0.18 for t in np.arange(1.0, 17.0, 1.0)]
        stem = map_stem(
            np.zeros(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            1,
            kind="instrumental",
            beats=np.arange(0.0, 20.0, 0.47),
            downbeats=np.arange(0.0, 20.0, 1.88),
            target_sec=17.0,
            in_sec=17.0,
            evidence=evidence,
        )
        for t, lag in zip(stem.times, stem.lags):
            self.assertLess(abs(float(lag) - (-0.33)), 0.04, f"t={t:.2f} lag={lag:+.3f}")

    def test_ambiguous_point_follows_the_line(self) -> None:
        beats = np.arange(0.0, 8.0, 0.5)

        def lag_at(t: float):
            if abs(t - 4.0) < 0.2:
                return 0.18, 0.2, 0.01
            return 0.0, 0.8, 0.4

        stem = map_from_points(
            self._points(lag_at),
            beats=beats,
            downbeats=beats[::4],
            target_sec=8.0,
            in_sec=8.0,
        )
        near = int(np.argmin(np.abs(stem.times - 4.0)))
        self.assertLess(abs(float(stem.lags[near])), 0.08)

    def test_curved_lag_gets_more_markers(self) -> None:
        times = np.arange(0.0, 30.0, 0.5)
        straight = adaptive_marker_times(times, 0.001 * times, min_spacing=2.0, max_spacing=30.0)
        curved = adaptive_marker_times(
            times,
            0.08 * np.sin(2 * np.pi * times / 4.0),
            min_spacing=2.0,
            max_spacing=30.0,
            curvature_gain=1.2,
        )
        self.assertGreater(len(curved), len(straight))
        mild = 0.0005 * times ** 2
        full = adaptive_marker_times(times, mild, min_spacing=2.0, max_spacing=30.0)
        weak = adaptive_marker_times(times, mild, min_spacing=2.0, max_spacing=30.0, scores=np.full(len(times), 0.1))
        self.assertLess(len(weak), len(full))

    def test_jump_can_leave_a_confident_point(self) -> None:
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        target = np.array([0.0, 0.0, 0.08, 0.08, 0.08])
        solved = solve_constrained_lag_path(
            times,
            target,
            np.full(5, 0.8),
            np.array([0.40, 0.40, 0.05, 0.40, 0.40]),
            0.0,
            0.0,
            high_conf=0.45,
            high_margin_min=0.12,
        )
        self.assertGreater(float(solved[2]), 0.06)

    def test_pad_is_preroll_before_source_zero(self) -> None:
        """A source event at 9.900 s with lag +0.100 s lands at reference 10.000 s."""
        points = [
            EvidencePoint(
                time=float(t),
                lag=0.10,
                confidence=0.85,
                best_score=0.85,
                second_score=0.1,
                match_margin=0.4,
            )
            for t in np.arange(0.5, 12.0, 0.5)
        ]
        stem = map_from_points(points, target_sec=12.0, in_sec=12.0)
        self.assertAlmostEqual(stem.pad_sec, 0.10, delta=0.03)
        src_at = float(np.interp(10.0, stem.dst, stem.src))
        self.assertAlmostEqual(src_at, 9.90, delta=0.04)
        file_at = float(source_on_padded_wav(np.array([src_at]), stem.pad_sec)[0])
        self.assertAlmostEqual(file_at, 10.0, delta=0.04)

        sr = 1000
        stem_audio = np.zeros(12 * sr, dtype=np.float32)
        stem_audio[int(round(9.9 * sr))] = 1.0
        pad_n = int(round(stem.pad_sec * sr))
        padded = np.concatenate([np.zeros(pad_n, dtype=np.float32), stem_audio])
        self.assertEqual(float(padded[int(round(file_at * sr))]), 1.0)

        late = [
            EvidencePoint(
                time=float(t),
                lag=-0.10,
                confidence=0.85,
                best_score=0.85,
                second_score=0.1,
                match_margin=0.4,
            )
            for t in np.arange(0.5, 12.0, 0.5)
        ]
        trimmed = map_from_points(late, target_sec=12.0, in_sec=12.0)
        self.assertAlmostEqual(trimmed.pad_sec, -0.10, delta=0.03)
        src_late = float(np.interp(10.0, trimmed.dst, trimmed.src))
        self.assertAlmostEqual(src_late, 10.10, delta=0.04)
        file_late = float(source_on_padded_wav(np.array([src_late]), trimmed.pad_sec)[0])
        self.assertAlmostEqual(file_late, 10.0, delta=0.04)

    def test_offset_past_max_pad_is_rejected(self) -> None:
        """An 800 ms offset with a 500 ms pad limit does not rewrite source time."""
        profile = resolve_profile(None, 0, name="default", max_pad_sec=0.50)
        late = map_from_points(self._points(lambda t: (0.80, 0.9, 0.4)), profile=profile, target_sec=8.0, in_sec=8.0)
        self.assertTrue(late.report.get("pad_rejected"))
        self.assertEqual(late.report.get("map_rejected"), "pad_limit")
        self.assertTrue(late.report.get("identity_fallback"))
        self.assertEqual(late.report.get("map_type"), "identity_fallback")
        self.assertAlmostEqual(late.pad_sec, 0.0)
        self.assertAlmostEqual(late.render_pad_sec, 0.0)
        self.assertGreater(abs(float(late.report["measured_pad_sec"])), 0.50)
        self.assertAlmostEqual(float(late.src[0]), 0.0)
        self.assertAlmostEqual(float(late.src[-1]), 8.0)
        kept = map_from_points(self._points(lambda t: (0.20, 0.9, 0.4)), profile=profile, target_sec=8.0, in_sec=8.0)
        self.assertFalse(kept.report.get("pad_rejected"))
        self.assertAlmostEqual(kept.pad_sec, 0.20, delta=0.03)
        src_at = float(np.interp(4.0, kept.dst, kept.src))
        self.assertAlmostEqual(src_at, 3.80, delta=0.04)

    def test_front_silence_is_a_pad_the_local_search_cannot_see(self) -> None:
        """A lead-in longer than max_lag delays the stem instead of leaving the gap at the end."""
        sr = 22050
        tone = np.sin(2 * np.pi * 220.0 * np.arange(3 * sr) / sr).astype(np.float32) * 0.5
        ref = np.concatenate([np.zeros(2 * sr, dtype=np.float32), tone])
        profile = resolve_profile(None, 0, name="default", max_lag_sec=0.40, max_pad_sec=10.0)
        stem = map_stem(
            ref,
            tone,
            sr,
            kind="instrumental",
            profile=profile,
            target_sec=len(ref) / sr,
            in_sec=len(tone) / sr,
        )
        self.assertGreater(stem.pad_sec, 1.6)
        self.assertLess(stem.pad_sec, 2.4)
        where = float(np.interp(0.0, stem.src, stem.dst))
        self.assertAlmostEqual(where, 2.0, delta=0.35)

    def test_repeating_onset_yields_to_the_matching_body(self) -> None:
        """A loud decoy shares the first onset. The pad follows the later copy that matches."""
        sr = 22050
        pattern = (0.12, 0.40, 0.12, 0.18, 0.55, 0.12, 0.30)

        def bursts(freq: float) -> np.ndarray:
            pieces = []
            for dur in pattern:
                t = np.arange(int(dur * sr)) / sr
                pieces.append((np.sin(2 * np.pi * freq * t) * 0.6).astype(np.float32))
                pieces.append(np.zeros(int(0.35 * sr), dtype=np.float32))
            return np.concatenate(pieces)

        body = bursts(220.0)
        t = np.arange(12 * sr) / sr
        decoy = (np.sin(2 * np.pi * 330.0 * t) * 0.6).astype(np.float32)
        ref = np.concatenate([np.zeros(2 * sr, dtype=np.float32), body, np.zeros(56 * sr, dtype=np.float32)])
        qry = np.concatenate([decoy, body, np.zeros(48 * sr, dtype=np.float32)])
        profile = resolve_profile(None, 0, name="default", max_lag_sec=1.5, max_pad_sec=30.0)
        pad = _front_silence_pad(ref, qry, sr, profile)
        self.assertAlmostEqual(pad, -10.0, delta=0.5)

    def test_renderer_pad_round_trip(self) -> None:
        """Stored source time 10.000 s plus the pad is the padded-file position."""
        stored = np.array([10.0])
        self.assertAlmostEqual(float(source_on_padded_wav(stored, 0.200)[0]), 10.200)
        self.assertAlmostEqual(float(source_on_padded_wav(stored, -0.200)[0]), 9.800)
        sr = 1000
        stem = np.zeros(12 * sr, dtype=np.float32)
        stem[10 * sr] = 1.0
        padded = np.concatenate([np.zeros(int(0.200 * sr), dtype=np.float32), stem])
        file_at = int(round(float(source_on_padded_wav(stored, 0.200)[0]) * sr))
        self.assertEqual(float(padded[file_at]), 1.0)
        trimmed = stem[int(0.200 * sr) :]
        file_trim = int(round(float(source_on_padded_wav(stored, -0.200)[0]) * sr))
        self.assertEqual(float(trimmed[file_trim]), 1.0)

    def test_marker_src_matches_residual_at_every_marker(self) -> None:
        """padded source = dst - residual lag, including the first marker."""
        src, dst = src_dst_from_lags(
            np.array([0.0, 5.0, 10.0]),
            np.array([0.08, 0.08, 0.08]),
            target_sec=10.0,
            in_sec=10.0,
            min_spacing=2.0,
            max_spacing=30.0,
            curvature_gain=1.0,
        )
        self.assertLess(float(np.max(np.abs(src - (dst - 0.08)))), 0.001)
        curves = (
            lambda t: (0.08, 0.9, 0.4),
            lambda t: (0.01 * t, 0.9, 0.4),
            lambda t: (0.04 * np.sin(2 * np.pi * t / 6.0), 0.9, 0.4),
            lambda t: (0.08 if t >= 4.0 else 0.0, 0.9, 0.45),
        )
        for lag_at in curves:
            stem = map_from_points(self._points(lag_at), target_sec=8.0, in_sec=8.0)
            padded = source_on_padded_wav(stem.src, stem.pad_sec)
            residual = np.interp(stem.dst, stem.times, stem.lags - stem.pad_sec)
            err = float(np.max(np.abs(padded - (stem.dst - residual))))
            self.assertLess(err, 0.001, lag_at(1.0))

    def test_marker_sampling_does_not_limit_acceleration(self) -> None:
        """A corner the solver would rewrite is copied through unchanged."""
        times = np.arange(0.0, 8.0, 0.5)
        lag = np.zeros_like(times)
        lag[8:] = 0.03
        src, dst = src_dst_from_lags(
            times,
            lag,
            target_sec=8.0,
            in_sec=8.0,
            min_spacing=0.5,
            max_spacing=0.5,
            curvature_gain=1.0,
        )
        lag_at = np.interp(dst, times, lag)
        self.assertLess(float(np.max(np.abs(src - (dst - lag_at)))), 0.001)
        _limited, stats = _clamp_accel(times, lag, 0.02)
        self.assertGreater(stats["n_modified"], 0)
        self.assertGreater(float(np.max(np.abs(lag - _limited))), 0.001)

    def test_final_map_coordinate_contract(self) -> None:
        """Stored source plus pad is the padded-wav position of the solved lag."""
        cases = (
            (lambda t: (0.10, 0.9, 0.4), 12.0),
            (lambda t: (-0.10, 0.9, 0.4), 12.0),
            (lambda t: (0.012 * t, 0.9, 0.4), 8.0),
            (lambda t: (0.04 * np.sin(2 * np.pi * t / 6.0), 0.9, 0.4), 8.0),
            (lambda t: (0.08 if t >= 4.0 else 0.0, 0.9, 0.45), 8.0),
        )
        for lag_at, length in cases:
            stem = map_from_points(self._points(lag_at), target_sec=length, in_sec=length)
            padded = source_on_padded_wav(stem.src, stem.pad_sec)
            self.assertTrue(np.allclose(padded, stem.src + stem.pad_sec), lag_at(1.0))
            self.assertGreater(len(padded), 1, lag_at(1.0))
            self.assertTrue(np.all(np.diff(stem.dst) > 0), lag_at(1.0))
            self.assertTrue(np.all(np.diff(padded) > -1e-3), lag_at(1.0))
            residual = np.interp(stem.dst, stem.times, stem.lags - stem.pad_sec)
            jump = np.zeros(len(stem.dst), dtype=bool)
            if len(residual) > 1:
                step = np.abs(np.diff(residual)) >= 0.04
                jump[1:] = step
                jump[:-1] |= step
            smooth = ~jump
            if np.any(smooth):
                err = np.abs(padded[smooth] - (stem.dst[smooth] - residual[smooth]))
                self.assertLess(float(np.max(err)), 0.001, lag_at(1.0))
            opened = length + max(float(stem.pad_sec), 0.0)
            # A marker inserted at destination 0, before the first evidence
            # sample, can sit a few milliseconds before file time 0.
            self.assertTrue(np.all(padded >= -0.02), lag_at(1.0))
            self.assertTrue(np.all(padded <= opened + 1e-3), lag_at(1.0))

    def test_rate_limiter_records_how_far_it_moves_the_map(self) -> None:
        times = np.arange(0.0, 8.0, 0.5)
        straight = 0.01 * times
        _limited, quiet = _clamp_accel(times, straight, 0.02)
        self.assertEqual(quiet["n_modified"], 0)
        self.assertLess(quiet["max_sec"], 1e-6)
        corner = np.zeros_like(times)
        corner[8:] = 0.03
        _moved, loud = _clamp_accel(times, corner, 0.02)
        self.assertGreater(loud["n_modified"], 0)
        self.assertGreater(loud["max_sec"], 0.001)
        self.assertGreater(loud["rms_sec"], 0.0)
        stem = map_from_points(self._points(lambda t: (0.02 * t, 0.8, 0.3)), target_sec=8.0, in_sec=8.0)
        recorded = stem.report["rate_limit"]
        self.assertEqual(set(recorded), {"max_sec", "rms_sec", "n_modified"})

    def test_short_envelope_pulls_a_late_hit_onto_the_waveform(self) -> None:
        """A hit 40 ms off the solved lag moves. A hit already on it stays."""
        sr = 22050
        n = int(3.0 * sr)
        ref = np.zeros(n, dtype=np.float32)
        qry = np.zeros(n, dtype=np.float32)
        click = np.hanning(int(0.08 * sr)).astype(np.float32)
        for t, extra in ((1.0, 0.0), (2.0, 0.040)):
            a = int(t * sr)
            ref[a : a + len(click)] = click
            b = int((t + 0.20 + extra) * sr)
            qry[b : b + len(click)] = click
        times = np.array([1.0, 2.0])
        lags = np.array([-0.20, -0.20])
        refined = refine_lags_with_envelope(ref, qry, sr, times, lags)
        self.assertLess(abs(float(refined[0]) - (-0.20)), 0.008)
        self.assertLess(abs(float(refined[1]) - (-0.24)), 0.008)

    def test_unmatched_opening_phrase_moves_onto_its_transient(self) -> None:
        """A phrase 240 ms late moves. The matched section after it stays."""
        sr = 22050
        n = int(16.0 * sr)
        ref = np.zeros(n, dtype=np.float32)
        qry = np.zeros(n, dtype=np.float32)
        pattern = (0.05, 0.14, 0.05, 0.22)

        def burst(y: np.ndarray, at: float) -> None:
            t = at
            for dur in pattern:
                a = int(t * sr)
                samples = int(dur * sr)
                y[a : a + samples] = np.hanning(samples).astype(np.float32)
                t += dur + 0.09

        burst(ref, 5.0)
        burst(qry, 5.24)
        click = np.hanning(int(0.04 * sr)).astype(np.float32)
        for t in np.arange(10.0, 15.0, 0.5):
            a = int(t * sr)
            ref[a : a + len(click)] = click
            qry[a : a + len(click)] = click
        times = np.arange(4.0, 14.0, 1.0)
        lags = np.zeros(len(times), dtype=float)
        moved = pull_unmatched_phrase(ref, qry, sr, times, lags)
        at_five = float(moved[np.argmin(np.abs(times - 5.0))])
        at_twelve = float(moved[np.argmin(np.abs(times - 12.0))])
        self.assertLess(at_five, -0.15)
        self.assertGreater(at_five, -0.36)
        self.assertLess(abs(at_twelve), 0.04)

    def test_short_stem_gap_opens_with_the_reference_silence(self) -> None:
        """The phrase ends when the reference goes quiet. The hit after the gap stays."""
        sr = 22050
        n = int(14.0 * sr)
        ref = np.zeros(n, dtype=np.float32)
        qry = np.zeros(n, dtype=np.float32)
        # Reference is quiet from 7.4 to 8.1. The stem keeps sounding until 7.8.
        ref[int(5.0 * sr) : int(7.4 * sr)] = 0.45
        qry[int(5.0 * sr) : int(7.8 * sr)] = 0.45
        click = np.hanning(int(0.08 * sr)).astype(np.float32) * 0.9
        ref[int(8.1 * sr) : int(8.1 * sr) + len(click)] = click
        qry[int(8.1 * sr) : int(8.1 * sr) + len(click)] = click
        # A later matched section, so the file is long enough to score.
        for t in np.arange(9.5, 13.0, 0.4):
            a = int(t * sr)
            ref[a : a + len(click)] = click * 0.5
            qry[a : a + len(click)] = click * 0.5
        times = np.arange(4.0, 13.0, 1.0)
        lags = np.zeros(len(times), dtype=float)
        new_t, new_lag, pins = fit_reference_gaps(ref, qry, sr, times, lags)
        self.assertGreaterEqual(len(pins), 2)
        lag_at_silence = float(np.interp(7.4, new_t, new_lag))
        lag_at_hit = float(np.interp(8.1, new_t, new_lag))
        self.assertAlmostEqual(7.4 - lag_at_silence, 7.8, delta=0.08)
        self.assertAlmostEqual(lag_at_hit, 0.0, delta=0.05)

    def test_rate_limiter_returns_after_a_blip(self) -> None:
        """One clamped corner must not leave the lag early for the rest of the file."""
        times = np.arange(0.0, 24.0, 1.0)
        lag = np.full(len(times), -0.33)
        lag[6] = -0.36
        limited, stats = _clamp_accel(times, lag, 0.02)
        self.assertGreater(stats["n_modified"], 0)
        tail = limited[14:]
        self.assertLess(float(np.max(np.abs(tail - (-0.33)))), 0.01)

    def test_jumps_and_spikes_in_the_marker_map(self) -> None:
        def hold(step: float):
            def lag_at(t: float):
                return (step if t >= 4.0 else 0.0), 0.9, 0.45
            return map_from_points(self._points(lag_at), target_sec=8.0, in_sec=8.0)

        for step in (0.08, 0.20):
            stem = hold(step)
            early = float(np.median(stem.lags[stem.times < 3.5]))
            late = float(np.median(stem.lags[stem.times > 5.0]))
            self.assertGreater(late - early, step * 0.6, step)
            self.assertTrue(np.all(np.diff(stem.src) >= -1e-3))
            self.assertTrue(np.all(np.diff(stem.dst) >= -1e-3))

        def weak_at(*centers: float):
            def lag_at(t: float):
                if any(abs(t - c) < 0.2 for c in centers):
                    return 0.05, 0.15, 0.01
                return 0.0, 0.85, 0.40
            return lag_at

        for centers in ((4.0,), (3.0, 5.0)):
            stem = map_from_points(self._points(weak_at(*centers)), target_sec=8.0, in_sec=8.0)
            self.assertLess(float(np.max(np.abs(stem.lags))), 0.03, centers)


class GapTests(unittest.TestCase):
    def test_stall_is_silence_and_monotonic(self) -> None:
        units = [
            PhraseUnit(0.0, 2.0, 0.4, 2.4, 0.4, 0.4, 0.0, 0.8, 0.4),
            PhraseUnit(2.0, 4.0, 5.0, 7.2, 3.0, 3.2, 0.2, 0.75, 0.3),
        ]
        spans = spans_from_units(units, 8.0)
        self.assertTrue(any(span.kind == "silence" and span.dst_end - span.dst_start > 1.0 for span in spans))
        for prev, span in zip(spans, spans[1:]):
            self.assertGreaterEqual(span.dst_start + 1e-6, prev.dst_end)
        y = np.ones(int(4.0 * 8000), dtype=np.float32)
        out = render_gap_aware(y, 8000, spans, int(8.0 * 8000), engine="linear")
        self.assertEqual(len(out), int(8.0 * 8000))
        silence = next(span for span in spans if span.kind == "silence")
        i0 = int(silence.dst_start * 8000) + 20
        i1 = int(silence.dst_end * 8000) - 20
        self.assertLess(float(np.max(np.abs(out[i0:i1]))), 1e-6)

    def test_near_identity_phrases_skip_elastique(self) -> None:
        """A 1%% length change is paste work. Only larger warps need REAPER."""
        self.assertFalse(_needs_elastique(44100, 44100, 44100))
        self.assertFalse(_needs_elastique(44100, int(44100 * 1.01), 44100))
        self.assertTrue(_needs_elastique(44100, int(44100 * 1.05), 44100))

    def test_two_phrases_keep_their_reference_slots(self) -> None:
        """Removed internal silence stays silent, and each phrase keeps its reference slot."""
        sr = 8000
        phrase_a = np.sin(2 * np.pi * 196.0 * np.arange(4 * sr) / sr).astype(np.float32)
        phrase_b = np.sin(2 * np.pi * 330.0 * np.arange(4 * sr) / sr).astype(np.float32)
        acapella = np.concatenate([phrase_a, phrase_b])
        units = [
            PhraseUnit(0.0, 4.0, 5.0, 9.0, 5.0, 5.0, 0.0, 0.9, 0.4),
            PhraseUnit(4.0, 8.0, 10.0, 14.0, 6.0, 6.0, 0.0, 0.9, 0.4),
        ]
        spans = spans_from_units(units, 15.0)
        speech = [span for span in spans if span.kind == "speech"]
        self.assertEqual(len(speech), 2)
        self.assertAlmostEqual(speech[0].dst_start, 5.0, delta=0.05)
        self.assertAlmostEqual(speech[1].dst_start, 10.0, delta=0.05)
        out = render_gap_aware(acapella, sr, spans, 15 * sr, engine="linear")[:, 0]
        gap = out[int(9.2 * sr) : int(9.8 * sr)]
        self.assertLess(float(np.max(np.abs(gap))), 1e-6)
        early = out[int(6.0 * sr) : int(8.0 * sr)]
        late = out[int(11.0 * sr) : int(13.0 * sr)]
        self.assertGreater(float(np.mean(np.abs(early))), 0.2)
        self.assertGreater(float(np.mean(np.abs(late))), 0.2)

        def peak_hz(y: np.ndarray) -> float:
            spec = np.abs(np.fft.rfft(y))
            return float(np.fft.rfftfreq(len(y), 1.0 / sr)[int(np.argmax(spec))])

        self.assertLess(abs(peak_hz(early) - 196.0), 4.0)
        self.assertLess(abs(peak_hz(late) - 330.0), 4.0)

    def test_source_tail_survives_the_stall(self) -> None:
        """The decay after the silence gate stays in the render. The inserted rest does not."""
        sr = 8000
        phrase = np.ones(sr, dtype=np.float32)
        tail = np.linspace(0.6, 0.05, int(0.8 * sr), dtype=np.float32)
        rest = np.zeros(int(1.2 * sr), dtype=np.float32)
        nxt = np.full(sr, 0.25, dtype=np.float32)
        y = np.concatenate([phrase, tail, rest, nxt])
        units = [
            PhraseUnit(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.9, 0.4),
            PhraseUnit(3.0, 4.0, 4.0, 5.0, 1.0, 1.0, 0.0, 0.9, 0.4),
        ]
        spans = spans_from_units(units, 6.0)
        out = render_gap_aware(y, sr, spans, 6 * sr, engine="linear")[:, 0]
        kept = out[int(1.15 * sr) : int(1.6 * sr)]
        self.assertGreater(float(np.max(np.abs(kept))), 0.2)
        stall = out[int(2.4 * sr) : int(3.4 * sr)]
        self.assertLess(float(np.max(np.abs(stall))), 1e-5)
        second = out[int(4.2 * sr) : int(4.8 * sr)]
        self.assertGreater(float(np.mean(np.abs(second))), 0.15)

    def test_short_speech_keeps_its_solved_duration(self) -> None:
        spans = _with_silence(
            [
                MapSpan("speech", 0.0, 0.015, 1.0, 1.015),
                MapSpan("speech", 0.015, 0.015, 1.5, 1.5),
            ],
            2.0,
        )
        speech = [span for span in spans if span.kind == "speech"]
        self.assertAlmostEqual(speech[0].dst_end - speech[0].dst_start, 0.015, places=4)
        self.assertGreater(speech[1].dst_end, speech[1].dst_start)
        self.assertLess(speech[1].dst_end - speech[1].dst_start, 0.005)

    def test_overlap_clips_the_earlier_phrase(self) -> None:
        spans = _with_silence(
            [
                MapSpan("speech", 0.0, 1.0, 11.0, 12.0),
                MapSpan("speech", 1.0, 2.0, 11.9, 12.9),
            ],
            14.0,
        )
        speech = [span for span in spans if span.kind == "speech"]
        self.assertEqual(len(speech), 2)
        self.assertAlmostEqual(speech[0].dst_end, 11.9, places=3)
        self.assertAlmostEqual(speech[0].src_end, 0.9, places=3)
        self.assertAlmostEqual(speech[1].dst_start, 11.9, places=3)
        self.assertAlmostEqual(speech[1].dst_end, 12.9, places=3)

    def test_uncovered_edge_uses_the_detector_placement(self) -> None:
        unit = PhraseUnit(0.0, 2.0, 10.0, 12.0, 10.0, 10.0, 0.0, 0.9, 0.4)
        late_dest = _cover_placed_edges([MapSpan("speech", 0.0, 1.5, 10.08, 11.5)], unit)
        self.assertAlmostEqual(late_dest[0].dst_start, 10.0, places=3)
        self.assertAlmostEqual(late_dest[0].src_start, 0.0, places=3)
        short_source = _cover_placed_edges([MapSpan("speech", 0.0, 1.5, 10.0, 12.0)], unit)
        self.assertAlmostEqual(short_source[-1].src_end, 2.0, places=3)
        self.assertAlmostEqual(short_source[-1].dst_end, 12.0, places=3)

    def test_missing_demucs_part_keeps_the_later_entrance(self) -> None:
        """A distinct later peak stays when the chain is a vocal the acapella lacks."""
        self.assertTrue(
            _keep_later_phrase(kind="wave", score=0.53, ratio=14.2, chain_ncc=0.03)
        )
        self.assertTrue(
            _keep_later_phrase(kind="wave", score=0.63, ratio=15.0, chain_ncc=0.02)
        )
        self.assertFalse(
            _keep_later_phrase(kind="wave", score=0.53, ratio=1.1, chain_ncc=0.03)
        )
        self.assertFalse(
            _keep_later_phrase(kind="wave", score=0.80, ratio=4.0, chain_ncc=0.70)
        )
        self.assertFalse(
            _keep_later_phrase(kind="wave", score=0.40, ratio=20.0, chain_ncc=0.0)
        )

    def test_pre_roll_before_the_file_stays_silent(self) -> None:
        """Source before time 0 is silence. It must not pull the attack early."""
        span = trim_span_to_file(MapSpan("speech", -0.115, 3.878, 2.445, 6.438), 10.0)
        self.assertIsNotNone(span)
        assert span is not None
        self.assertGreaterEqual(span.src_start, 0.0)
        rate = (span.dst_end - span.dst_start) / (span.src_end - span.src_start)
        attack = span.dst_start + (0.202 - span.src_start) * rate
        self.assertGreater(attack, 2.74)
        self.assertLess(attack, 2.78)
        sr = 8000
        click_at = int(0.202 * sr)
        audio = np.zeros((int(4.0 * sr), 1), dtype=np.float32)
        audio[click_at : click_at + 40, 0] = 1.0
        rendered = render_gap_aware(
            audio,
            sr,
            [MapSpan("speech", -0.115, 3.878, 2.445, 6.438)],
            int(8.0 * sr),
            engine="linear",
        )
        energy = np.sqrt(np.mean(rendered.reshape(-1) ** 2))
        self.assertGreater(energy, 0.0)
        hop = int(0.002 * sr)
        first = None
        for i in range(0, len(rendered) - hop, hop):
            if float(np.max(np.abs(rendered[i : i + hop]))) > 0.2:
                first = i / sr
                break
        self.assertIsNotNone(first)
        assert first is not None
        self.assertGreater(first, 2.74)
        self.assertLess(first, 2.80)

    def test_early_lock_keeps_the_placed_entrance(self) -> None:
        """A map that opens on the vocal before the placement is discarded."""
        unit = PhraseUnit(10.0, 14.0, 40.0, 44.0, 30.0, 30.0, 0.0, 0.9, 0.4)
        locked = _hold_placed_entrance([MapSpan("speech", 10.0, 14.0, 38.4, 42.4)], unit)
        self.assertIsNone(locked)
        nudge = _hold_placed_entrance([MapSpan("speech", 10.0, 14.0, 39.95, 43.95)], unit)
        self.assertIsNotNone(nudge)
        self.assertAlmostEqual(nudge[0].dst_start, 39.95, places=3)

    def test_weak_placement_keeps_a_confident_earlier_map(self) -> None:
        """A 0.13 placement can sit late. A confident map a fraction earlier stands."""
        unit = PhraseUnit(10.0, 14.0, 40.0, 44.0, 30.0, 30.0, 0.0, 0.13, 0.08)
        kept = _hold_placed_entrance(
            [MapSpan("speech", 10.0, 14.0, 39.78, 43.78)],
            unit,
            map_confidence=0.91,
        )
        self.assertIsNotNone(kept)
        self.assertAlmostEqual(kept[0].dst_start, 39.78, places=3)
        unsure = _hold_placed_entrance(
            [MapSpan("speech", 10.0, 14.0, 39.78, 43.78)],
            unit,
            map_confidence=0.30,
        )
        self.assertIsNone(unsure)
        far = _hold_placed_entrance(
            [MapSpan("speech", 10.0, 14.0, 38.4, 42.4)],
            unit,
            map_confidence=0.91,
        )
        self.assertIsNone(far)

    def test_long_phrase_does_not_start_on_the_lookbehind(self) -> None:
        """The entrance stays where it was placed when Demucs still has vocal just before it."""
        dur = 36.0
        phrase = harmonic(SR, dur, 220.0)
        total = int(80.0 * SR)
        ref = np.zeros(total, dtype=np.float32)
        qry = np.zeros(total, dtype=np.float32)
        placed = 40.0
        src = 2.0
        decoy = placed - 1.7
        opening = phrase[: int(2.0 * SR)]
        ref[int(decoy * SR) : int(decoy * SR) + len(opening)] = opening
        ref[int(placed * SR) : int(placed * SR) + len(phrase)] = phrase
        qry[int(src * SR) : int(src * SR) + len(phrase)] = phrase
        unit = PhraseUnit(src, src + dur, placed, placed + dur, placed - src, placed - src, 0.0, 0.9, 0.4)
        profile = resolve_profile(ref, SR, name="default")
        spans = _model_phrase(unit, qry, ref, SR, profile, None, None)
        self.assertIsNotNone(spans)
        self.assertGreater(_dst_at_src(spans, src), placed - 0.15)

    def test_phrase_model_rejects_a_stolen_earlier_source(self) -> None:
        """A multi-second pad that reaches before the phrase is discarded."""
        dur = 5.0
        earlier = harmonic(SR, 3.0, 196.0)
        phrase = harmonic(SR, dur, 247.0)
        total = int(30.0 * SR)
        ref = np.zeros(total, dtype=np.float32)
        qry = np.zeros(total, dtype=np.float32)
        src = 5.0
        placed = 12.0
        # Earlier lead vocal sits before the phrase in the acapella.
        qry[int(1.0 * SR) : int(1.0 * SR) + len(earlier)] = earlier
        qry[int(src * SR) : int(src * SR) + len(phrase)] = phrase
        # Demucs only has that earlier vocal in the lookbehind, then silence,
        # then the real phrase. A body pad can lock onto the decoy.
        ref[int((placed - 4.2) * SR) : int((placed - 4.2) * SR) + len(earlier)] = earlier
        ref[int(placed * SR) : int(placed * SR) + len(phrase)] = phrase
        unit = PhraseUnit(
            src, src + dur, placed, placed + dur, placed - src, placed - src, 0.0, 0.05, 0.02
        )
        profile = resolve_profile(ref, SR, name="default", max_pad_sec=10.0)
        spans = _model_phrase(unit, qry, ref, SR, profile, None, None)
        if spans is None:
            return
        for span in spans:
            self.assertGreaterEqual(float(span.src_start), src - 0.35)

    def test_short_phrase_is_not_stretched_across_the_lookahead(self) -> None:
        dur = 0.95
        phrase = harmonic(SR, dur, 330.0)
        total = int(8.0 * SR)
        ref = np.zeros(total, dtype=np.float32)
        qry = np.zeros(total, dtype=np.float32)
        src = 1.0
        placed = 3.0
        n = len(phrase)
        ref[int(placed * SR) : int(placed * SR) + n] = phrase
        qry[int(src * SR) : int(src * SR) + n] = phrase
        unit = PhraseUnit(
            src, src + dur, placed, placed + dur, placed - src, placed - src, 0.0, 0.4, 0.2
        )
        profile = resolve_profile(ref, SR, name="default")
        spans = _model_phrase(unit, qry, ref, SR, profile, None, None)
        if spans is None:
            return
        for span in spans:
            width = float(span.src_end - span.src_start)
            rate = float(span.dst_end - span.dst_start) / width
            self.assertLess(rate, 1.08, f"rate {rate:.3f}")
        self.assertAlmostEqual(spans[0].dst_start, placed, delta=0.12)

    def test_silence_does_not_outrank_the_phrase(self) -> None:
        qry = harmonic(SR, 1.2, 440.0).astype(np.float64)
        ref = np.zeros(int(6.0 * SR), dtype=np.float64)
        at = int(2.5 * SR)
        ref[at : at + len(qry)] = qry
        lag, score, _ratio = waveform_ncc_offset(
            ref, qry, sr=SR, max_lag_sec=5.0, finger_sec=2.0
        )
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.5)
        self.assertAlmostEqual(lag, 2.5, delta=0.05)

    def test_rest_entrance_beats_a_later_repeat(self) -> None:
        """A phrase chained into a rest starts when the rest ends, not on a later copy."""
        sr = 22050
        phrase = harmonic(sr, 1.2, 220.0).astype(np.float32) * 0.8
        repeat = harmonic(sr, 1.2, 220.0).astype(np.float32) * 0.8
        lead = harmonic(sr, 1.0, 330.0).astype(np.float32) * 0.7
        ref = np.zeros(int(30.0 * sr), dtype=np.float32)
        ref[int(1.0 * sr) : int(1.0 * sr) + len(lead)] = lead
        ref[int(10.0 * sr) : int(10.0 * sr) + len(phrase)] = phrase
        ref[int(22.0 * sr) : int(22.0 * sr) + len(repeat)] = repeat
        aca = np.zeros(int(8.0 * sr), dtype=np.float32)
        aca[int(0.2 * sr) : int(0.2 * sr) + len(lead)] = lead
        # File gap puts the chain inside the rest, before the real entrance.
        aca[int(6.0 * sr) : int(6.0 * sr) + len(phrase)] = phrase
        _src, dst, _notes = place_segments(aca, ref, sr=sr, prefer_wave=True)
        self.assertGreaterEqual(len(dst), 2)
        self.assertAlmostEqual(dst[-1][0], 10.0, delta=0.4)
        self.assertLess(dst[-1][0], 12.0)

    def test_snap_keeps_gradual_tempo_inserts(self) -> None:
        """15–40 ms Mel/aca tempo steps must not flatten to the first pad."""
        segs = [(0.0, 2.0), (2.5, 4.5), (5.0, 7.0), (7.5, 9.5)]
        # Contiguous file gaps of 0.5 s; dest gaps grow by ~25 ms each (tempo).
        placements = [
            (2.578, 4.578, 0.8),
            (5.103, 7.103, 0.8),  # insert +25 ms vs file_gap
            (7.653, 9.653, 0.8),  # +25 ms again
            (10.228, 12.228, 0.8),
        ]
        snapped, n = snap_tiny_inserts(segs, placements)
        self.assertEqual(n, 0)
        offsets = [p[0] - s[0] for s, p in zip(segs, snapped)]
        self.assertAlmostEqual(offsets[0], 2.578, places=3)
        self.assertAlmostEqual(offsets[-1], 2.728, places=3)
        self.assertGreater(offsets[-1] - offsets[0], 0.10)

    def test_snap_still_drops_micro_jitter(self) -> None:
        segs = [(0.0, 2.0), (2.5, 4.5)]
        placements = [(1.0, 3.0, 0.8), (3.508, 5.508, 0.8)]  # +8 ms insert
        snapped, n = snap_tiny_inserts(segs, placements)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(snapped[1][0], 3.5, places=3)

    def test_gap_report_names_the_render_map(self) -> None:
        row = combine_reports(
            {
                "map_type": "gap_spans",
                "offset_sec": 0.123,
                "drift": 0.0003,
                "global_offset_sec": 0.123,
                "global_drift": 0.0003,
                "render_pad_sec": 0.0,
                "phrase_start_offset": 0.0,
                "times": [0.0, 1.0],
                "lags": [0.0, 0.1],
                "confidence": 0.8,
                "failures": [],
            },
            {
                "offset_sec": 0.01,
                "drift": 0.0,
                "times": [0.0],
                "lags": [0.0],
                "confidence": 0.9,
                "failures": [],
            },
            profile="default",
        )
        self.assertEqual(row["map_type"], "gap_spans")
        self.assertAlmostEqual(row["global_offset_sec"], 0.123)
        self.assertAlmostEqual(row["render_pad_sec"], 0.0)

    def test_phrase_offset_is_applied_once(self) -> None:
        """Query 9.900–11.900 against reference 10.000–12.000 stays a +100 ms map."""
        phrase = harmonic(SR, 2.0, 220.0)
        click = int(1.0 * SR)
        phrase = phrase.copy()
        phrase[click : click + 8] = 1.0
        total = int(14.0 * SR)
        ref = np.zeros(total, dtype=np.float32)
        qry = np.zeros(total, dtype=np.float32)
        ref[int(10.0 * SR) : int(10.0 * SR) + len(phrase)] = phrase
        qry[int(9.9 * SR) : int(9.9 * SR) + len(phrase)] = phrase
        unit = PhraseUnit(9.9, 11.9, 10.0, 12.0, 0.1, 0.1, 0.0, 0.9, 0.4)
        profile = resolve_profile(ref, SR, name="default", win_sec=1.2, step_sec=0.4, max_lag_sec=0.6)
        spans = _model_phrase(unit, qry, ref, SR, profile, None, None)
        self.assertIsNotNone(spans)

        def dst_at(src: float) -> float:
            for span in spans:
                if span.kind != "speech":
                    continue
                if span.src_start - 1e-3 <= src <= span.src_end + 1e-3:
                    width = span.src_end - span.src_start
                    share = 0.0 if width <= 1e-6 else (src - span.src_start) / width
                    return span.dst_start + share * (span.dst_end - span.dst_start)
            self.fail(f"no speech span covers source {src}")

        self.assertLess(abs(dst_at(9.9) - 10.0), 0.03)
        self.assertLess(abs(dst_at(11.9) - 12.0), 0.03)
        rendered = render_gap_aware(qry, SR, spans, total, engine="linear")[:, 0]
        i0, i1 = int(10.5 * SR), int(11.5 * SR)
        landed = (i0 + int(np.argmax(np.abs(rendered[i0:i1])))) / SR
        self.assertLess(abs(landed - 11.0), 0.03)

    def test_continuous_vocal_is_one_global_map(self) -> None:
        """Same timing stays on map_stem: one curve, no restored rest."""
        row = case_continuous_vocal()
        self.assertEqual(row["category"], "global")
        self.assertEqual(row["n_phrases"], 0)
        self.assertFalse(row["silence"])
        self.assertTrue(row["monotonic"])
        self.assertLess(row["errors"]["mae"], 0.05)
        self.assertLess(row["max_abs_lag"], 0.05)

    def test_stripped_vocal_does_not_inherit_the_short_gap(self) -> None:
        """Phrase B keeps the reference rest. The short breath is not its timing."""
        row = case_stripped_vocal()
        self.assertEqual(row["category"], "gaps")
        self.assertGreaterEqual(row["n_phrases"], 2, row["gap_notes"])
        self.assertTrue(row["monotonic"])
        self.assertLess(abs(row["phrase_dst"][0]), 0.30)
        self.assertLess(abs(row["phrase_dst"][1] - row["b_reference"]), 0.40)
        self.assertGreater(row["phrase_dst"][1] - row["b_if_chained"], 1.0)
        self.assertTrue(any(dur >= 1.5 for dur in row["durations"]))
        self.assertLess(row["stall_peak"], 1e-4)
        self.assertGreater(row["stall_start"], 2.15)
        self.assertGreater(row["speech_a_end"], 2.20)
        self.assertGreater(row["speech_b_end"], 7.05)
        self.assertLess(abs(row["pitch_a"] - 196.0), 6.0)
        self.assertLess(abs(row["pitch_b"] - 330.0), 6.0)
        self.assertLess(abs(row["pitch_a_tail"] - 196.0), 6.0)
        self.assertLess(abs(row["pitch_b_tail"] - 330.0), 6.0)
        self.assertLess(row["errors"]["mae"], 0.25)
        self.assertIsNotNone(row["phrase_start_offset"])
        self.assertIsNotNone(row["global_pad_sec"])
        self.assertAlmostEqual(row["pad_sec"], float(row["phrase_start_offset"]), places=5)


class QualityTests(unittest.TestCase):
    def test_codes_and_compensation(self) -> None:
        times = np.arange(0.5, 8.0, 0.5)
        lags = np.full(len(times), 0.2)
        report = measure_curve(times, lags, np.full(len(times), 0.2))
        coded = apply_codes(report, margins=np.full(len(times), 0.01))
        codes = {item["code"] for item in coded["failures"]}
        self.assertIn("OFFSET_TOO_LARGE", codes)
        self.assertIn("LOW_SIGNAL", codes)
        self.assertIn("AMBIGUOUS_MATCH", codes)
        self.assertTrue(
            is_compensating(
                {"verdict": "pass", "lag_sec": 0.03},
                {"verdict": "pass", "lag_sec": -0.04},
                {"verdict": "fail", "lag_sec": 0.0},
            )
        )
        self.assertFalse(
            is_compensating(
                {"verdict": "pass", "lag_sec": 0.03},
                {"verdict": "pass", "lag_sec": 0.02},
                {"verdict": "fail", "lag_sec": 0.0},
            )
        )

        class Row:
            aca_verdict = "pass"
            inst_verdict = "pass"
            aca_corr = 0.8
            inst_corr = 0.8
            aca_lag_sec = 0.04
            inst_lag_sec = -0.05
            aca_check_drift_ms = 4.0
            inst_check_drift_ms = 4.0
            alignment_report = {"failures": [], "offset_sec": 0.0, "profile": "default"}
            verdict = "pass"
            notes = "mix was off"

        row = Row()
        stamp_result(row, mix_verdict="fail", mix_corr=0.2, mix_lag=0.0, mix_drift=40.0)
        self.assertEqual(row.verdict, "fail")
        self.assertIn("MIX_COMPENSATION", row.notes)
        self.assertEqual(row.alignment_report["vocal"]["verdict"], "pass")
        self.assertEqual(row.alignment_report["mix"]["verdict"], "fail")


class BenchmarkTests(unittest.TestCase):
    def test_offset_drift_repeat_and_gaps(self) -> None:
        perfect = case_perfect()
        self.assertLess(perfect["errors"]["mae"], 0.05)
        self.assertTrue(perfect["monotonic"])

        offset = case_offset()
        self.assertLess(abs(offset["offset"] - offset["true_offset"]), 0.06)
        self.assertLess(offset["errors"]["mae"], 0.08)
        self.assertTrue(offset["monotonic"])

        drift = case_drift()
        self.assertLess(abs(drift["drift"] - drift["true_slope"]), 0.008)
        self.assertTrue(drift["monotonic"])

        repeated = case_repeated()
        self.assertLess(repeated["errors"]["mae"], 0.35)
        self.assertTrue(repeated["monotonic"])

        gaps = case_gaps()
        self.assertTrue(gaps["monotonic"])
        self.assertTrue(gaps["silence"])
        self.assertTrue(_one_speech_run(gaps["speech"]))

        nonlinear = case_nonlinear()
        self.assertLess(nonlinear["errors"]["mae"], 0.08)
        self.assertTrue(nonlinear["monotonic"])

        repeated_ok = case_noise()
        self.assertLess(repeated_ok["errors"]["mae"], 0.1)
        self.assertTrue(repeated_ok["monotonic"])

        sparse = case_sparse()
        self.assertLess(sparse["errors"]["mae"], 0.12)
        self.assertTrue(sparse["monotonic"])

        tempo = case_tempo_change()
        self.assertTrue(tempo["monotonic"])
        self.assertLess(tempo["errors"]["mae"], 0.12)
        false_pos, missed = discontinuity_counts(tempo["jumps"], tempo["true_jumps"], tol=1.5)
        self.assertEqual(missed, 0)
        self.assertLessEqual(false_pos, 2)

    def test_ground_truth_suite(self) -> None:
        from align_model.benchmark import run_all

        rows = run_all()
        by_name = {row["name"]: row for row in rows}
        self.assertGreaterEqual(len(rows), 32)
        for row in rows:
            self.assertTrue(row["monotonic"], row["name"])
            if row["name"] == "missing_internal_gaps":
                # No breath in the acapella, so this stays one speech run.
                # The restored rest is stripped_vocal.
                self.assertTrue(row["silence"])
                self.assertTrue(_one_speech_run(row["speech"]))
                continue
            if row["name"] == "continuous_vocal":
                self.assertEqual(row["n_phrases"], 0)
                self.assertFalse(row["silence"])
                self.assertLess(row["max_abs_lag"], 0.05)
            if row["name"] == "stripped_vocal":
                self.assertGreaterEqual(row["n_phrases"], 2, row.get("gap_notes"))
                self.assertLess(abs(row["phrase_dst"][1] - row["b_reference"]), 0.40)
                self.assertGreater(row["phrase_dst"][1] - row["b_if_chained"], 1.0)
                self.assertLess(row["stall_peak"], 1e-4)
                self.assertGreater(row["speech_a_end"], 2.20)
                self.assertGreater(row["speech_b_end"], 7.05)
                self.assertLess(row["errors"]["mae"], 0.25)
                self.assertAlmostEqual(row["pad_sec"], float(row["phrase_start_offset"]), places=5)
                self.assertIsNotNone(row["global_pad_sec"])
                continue
            self.assertLess(row["errors"]["mae"], 0.25, row["name"])
        for name, truth in (
            ("offset_50ms", -0.05),
            ("offset_m50ms", 0.05),
            ("offset_p200ms", -0.20),
            ("offset_m200ms", 0.20),
        ):
            self.assertLess(abs(by_name[name]["offset"] - truth), 0.05, name)
        for name, slope in (
            ("drift_0_1pct", -0.001),
            ("drift_0_5pct", -0.005),
            ("drift_1_0pct", -0.010),
        ):
            self.assertLess(abs(by_name[name]["drift"] - slope), 0.006, name)
            self.assertLess(by_name[name]["errors"]["mae"], 0.02, name)
        self.assertLess(by_name["nonlinear_drift"]["errors"]["mae"], 0.08)
        self.assertLess(by_name["leading_silence"]["errors"]["mae"], 0.12)
        self.assertLess(by_name["trailing_silence"]["errors"]["mae"], 0.12)
        self.assertLess(by_name["repeated_section"]["errors"]["mae"], 0.35)
        middle = by_name["middle_discontinuity"]
        for part, err in middle["section_mae"].items():
            self.assertLess(err, 0.05, part)
        false_pos, missed = discontinuity_counts(middle["jumps"], middle["true_jumps"], tol=2.0)
        self.assertEqual(missed, 0)
        self.assertLessEqual(false_pos, 2)

    def test_profile_override(self) -> None:
        profile = resolve_profile(None, 0, name="edm", win_sec=5.0)
        self.assertEqual(profile.name, "edm")
        self.assertEqual(profile.win_sec, 5.0)
        self.assertGreater(profile.weights["onset"], profile.weights["chroma"])
        self.assertFalse(hasattr(profile, "corr_min"))
        self.assertFalse(hasattr(profile, "drift_ms"))
        self.assertAlmostEqual(profile.fit_tol_sec, 0.020)
        self.assertAlmostEqual(resolve_profile(None, 0, name="acoustic").fit_tol_sec, 0.030)
        self.assertAlmostEqual(resolve_profile(None, 0, name="sparse_vocal").fit_tol_sec, 0.018)
        stock = resolve_profile(None, 0, name="edm", max_pad_sec=90.0)
        self.assertEqual(stock.max_pad_sec, 30.0)


if __name__ == "__main__":
    unittest.main()
