"""Math and synthetic-audio checks for the alignment model."""

from __future__ import annotations

import unittest

import numpy as np

from align_model.benchmark import (
    case_continuous_vocal,
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
from align_model.evidence import EvidencePoint, context_weights, fuse_candidates
from align_model.gaps import PhraseUnit, render_gap_aware, spans_from_units
from align_model.params import resolve_profile
from align_model.pipeline import map_from_points
from align_model.quality import apply_codes, is_compensating, stamp_result
from align_model.time_map import (
    _clamp_accel,
    adaptive_marker_times,
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
