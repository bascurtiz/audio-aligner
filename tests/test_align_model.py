"""Math and synthetic-audio checks for the alignment model."""

from __future__ import annotations

import unittest

import numpy as np

from align_model.benchmark import (
    case_drift,
    case_gaps,
    case_noise,
    case_nonlinear,
    case_offset,
    case_perfect,
    case_repeated,
    case_sparse,
    case_tempo_change,
    discontinuity_counts,
    measure_curve,
)
from align_model.beats import beat_score_at
from align_model.evidence import EvidencePoint, context_weights, fuse_candidates
from align_model.gaps import PhraseUnit, render_gap_aware, spans_from_units
from align_model.params import resolve_profile
from align_model.pipeline import map_from_points
from align_model.quality import apply_codes, is_compensating, stamp_result
from align_model.time_map import adaptive_marker_times


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
        beats = np.arange(0.0, 4.0, 0.5)
        self.assertGreater(beat_score_at(1.0, 0.0, beats, beats[::4], 0.5), 0.8)
        self.assertLess(beat_score_at(1.0, 0.2, beats, beats[::4], 0.5), 0.4)


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
        self.assertTrue(any(0.7 <= dur <= 2.4 for dur in gaps["durations"]))

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

    def test_profile_override(self) -> None:
        profile = resolve_profile(None, 0, name="edm", drift_ms=12.0, win_sec=5.0)
        self.assertEqual(profile.name, "edm")
        self.assertEqual(profile.drift_ms, 12.0)
        self.assertEqual(profile.win_sec, 5.0)
        self.assertGreater(profile.weights["onset"], profile.weights["chroma"])
        stock = resolve_profile(None, 0, name="edm", max_pad_sec=90.0, corr_min=0.35, drift_ms=20.0)
        self.assertEqual(stock.max_pad_sec, 30.0)
        self.assertEqual(stock.corr_min, 0.30)
        self.assertEqual(stock.drift_ms, 25.0)


if __name__ == "__main__":
    unittest.main()
