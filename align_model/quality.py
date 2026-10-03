"""Multidimensional score, failure codes, and mix-compensation check."""

from __future__ import annotations

import numpy as np

from align_model.decompose import AlignmentReport

OFFSET_TOO_LARGE_SEC = 0.05
TEMPO_DRIFT_MS_PER_MIN = 30.0
DISCONTINUITY_SEC = 0.04
AMBIGUOUS_MARGIN = 0.08
LOW_SIGNAL_WEAK = 0.35
GAP_CONFIDENCE = 0.20
GAP_MARGIN = 0.05

CODES = (
    "OFFSET_TOO_LARGE",
    "TEMPO_DRIFT",
    "LOCAL_DISCONTINUITY",
    "AMBIGUOUS_MATCH",
    "LOW_SIGNAL",
    "GAP_RESTORATION_UNCERTAIN",
    "MIX_COMPENSATION",
)


def _failure(code: str, time: float, detail: str) -> dict:
    return {"code": code, "time": round(float(time), 3), "detail": detail}


def apply_codes(
    report: AlignmentReport | dict,
    *,
    margins: np.ndarray | None = None,
    gap_units: list | None = None,
) -> dict:
    """Attach diagnostic codes. Passing a policy threshold does not clear them."""
    data = report.as_dict() if isinstance(report, AlignmentReport) else dict(report)
    kept = [item for item in data.get("failures") or [] if item.get("code") == "MIX_COMPENSATION"]
    failures = list(kept)
    offset = float(data.get("offset_sec") or 0.0)
    if abs(offset) >= OFFSET_TOO_LARGE_SEC:
        failures.append(_failure("OFFSET_TOO_LARGE", 0.0, f"offset {offset:+.3f}s"))
    drift = float(data.get("drift_ms_per_min") or 0.0)
    if abs(drift) >= TEMPO_DRIFT_MS_PER_MIN:
        failures.append(_failure("TEMPO_DRIFT", 0.0, f"drift {drift:+.1f} ms/min"))
    for jump in data.get("jumps") or []:
        size = float(jump.get("size_sec") or 0.0)
        if abs(size) >= DISCONTINUITY_SEC:
            failures.append(
                _failure("LOCAL_DISCONTINUITY", float(jump.get("time") or 0.0), f"{size * 1000:.0f} ms")
            )
    if margins is not None and len(margins):
        margins = np.asarray(margins, dtype=float)
        i = int(np.argmin(margins))
        if float(margins[i]) < AMBIGUOUS_MARGIN:
            times = data.get("times") or []
            when = float(times[i]) if i < len(times) else 0.0
            failures.append(_failure("AMBIGUOUS_MATCH", when, f"margin {float(margins[i]):.3f}"))
    weak = float(data.get("weak_frac") or 0.0)
    if weak >= LOW_SIGNAL_WEAK:
        failures.append(_failure("LOW_SIGNAL", 0.0, f"weak {weak:.0%}"))
    uncertain = 0
    for unit in gap_units or []:
        conf = float(getattr(unit, "confidence", 0.0))
        margin = float(getattr(unit, "match_margin", 0.0))
        if conf >= GAP_CONFIDENCE and margin >= GAP_MARGIN:
            continue
        when = float(getattr(unit, "dst_start", 0.0))
        failures.append(_failure("GAP_RESTORATION_UNCERTAIN", when, f"conf {conf:.2f}"))
        uncertain += 1
        if uncertain >= 8:
            break
    data["failures"] = failures
    if isinstance(report, AlignmentReport):
        report.failures = failures
    return data


def post_render_validation(aca, inst, orig, **kwargs) -> dict:
    """Score the rendered mix against the original.

    This is an independent check. It is not the alignment model's own confidence.
    """
    from check_alignment import analyze_alignment

    verdict, corr, lag, drift, weak, notes = analyze_alignment(aca, inst, orig, **kwargs)
    return {
        "source": "post_render_validation",
        "verdict": verdict,
        "corr": float(corr),
        "lag_sec": float(lag),
        "drift_ms": float(drift),
        "weak_frac": float(weak),
        "notes": notes,
    }


def relationship(verdict: str, corr: float, lag_sec: float, drift_ms: float) -> dict:
    return {
        "verdict": verdict,
        "corr": float(corr),
        "lag_sec": float(lag_sec),
        "drift_ms": float(drift_ms),
        "confidence": float(corr),
    }


def is_compensating(vocal: dict, instrumental: dict, mix: dict) -> bool:
    """Both stems pass, the mix fails, and the stem lags point opposite ways."""
    if vocal.get("verdict") != "pass" or instrumental.get("verdict") != "pass":
        return False
    if mix.get("verdict") != "fail":
        return False
    a = float(vocal.get("lag_sec") or 0.0)
    b = float(instrumental.get("lag_sec") or 0.0)
    return a * b < 0 and abs(a) >= 0.005 and abs(b) >= 0.005


def apply_relationships(report: dict, *, vocal: dict, instrumental: dict, mix: dict) -> dict:
    report = dict(report)
    report["vocal"] = vocal
    report["instrumental"] = instrumental
    report["mix"] = mix
    failures = [item for item in report.get("failures") or [] if item.get("code") != "MIX_COMPENSATION"]
    if is_compensating(vocal, instrumental, mix):
        failures.append(
            _failure(
                "MIX_COMPENSATION",
                0.0,
                f"vocal {vocal['lag_sec']:+.3f}s instrumental {instrumental['lag_sec']:+.3f}s",
            )
        )
    report["failures"] = failures
    return report


def stamp_result(result, *, mix_verdict: str, mix_corr: float, mix_lag: float, mix_drift: float):
    """Store vocal, instrumental, and mix checks. Fail the folder when they compensate."""
    report = dict(getattr(result, "alignment_report", None) or {})
    vocal = relationship(
        str(getattr(result, "aca_verdict", "") or ""),
        float(getattr(result, "aca_corr", 0.0) or 0.0),
        float(getattr(result, "aca_lag_sec", 0.0) or 0.0),
        float(getattr(result, "aca_check_drift_ms", 0.0) or 0.0),
    )
    instrumental = relationship(
        str(getattr(result, "inst_verdict", "") or ""),
        float(getattr(result, "inst_corr", 0.0) or 0.0),
        float(getattr(result, "inst_lag_sec", 0.0) or 0.0),
        float(getattr(result, "inst_check_drift_ms", 0.0) or 0.0),
    )
    mix = relationship(mix_verdict, mix_corr, mix_lag, mix_drift)
    mix["source"] = "post_render_validation"
    if "vocal_confidence" in report:
        vocal["confidence"] = float(report.get("vocal_confidence") or 0.0)
    if "inst_confidence" in report:
        instrumental["confidence"] = float(report.get("inst_confidence") or 0.0)
    report = apply_relationships(report, vocal=vocal, instrumental=instrumental, mix=mix)
    result.alignment_report = report
    if any(item.get("code") == "MIX_COMPENSATION" for item in report.get("failures") or []):
        if getattr(result, "verdict", "") == "pass":
            result.verdict = "fail"
        notes = str(getattr(result, "notes", "") or "")
        if "MIX_COMPENSATION" not in notes:
            result.notes = notes + "; MIX_COMPENSATION"
    return result


def notes_fragment(report: dict) -> str:
    codes = [item.get("code", "") for item in report.get("failures") or []]
    shown = ",".join(code for code in codes if code) or "none"
    return (
        f"profile={report.get('profile', 'default')}; "
        f"offset={float(report.get('offset_sec') or 0.0):+.3f}s; "
        f"drift={float(report.get('drift_ms_per_min') or 0.0):+.1f}ms/min; "
        f"p95={float(report.get('p95_error_sec') or 0.0) * 1000:.0f}ms; "
        f"jumps={int(report.get('n_discontinuities') or 0)}; "
        f"conf={float(report.get('confidence') or 0.0):.2f}; "
        f"codes={shown}"
    )
