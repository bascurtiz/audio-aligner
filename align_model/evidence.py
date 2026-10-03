"""Multi-feature alignment evidence and confidence-weighted fusion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from align_model.params import FEATURE_NAMES, TrackProfile

HOP = 512
WAVE_SR = 4000
HIGH_MARGIN = 0.12
MIN_SEP_SEC = 0.15


@dataclass
class EvidencePoint:
    time: float
    lag: float
    confidence: float
    chroma: float = 0.0
    onset: float = 0.0
    spectral: float = 0.0
    waveform: float = 0.0
    beat_score: float = 0.0
    best_score: float = 0.0
    second_score: float = 0.0
    match_margin: float = 0.0
    low_signal: bool = False

    def feature_scores(self) -> dict[str, float]:
        return {
            "chroma": self.chroma,
            "onset": self.onset,
            "spectral": self.spectral,
            "waveform": self.waveform,
        }


@dataclass
class AlignmentEvidence:
    points: list[EvidencePoint] = field(default_factory=list)
    profile: str = "default"
    kind: str = "vocal"
    low_signal_frac: float = 0.0

    @property
    def times(self) -> np.ndarray:
        return np.asarray([p.time for p in self.points], dtype=float)

    @property
    def lags(self) -> np.ndarray:
        return np.asarray([p.lag for p in self.points], dtype=float)

    @property
    def scores(self) -> np.ndarray:
        return np.asarray([p.confidence for p in self.points], dtype=float)

    @property
    def margins(self) -> np.ndarray:
        return np.asarray([p.match_margin for p in self.points], dtype=float)


def best_and_second(
    values: np.ndarray,
    lag_sec: np.ndarray,
    *,
    min_sep_sec: float = MIN_SEP_SEC,
) -> tuple[float, float, float, float]:
    """Return best lag, best score, second lag, second score."""
    values = np.asarray(values, dtype=float).reshape(-1)
    lag_sec = np.asarray(lag_sec, dtype=float).reshape(-1)
    if values.size == 0 or lag_sec.size != values.size:
        return 0.0, 0.0, 0.0, 0.0
    i = int(np.argmax(values))
    best = float(values[i])
    best_lag = float(lag_sec[i])
    mask = np.abs(lag_sec - best_lag) >= float(min_sep_sec)
    if not np.any(mask):
        return best_lag, best, best_lag, 0.0
    masked = np.where(mask, values, -np.inf)
    j = int(np.argmax(masked))
    return best_lag, best, float(lag_sec[j]), float(values[j])


def context_weights(
    base: dict[str, float],
    *,
    chroma: float,
    onset: float,
    kind: str,
) -> dict[str, float]:
    """Vocal windows favor chroma and waveform. Transient windows favor onset."""
    weights = {name: float(base.get(name, 0.0)) for name in FEATURE_NAMES}
    transient = kind == "instrumental" or onset > chroma + 0.05
    if transient:
        weights["onset"] *= 1.5
        weights["waveform"] *= 1.15
        weights["chroma"] *= 0.55
    else:
        weights["chroma"] *= 1.45
        weights["waveform"] *= 1.2
        weights["onset"] *= 0.45
        weights["spectral"] *= 1.05
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        return weights
    return {name: max(0.0, value) / total for name, value in weights.items()}


def fuse_candidates(
    candidates: list[tuple[str, float, float, float, float]],
    weights: dict[str, float],
    *,
    cluster_sec: float = 0.04,
) -> tuple[float, float, float, float] | None:
    """Cluster feature lags. Return lag, confidence, second score, margin.

    Weights are renormalized over the features that scored. The margin is the
    winning cluster against the next cluster and against each feature's own
    second peak.
    """
    alive: list[tuple[str, float, float, float, float]] = []
    for name, lag, best, _second_lag, second in candidates:
        weight = float(weights.get(name, 0.0))
        if best <= 0.02 or weight <= 0.0:
            continue
        alive.append((name, float(lag), float(best), float(second), weight))
    if not alive:
        return None
    weight_sum = sum(item[4] for item in alive) or 1.0
    alive = [(name, lag, best, second, weight / weight_sum) for name, lag, best, second, weight in alive]
    alive.sort(key=lambda item: item[1])
    clusters: list[dict] = []
    for _name, lag, best, second, weight in alive:
        if clusters and abs(lag - clusters[-1]["lags"][-1]) <= cluster_sec:
            bucket = clusters[-1]
        else:
            bucket = {"lags": [], "weights": [], "raw": [], "second": []}
            clusters.append(bucket)
        bucket["lags"].append(lag)
        bucket["weights"].append(weight * best)
        bucket["raw"].append(best)
        bucket["second"].append(second)
    scored: list[tuple[float, float, float, float]] = []
    for bucket in clusters:
        w = np.maximum(np.asarray(bucket["weights"], dtype=float), 1e-6)
        lags = np.asarray(bucket["lags"], dtype=float)
        raw = np.asarray(bucket["raw"], dtype=float)
        second_peaks = np.asarray(bucket["second"], dtype=float)
        scored.append(
            (
                float(w.sum()),
                float(np.average(lags, weights=w)),
                float(np.average(raw, weights=w)),
                float(np.average(second_peaks, weights=w)),
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    _total, lag, confidence, internal_second = scored[0]
    other = scored[1][2] if len(scored) > 1 else 0.0
    second = max(float(other), float(internal_second))
    margin = max(0.0, float(confidence) - second)
    return float(lag), float(confidence), second, margin


def _frame_corr(
    ref: np.ndarray,
    qry: np.ndarray,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of per-row FFT correlations inside ±max_lag frames, energy-normalized."""
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
        qry = qry.reshape(1, -1)
    n = min(ref.shape[1], qry.shape[1])
    ref = ref[:, :n]
    qry = qry[:, :n]
    corr_sum = None
    energy_r = 0.0
    energy_q = 0.0
    for row in range(ref.shape[0]):
        rb = ref[row] - float(np.mean(ref[row]))
        qb = qry[row] - float(np.mean(qry[row]))
        energy_r += float(np.dot(rb, rb))
        energy_q += float(np.dot(qb, qb))
        corr = signal.correlate(rb, qb, mode="full", method="fft")
        corr_sum = corr if corr_sum is None else corr_sum + corr
    if corr_sum is None:
        return np.zeros(1), np.zeros(1)
    lags = np.arange(-qry.shape[1] + 1, ref.shape[1])
    valid = (lags >= -max_lag) & (lags <= max_lag)
    denom = float(np.sqrt(max(energy_r, 0.0) * max(energy_q, 0.0))) or 1e-12
    return lags[valid].astype(float), corr_sum[valid] / denom


def _wave_peaks(ref: np.ndarray, qry: np.ndarray, max_lag: int) -> tuple[float, float, float, float]:
    """Waveform NCC peaks. Lag is in samples of the decimated audio."""
    if ref.size < 8 or qry.size < 8:
        return 0.0, 0.0, 0.0, 0.0
    ref = np.asarray(ref, dtype=np.float64)
    qry = np.asarray(qry, dtype=np.float64)
    ref = ref - float(np.mean(ref))
    qry = qry - float(np.mean(qry))
    corr = signal.correlate(ref, qry, mode="full", method="fft")
    lags = np.arange(-len(qry) + 1, len(ref))
    valid = (lags >= -max_lag) & (lags <= max_lag)
    if not np.any(valid):
        return 0.0, 0.0, 0.0, 0.0
    denom = float(np.linalg.norm(ref) * np.linalg.norm(qry)) or 1e-12
    return best_and_second(
        corr[valid] / denom,
        lags[valid].astype(float),
        min_sep_sec=MIN_SEP_SEC * WAVE_SR,
    )


def _decimate(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr <= target_sr:
        return np.asarray(y, dtype=np.float64)
    step = max(1, int(round(sr / target_sr)))
    return np.asarray(y[::step], dtype=np.float64)


def extract_evidence(
    ref: np.ndarray,
    qry: np.ndarray,
    sr: int,
    *,
    profile: TrackProfile,
    kind: str = "vocal",
) -> AlignmentEvidence:
    """Sliding multi-feature match. Silent query windows are dropped."""
    import librosa

    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    qry = np.asarray(qry, dtype=np.float32).reshape(-1)
    n = min(len(ref), len(qry))
    ref = ref[:n]
    qry = qry[:n]
    evidence = AlignmentEvidence(profile=profile.name, kind=kind)
    if sr <= 0 or n < int(sr * 0.5):
        evidence.low_signal_frac = 1.0
        return evidence

    hop_sec = HOP / float(sr)
    chroma_r = librosa.feature.chroma_cqt(y=ref, sr=sr, hop_length=HOP)
    chroma_q = librosa.feature.chroma_cqt(y=qry, sr=sr, hop_length=HOP)
    chroma_r /= np.maximum(np.linalg.norm(chroma_r, axis=0, keepdims=True), 1e-9)
    chroma_q /= np.maximum(np.linalg.norm(chroma_q, axis=0, keepdims=True), 1e-9)
    onset_r = librosa.onset.onset_strength(y=ref, sr=sr, hop_length=HOP).astype(np.float64)
    onset_q = librosa.onset.onset_strength(y=qry, sr=sr, hop_length=HOP).astype(np.float64)
    mel_r = np.log1p(librosa.feature.melspectrogram(y=ref, sr=sr, n_mels=16, hop_length=HOP))
    mel_q = np.log1p(librosa.feature.melspectrogram(y=qry, sr=sr, n_mels=16, hop_length=HOP))
    rms = librosa.feature.rms(y=qry, hop_length=HOP)[0]
    frames = min(chroma_r.shape[1], chroma_q.shape[1], len(onset_r), len(onset_q), mel_r.shape[1], mel_q.shape[1], len(rms))
    chroma_r, chroma_q = chroma_r[:, :frames], chroma_q[:, :frames]
    onset_r, onset_q = onset_r[:frames], onset_q[:frames]
    mel_r, mel_q = mel_r[:, :frames], mel_q[:, :frames]
    rms = rms[:frames]
    rms_thr = max(float(np.percentile(rms, 60)) * 0.35, 1e-5) if rms.size else 1e-5

    wave_r = _decimate(ref, sr, WAVE_SR)
    wave_q = _decimate(qry, sr, WAVE_SR)
    wave_hop = max(1, int(round(HOP * WAVE_SR / sr)))

    win = max(8, int(profile.win_sec / hop_sec))
    step = max(1, int(profile.step_sec / hop_sec))
    max_lag = max(2, int(profile.max_lag_sec / hop_sec))
    considered = 0
    dropped = 0
    base = profile.weight_map()
    for start in range(0, max(1, frames - win), step):
        end = min(frames, start + win)
        considered += 1
        if float(np.mean(rms[start:end])) < rms_thr:
            dropped += 1
            continue
        c_frames, c_vals = _frame_corr(chroma_r[:, start:end], chroma_q[:, start:end], max_lag)
        c_lag, c_best, _c2_lag, c_second = best_and_second(c_vals, c_frames * hop_sec)
        o_frames, o_vals = _frame_corr(onset_r[start:end], onset_q[start:end], max_lag)
        o_lag, o_best, _o2, o_second = best_and_second(o_vals, o_frames * hop_sec)
        s_frames, s_vals = _frame_corr(mel_r[:, start:end], mel_q[:, start:end], max_lag)
        s_lag, s_best, _s2, s_second = best_and_second(s_vals, s_frames * hop_sec)
        w0 = int(start * wave_hop)
        w1 = int(end * wave_hop)
        wave_max = max(2, int(profile.max_lag_sec * WAVE_SR))
        w_lag, w_best, _w2, w_second = _wave_peaks(wave_r[w0:w1], wave_q[w0:w1], wave_max)
        w_lag = w_lag / float(WAVE_SR)
        candidates = [
            ("chroma", c_lag, c_best, 0.0, c_second),
            ("onset", o_lag, o_best, 0.0, o_second),
            ("spectral", s_lag, s_best, 0.0, s_second),
            ("waveform", w_lag, w_best, 0.0, w_second),
        ]
        weights = context_weights(base, chroma=c_best, onset=o_best, kind=kind)
        fused = fuse_candidates(candidates, weights)
        if fused is None:
            dropped += 1
            continue
        lag, confidence, second, margin = fused
        center = (start + (end - start) / 2.0) * hop_sec
        evidence.points.append(
            EvidencePoint(
                time=float(center),
                lag=float(lag),
                confidence=float(confidence),
                chroma=float(c_best),
                onset=float(o_best),
                spectral=float(s_best),
                waveform=float(w_best),
                best_score=float(confidence),
                second_score=float(second),
                match_margin=float(margin),
            )
        )
    evidence.low_signal_frac = float(dropped / considered) if considered else 1.0
    return evidence
