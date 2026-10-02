"""Master clock from the original mix.

Beat This! runs on the original only, in a child process so the aligner
never imports torch. Beats become a continuous musical phase. Onset and
vocal lags are smoothed along that phase, so a tempo change and a silent
gap share one clock. The acapella and the instrumental stem are not
beat-tracked.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SERVER: "_BeatServer | None" = None
_SERVER_LOCK = threading.Lock()


@dataclass
class BeatLock:
    times: np.ndarray
    lags: np.ndarray
    scores: np.ndarray | None
    pad_sec: float
    note: str
    locked: bool


@dataclass
class MasterClock:
    ok: bool
    beats: np.ndarray
    downbeats: np.ndarray
    period: float
    note: str


def beat_period(beats: np.ndarray) -> float:
    """Median beat interval, in seconds. 0.5 when the pulse is too thin to trust."""
    beats = _clean_beats(beats)
    if len(beats) < 4:
        return 0.5
    gaps = np.diff(beats)
    gaps = gaps[(gaps > 0.2) & (gaps < 2.0)]
    if len(gaps) < 3:
        return 0.5
    return float(np.median(gaps))


def _clean_beats(beats: np.ndarray) -> np.ndarray:
    beats = np.sort(np.asarray(beats, dtype=float))
    beats = beats[np.isfinite(beats)]
    if len(beats) < 2:
        return beats
    keep = np.ones(len(beats), dtype=bool)
    keep[1:] = np.diff(beats) > 1e-3
    return beats[keep]


# Same gates Mixxx uses when it irons Queen Mary beats onto a local grid.
# A single beat may sit 25 ms off. A slow drift that adds up past 100 ms
# is a real tempo change, so it starts a new region. A region shorter
# than 16 beats is left as detected.
_IRON_PHASE_SEC = 0.025
_IRON_PHASE_SUM_SEC = 0.1
_IRON_MAX_OUTLIERS = 1
_IRON_MIN_BEATS = 16


def iron_beats(beats: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace detector jitter with a piecewise-constant beat grid.

    Returns the ironed beat times and how many constant regions were long
    enough to iron. A steady 120 BPM track with ±10 ms of jitter becomes
    one grid. A tempo that walks away by 10 ms a beat does not.
    """
    beats = _clean_beats(beats)
    if len(beats) < _IRON_MIN_BEATS + 1:
        return beats, 0
    regions = _const_regions(beats)
    ironed = _beats_from_regions(regions)
    if len(ironed) < 4:
        return beats, 0
    ironed_count = sum(1 for _start, length, count in regions if count >= _IRON_MIN_BEATS and length > 0)
    return _clean_beats(ironed), int(ironed_count)


def _const_regions(beats: np.ndarray) -> list[tuple[float, float, int]]:
    """Spans of one beat length. The last entry marks the final beat."""
    n = len(beats)
    regions: list[tuple[float, float, int]] = []
    left = 0
    right = n - 1
    while left < n - 1:
        if right <= left:
            step = float(beats[left + 1] - beats[left]) if left + 1 < n else 0.0
            regions.append((float(beats[left]), step, 1))
            left += 1
            right = n - 1
            continue
        mean = float(beats[right] - beats[left]) / float(right - left)
        outliers = 0
        ironed = float(beats[left])
        phase_sum = 0.0
        i = left + 1
        steady = True
        while i <= right:
            ironed += mean
            phase_error = ironed - float(beats[i])
            phase_sum += phase_error
            if abs(phase_error) > _IRON_PHASE_SEC:
                outliers += 1
                if outliers > _IRON_MAX_OUTLIERS or i == left + 1:
                    steady = False
                    break
            if abs(phase_sum) > _IRON_PHASE_SUM_SEC:
                steady = False
                break
            i += 1
        if steady and i > right:
            border = 0.0
            if right > left + 2:
                first = float(beats[left + 1] - beats[left])
                last = float(beats[right] - beats[right - 1])
                border = abs(first + last - 2.0 * mean)
            if border < _IRON_PHASE_SEC / 2.0:
                count = right - left
                if count >= _IRON_MIN_BEATS:
                    regions.append((float(beats[left]), mean, count))
                else:
                    for k in range(left, right):
                        regions.append((float(beats[k]), float(beats[k + 1] - beats[k]), 1))
                left = right
                right = n - 1
                continue
        right -= 1
    regions.append((float(beats[-1]), 0.0, 0))
    return regions


def _beats_from_regions(regions: list[tuple[float, float, int]]) -> np.ndarray:
    times: list[float] = []
    for index in range(len(regions) - 1):
        beat, length, _count = regions[index]
        end = regions[index + 1][0]
        if length <= 0:
            continue
        while beat < end - 0.002:
            times.append(beat)
            beat += length
    if regions:
        times.append(regions[-1][0])
    return np.asarray(times, dtype=float)


def phase_at(times: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Unwrapped beat index. Beat i has phase i. Between beats, phase is linear.

    Past the last beat the slope holds, so a silent ending still advances
    in musical time.
    """
    beats = _clean_beats(beats)
    times = np.asarray(times, dtype=float)
    phase = np.arange(len(beats), dtype=float)
    if len(beats) < 2:
        return np.zeros(np.shape(times), dtype=float)
    out = np.interp(times, beats, phase)
    left = times < beats[0]
    right = times > beats[-1]
    slope_l = (phase[1] - phase[0]) / (beats[1] - beats[0])
    slope_r = (phase[-1] - phase[-2]) / (beats[-1] - beats[-2])
    out = np.asarray(out, dtype=float).copy()
    out[left] = phase[0] + (times[left] - beats[0]) * slope_l
    out[right] = phase[-1] + (times[right] - beats[-1]) * slope_r
    return out


def local_period(times: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Smoothed beat length at each time, in seconds."""
    beats = _clean_beats(beats)
    times = np.asarray(times, dtype=float)
    if len(beats) < 4:
        return np.full(np.shape(times), 0.5, dtype=float)
    intervals = np.diff(beats)
    good = (intervals > 0.2) & (intervals < 2.0)
    fallback = float(np.median(intervals[good])) if np.any(good) else 0.5
    intervals = np.where(good, intervals, fallback)
    smooth = np.empty_like(intervals)
    for i in range(len(intervals)):
        lo, hi = max(0, i - 4), min(len(intervals), i + 5)
        smooth[i] = float(np.median(intervals[lo:hi]))
    mids = 0.5 * (beats[:-1] + beats[1:])
    return np.interp(times, mids, smooth, left=float(smooth[0]), right=float(smooth[-1]))


def phase_smooth_lags(
    times: np.ndarray,
    lags: np.ndarray,
    scores: np.ndarray | None,
    beats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Smooth a lag curve along musical phase, and fill the holes.

    A window that jumps by more than about a third of a beat is replaced
    by the phase-local median. Times with no measurement, including a
    vocal gap, take their lag from that same phase curve. Samples are
    written every 2 s so the later time interpolation stays close to the
    phase interpolation.
    """
    times = np.asarray(times, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if scores is None or len(scores) != len(lags):
        scores = np.ones(len(lags), dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)
    order = np.argsort(times)
    times, lags, scores = times[order], lags[order], scores[order]
    if len(times) < 4 or len(_clean_beats(beats)) < 4:
        return times, lags, scores, 0

    phi = phase_at(times, beats)
    period = local_period(times, beats)
    trusted = scores >= 0.35
    if int(np.sum(trusted)) < 4:
        trusted = np.ones(len(scores), dtype=bool)
    phi_t = phi[trusted]
    lag_t = lags[trusted]
    pred = np.empty(len(times), dtype=float)
    for i, p in enumerate(phi):
        near = np.abs(phi_t - p) <= 12.0
        pred[i] = float(np.median(lag_t[near])) if np.any(near) else float(lag_t[np.argmin(np.abs(phi_t - p))])
    gate = np.maximum(0.08, 0.35 * period)
    corrected = lags.copy()
    corrected_scores = scores.copy()
    replaced = 0
    for i in range(len(corrected)):
        if (not trusted[i]) or abs(float(corrected[i]) - pred[i]) > float(gate[i]):
            if abs(float(corrected[i]) - pred[i]) > 1e-4:
                replaced += 1
            corrected[i] = pred[i]
            corrected_scores[i] = 1.0

    grid = np.arange(float(times[0]), float(times[-1]) + 1e-6, 2.0)
    all_t = np.unique(np.concatenate([times, grid]))
    phi_src = phase_at(times, beats)
    phi_dst = phase_at(all_t, beats)
    src_order = np.argsort(phi_src)
    filled = np.interp(phi_dst, phi_src[src_order], corrected[src_order])
    filled_scores = np.full(len(all_t), 0.8, dtype=float)
    for i, t in enumerate(times):
        j = int(np.argmin(np.abs(all_t - t)))
        if abs(float(all_t[j]) - float(t)) <= 1e-3:
            filled[j] = corrected[i]
            filled_scores[j] = corrected_scores[i]
    return all_t, filled, filled_scores, replaced


def master_clock(ref_path: Path, *, ref_label: str = "original") -> MasterClock:
    """Beats, downbeats, and tempo of the original. Nothing else is tracked."""
    try:
        beats, downbeats = track_file(ref_path)
    except Exception as exc:  # noqa: BLE001 — a beat failure must not abort the song
        return MasterClock(False, np.zeros(0), np.zeros(0), 0.5, f"beat_skip={type(exc).__name__}")
    beats, regions = iron_beats(_clean_beats(beats))
    downbeats = _clean_beats(downbeats)
    if len(beats) < 4:
        return MasterClock(False, beats, downbeats, 0.5, "beat_skip=few_beats")
    period = beat_period(beats)
    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
    spread = float(np.std(intervals) / period) if len(intervals) > 3 else 0.0
    note = (
        f"beat_ref={ref_label}; beat_side=master; beat_period={period:.3f}s; "
        f"tempo_bpm={60.0 / period:.1f}; phase=master; beat_regions={regions}"
    )
    if spread >= 0.03:
        note += f"; tempo_spread={spread:.3f}"
    if len(downbeats) >= 4:
        note += f"; downbeats={len(downbeats)}"
    return MasterClock(True, beats, downbeats, period, note)


def phase_lock_lags(
    clock: MasterClock,
    times: np.ndarray,
    lags: np.ndarray,
    scores: np.ndarray | None,
) -> BeatLock:
    """Project one measured lag curve onto the master phase."""
    times = np.asarray(times, dtype=float)
    lags = np.asarray(lags, dtype=float)
    if not clock.ok or len(lags) < 4:
        return BeatLock(times, lags, scores, 0.0, clock.note, False)
    new_t, new_lags, new_scores, replaced = phase_smooth_lags(times, lags, scores, clock.beats)
    from warp_align_fail_all import absorb_lag_offset

    new_lags, bias = absorb_lag_offset(new_t, new_lags, new_scores)
    note = f"{clock.note}; beat_replaced={replaced}"
    if abs(bias) >= 1e-4:
        note += f"; beat_pad={bias:+.3f}s"
    return BeatLock(new_t, new_lags, new_scores, float(bias), note, True)


def track_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Beats and downbeats for one file, from disk cache or the tracker process."""
    path = Path(path)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    with _SERVER_LOCK:
        server = _server()
        assert server.proc.stdin is not None
        server.proc.stdin.write(json.dumps({"path": str(path)}) + "\n")
        server.proc.stdin.flush()
        payload = _read_json(server, timeout=300.0)
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "beat tracker failed"))
    beats = np.asarray(payload.get("beats") or [], dtype=float)
    downbeats = np.asarray(payload.get("downbeats") or [], dtype=float)
    _write_cache(path, beats, downbeats)
    return beats, downbeats


def _cache_file(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(
        f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()
    ).hexdigest()
    root = Path(os.environ.get("APPDATA", ".")) / "Audio Aligner" / "beat_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.json"


def _read_cache(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        data = json.loads(_cache_file(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    beats = np.asarray(data.get("beats") or [], dtype=float)
    downbeats = np.asarray(data.get("downbeats") or [], dtype=float)
    if len(beats) < 4 or len(downbeats) < 4:
        return None
    return beats, downbeats


def _write_cache(path: Path, beats: np.ndarray, downbeats: np.ndarray) -> None:
    _cache_file(path).write_text(
        json.dumps(
            {
                "beats": [float(x) for x in np.asarray(beats).reshape(-1)],
                "downbeats": [float(x) for x in np.asarray(downbeats).reshape(-1)],
            }
        ),
        encoding="utf-8",
    )


class _BeatServer:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        self.errors: deque[str] = deque(maxlen=8)
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._wait_ready(300.0)

    def _pump_stdout(self) -> None:
        if self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            self.lines.put(line)

    def _pump_stderr(self) -> None:
        if self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            text = line.strip()
            if text:
                self.errors.append(text)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.lines.get(timeout=1.0)
            except queue.Empty:
                if self.proc.poll() is not None:
                    detail = " | ".join(self.errors)[-400:]
                    raise RuntimeError(detail or "beat tracker exited before it was ready")
                continue
            if line.strip() == "READY":
                return
        raise TimeoutError("beat tracker did not start")

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.write(json.dumps({"quit": True}) + "\n")
                self.proc.stdin.flush()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.terminate()


def _server() -> _BeatServer:
    global _SERVER
    if _SERVER is None or _SERVER.proc.poll() is not None:
        _SERVER = _BeatServer()
    return _SERVER


def _read_json(server: _BeatServer, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = server.lines.get(timeout=1.0)
        except queue.Empty:
            if server.proc.poll() is not None:
                raise RuntimeError("beat tracker exited")
            continue
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise TimeoutError("beat tracker timed out")


def _shutdown() -> None:
    global _SERVER
    server = _SERVER
    _SERVER = None
    if server is not None:
        server.close()


atexit.register(_shutdown)


def _ensure_checkpoint() -> None:
    """Put final0.ckpt in the torch hub cache.

    The official host's TLS certificate is expired, so the stock downloader
    fails. A verified download is tried first. If that fails, the same URL
    is fetched once without certificate checks.
    """
    import ssl
    import urllib.request

    import torch

    dest = Path(torch.hub.get_dir()) / "checkpoints" / "beat_this-final0.ckpt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 50_000_000:
        return
    if dest.is_file():
        dest.unlink()
    url = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
    tmp = dest.with_suffix(".ckpt.partial")
    try:
        try:
            _download(url, tmp, context=None)
        except Exception:
            _download(url, tmp, context=ssl._create_unverified_context())
        tmp.replace(dest)
    finally:
        if tmp.is_file():
            tmp.unlink()
    if not dest.is_file() or dest.stat().st_size < 50_000_000:
        raise RuntimeError("could not download the beat_this checkpoint")


def _download(url: str, dest: Path, context) -> None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "align-checker"})
    with urllib.request.urlopen(req, context=context) as src, dest.open("wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def _serve() -> None:
    import torch
    from beat_this.inference import File2Beats

    _ensure_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    print("READY", flush=True)
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            request = json.loads(text)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "bad request"}), flush=True)
            continue
        if request.get("quit"):
            break
        path = str(request.get("path") or "")
        try:
            beats, downbeats = tracker(path)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "beats": [float(x) for x in np.asarray(beats).reshape(-1)],
                        "downbeats": [float(x) for x in np.asarray(downbeats).reshape(-1)],
                    }
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — report it, keep the server up
            print(
                json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
                flush=True,
            )


if __name__ == "__main__":
    if "--serve" in sys.argv:
        _serve()
