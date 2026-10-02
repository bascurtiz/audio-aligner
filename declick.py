"""Acapella de-click.

RX 11 De-click when it is installed and the engine can host it.
Otherwise the acapella is left as the stretch wrote it. The instrumental
is never de-clicked: a blind scan treats drum attacks as clicks.
"""
from __future__ import annotations

import functools
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Stretch-marker clicks are a few milliseconds wide. FFmpeg's default
# window is 55 ms and its default threshold of 2 marks a lot of audio as
# clicks. A shorter window and a higher threshold keep sung consonants.
ADECLICK_FILTER = "adeclick=window=20:overlap=75:arorder=2:threshold=8:burst=2:method=add"

# Declicker defaults, from its editor: smoothness 3 becomes a blend of
# 1/3, max sample step 0.12, max change of that step 0.2. The reference
# rate is 48 kHz, which is where those slider values were defined.
_DECLICKER_REF_SR = 48000.0
_DECLICKER_RATIO = 1.0 / 3.0
_DECLICKER_MAX_DELTA = 0.12
_DECLICKER_MAX_2ND = 0.2


@dataclass(frozen=True)
class DeclickPlan:
    """What will actually run.

    method is "rx" or "off". requested is the menu choice.
    note is the machine-readable tag stored on the song. summary is the
    run log line.
    """

    method: str
    requested: str
    note: str
    summary: str


def _ffmpeg_has_adeclick(exe: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(exe), "-hide_banner", "-h", "filter=adeclick"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    text = f"{proc.stdout}\n{proc.stderr}"
    return proc.returncode == 0 and "Filter adeclick" in text


def _ffmpeg_candidates() -> list[Path]:
    seen: set[str] = set()
    found: list[Path] = []

    def add(path: Path) -> None:
        key = str(path).casefold()
        if key in seen:
            return
        seen.add(key)
        if path.is_file():
            found.append(path)

    env = os.environ.get("FFMPEG_EXE")
    if env:
        add(Path(env))
    which = shutil.which("ffmpeg")
    if which:
        add(Path(which))
    try:
        listed = subprocess.run(
            ["where.exe", "ffmpeg"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        listed = None
    if listed is not None and listed.returncode == 0:
        for line in listed.stdout.splitlines():
            line = line.strip().strip('"')
            if line:
                add(Path(line))
    dev = Path(r"C:\dev")
    if dev.is_dir():
        for path in sorted(dev.glob("ffmpeg*/bin/ffmpeg.exe")):
            add(path)
    program = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    add(program / "ffmpeg" / "bin" / "ffmpeg.exe")
    add(Path(r"C:\ffmpeg\bin\ffmpeg.exe"))
    return found


@functools.lru_cache(maxsize=1)
def find_ffmpeg() -> Path | None:
    """First ffmpeg whose build actually includes the adeclick filter.

    The copy on PATH may be a stripped audio build that does not.
    """
    for path in _ffmpeg_candidates():
        if _ffmpeg_has_adeclick(path):
            return path
    return None


def _vst3_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("CommonProgramFiles", "CommonProgramFiles(x86)", "ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(key)
        if not base:
            continue
        root = Path(base)
        roots.append(root / "VST3")
        roots.append(root / "Common Files" / "VST3")
    appdata = os.environ.get("APPDATA")
    if appdata:
        roots.append(Path(appdata) / "VST3")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "Programs" / "Common" / "VST3")
    return roots


def _reaper_plugin_cache_has_rx() -> bool:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return False
    folder = Path(appdata) / "REAPER"
    for name in ("reaper-vstplugins64.ini", "reaper-vstplugins.ini", "reaper-vstplugins_arm64.ini"):
        path = folder / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "RX 11 De-click" in text:
            return True
    return False


@functools.lru_cache(maxsize=1)
def find_rx11_declick() -> str | None:
    """A location string when RX 11 De-click is installed, else None."""
    needle = "rx 11 de-click"
    for root in _vst3_roots():
        if not root.is_dir():
            continue
        try:
            matches = root.rglob("*.vst3")
        except OSError:
            continue
        for path in matches:
            if needle in path.name.casefold():
                return str(path)
    if _reaper_plugin_cache_has_rx():
        return "REAPER plugin cache"
    return None


def _require_ffmpeg() -> Path:
    exe = find_ffmpeg()
    if exe is None:
        raise FileNotFoundError(
            "FFmpeg with the adeclick filter was not found. "
            "The ffmpeg on PATH may be a stripped build that does not include it."
        )
    return exe


@functools.lru_cache(maxsize=8)
def resolve_declick(choice: str, host_rx: bool = True) -> DeclickPlan:
    """Pick the de-clicker.

    choice "rx" uses RX 11 when this process can host it and the plugin is
    installed. Otherwise the vocal is left untouched. choice "off" always
    leaves it untouched. Older menu values are treated as "rx". host_rx
    is false for Rubber Band: that render has no plugin host.
    """
    requested = (choice or "rx").strip().lower()
    if requested == "off":
        return DeclickPlan("off", "off", "aca_declick=off", "Off")

    if not host_rx:
        return DeclickPlan(
            "off",
            "rx",
            "aca_declick=off; declick_fallback=rubberband",
            "Off (Rubber Band cannot host RX 11)",
        )
    if not find_rx11_declick():
        return DeclickPlan(
            "off",
            "rx",
            "aca_declick=off; declick_fallback=rx_missing",
            "Off (RX 11 De-click not found)",
        )
    return DeclickPlan(
        "rx",
        "rx",
        "aca_declick=rx11_fix_discontinuous",
        "RX 11 De-click",
    )


def _declicker_channel(x: np.ndarray, sr: int) -> np.ndarray:
    """One channel through Declicker's three stages, in the plugin's order."""
    factor = _DECLICKER_REF_SR / float(sr)
    ratio = _DECLICKER_RATIO
    max_delta = _DECLICKER_MAX_DELTA
    max_2nd = _DECLICKER_MAX_2ND
    n = len(x)
    y = np.empty(n, dtype=np.float64)

    last = 0.0
    for i in range(n):
        sample = float(x[i])
        delta = sample - last * factor
        last = last + ratio * delta
        if last > 1.0:
            last = 1.0
        elif last < -1.0:
            last = -1.0
        y[i] = last

    last = 0.0
    for i in range(n):
        sample = y[i]
        delta = sample - last * factor
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        last = last + delta
        if last > 1.0:
            last = 1.0
        elif last < -1.0:
            last = -1.0
        y[i] = last

    last = 0.0
    last_delta = 0.0
    for i in range(n):
        sample = y[i]
        delta = sample - last * factor
        second = delta - last_delta
        if second > max_2nd:
            second = max_2nd
        elif second < -max_2nd:
            second = -max_2nd
        new_delta = last_delta + second
        last = last + new_delta
        if last > 1.0:
            last = 1.0
        elif last < -1.0:
            last = -1.0
        last_delta = delta
        y[i] = last
    return y


def apply_declicker(path: Path) -> None:
    """Replace ``path`` with the same audio after Declicker.

    Length, sample rate, and channel count stay put. The smoother lowers
    the sung phrases as well as the clicks, so the loud-phrase height is
    put back to what it was. A single gain does that: a click that fell
    further than the phrase stays quieter.
    """
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr <= 0 or len(y) == 0:
        return
    out_audio = np.empty_like(y)
    for ch in range(y.shape[1]):
        out_audio[:, ch] = _declicker_channel(y[:, ch], sr).astype(np.float32)
    from demucs_vocals import _phrase_level

    before = _phrase_level(y)
    after = _phrase_level(out_audio)
    if after > 0.0 and before > 0.0:
        gain = float(np.clip(before / after, 1.0 / 8.0, 8.0))
        out_audio *= np.float32(gain)
        peak = float(np.max(np.abs(out_audio))) if out_audio.size else 0.0
        if peak > 0.99:
            out_audio *= np.float32(0.99 / peak)
    out = path.with_name(path.stem + "._declicker" + path.suffix)
    if out.exists():
        out.unlink()
    try:
        if path.suffix.lower() == ".flac":
            sf.write(str(out), out_audio, sr, format="FLAC")
        else:
            sf.write(str(out), out_audio, sr, subtype="FLOAT")
        out.replace(path)
    finally:
        out.unlink(missing_ok=True)


# De-Click Pro defaults (Rigaudio - DeClick_Pro.jsfx).
_RIGA_BS = 16384
_RIGA_THRESHOLD = 10.0
_RIGA_NOISE_FLOOR_DB = -70.0
_RIGA_MAX_CLICK = 64
_RIGA_PRE = 1
_RIGA_POST = 1
_RIGA_ADAPT_MS = 10.0
_RIGA_FADE = 128
_RIGA_LATENCY_PAD = 10


def _rigaudio_declick(y: np.ndarray, sr: int) -> np.ndarray:
    """De-Click Pro at its default sliders.

    Short spikes are interpolated. A level jump is faded out. Anything
    longer than the max click length is left as sung audio. The plugin
    delays by max-click plus 10 samples; that delay is removed here so
    the stem stays aligned.
    """
    n, nch = y.shape
    if n == 0 or nch == 0 or sr <= 0:
        return y
    nch = min(nch, 2)
    link = nch > 1
    bs = _RIGA_BS
    mult = _RIGA_THRESHOLD
    nfloor = 10.0 ** (_RIGA_NOISE_FLOOR_DB / 20.0)
    ml = _RIGA_MAX_CLICK
    pre = _RIGA_PRE
    post = _RIGA_POST
    fade_len = _RIGA_FADE
    coef = 1.0 - math.exp(-1.0 / (max(_RIGA_ADAPT_MS, 0.1) * 0.001 * sr))
    latency = ml + _RIGA_LATENCY_PAD
    work = np.zeros((nch, bs), dtype=np.float64)
    env = [0.0] * nch
    active = [0] * nch
    first = [0] * nch
    last_i = [0] * nch
    quiet = [0] * nch
    locked = [0] * nch
    x_hist = [[0.0, 0.0, 0.0] for _ in range(nch)]
    thr = [nfloor] * nch
    step_amp = [0.0] * nch
    step_jj = [0] * nch
    step_fw = [0] * nch
    flag = [False] * nch
    out = np.zeros((n, nch), dtype=np.float64)
    src = y[:, :nch]
    cos = math.cos
    pi = math.pi
    wp = 0

    for step in range(n + latency):
        for c in range(nch):
            x = float(src[step, c]) if step < n else 0.0
            jj = step_jj[c]
            fw = step_fw[c]
            corr = step_amp[c] * 0.5 * (1.0 + cos(pi * jj / fw)) if jj < fw and fw > 0 else 0.0
            if jj < fw:
                step_jj[c] = jj + 1
            work[c, wp] = x - corr
            x1, x2, x3 = x_hist[c]
            hd = x - 3.0 * x1 + 3.0 * x2 - x3
            ad = abs(hd)
            e = env[c]
            t = max(e * mult, nfloor)
            thr[c] = t
            flag[c] = ad > t
            env[c] = e + (min(ad, t) - e) * coef
            x_hist[c][0] = x
            x_hist[c][1] = x1
            x_hist[c][2] = x2

        for c in range(nch):
            flg = (flag[0] or flag[1]) if link else flag[c]
            if locked[c]:
                if not flg:
                    locked[c] = 0
                continue
            if not active[c]:
                if flg:
                    active[c] = 1
                    first[c] = wp
                    last_i[c] = wp
                    quiet[c] = 0
                continue
            if flg:
                last_i[c] = wp
                quiet[c] = 0
            else:
                quiet[c] += 1
            span = (wp - first[c] + bs) % bs
            if span > ml + 4:
                active[c] = 0
                locked[c] = 1
                continue
            if quiet[c] < 4:
                continue
            active[c] = 0
            lfl = (last_i[c] - first[c] + bs) % bs
            nbad = pre + max(0, lfl - 2) + post + 1
            s0 = (first[c] - pre + bs) % bs
            i0 = (s0 - 1 + bs) % bs
            im1 = (s0 - 2 + bs) % bs
            i1 = (s0 + nbad) % bs
            i2 = (s0 + nbad + 1) % bs
            p0 = float(work[c, i0])
            pm = float(work[c, im1])
            p1 = float(work[c, i1])
            p2 = float(work[c, i2])
            nn = nbad + 1
            sl0 = p0 - pm
            sl1 = p2 - p1
            jump = p1 - p0
            slope_expect = nn * 0.5 * (sl0 + sl1)
            st_a = jump - slope_expect
            # The plugin treats st_a as an edit-point jump. On a sung note
            # the slopes are large and st_a becomes about a full sample,
            # then the fade rails the vocal. A real jump is a level change
            # the two sides do not already slope toward.
            is_step = abs(jump) > 0.5 * thr[c] and abs(slope_expect) <= 0.5 * abs(jump)
            if is_step:
                cnt = (wp - i1 + bs) % bs + 1
                for j in range(cnt):
                    w = 0.5 * (1.0 + cos(pi * j / fade_len)) if j < fade_len else 0.0
                    idx = (i1 + j) % bs
                    work[c, idx] -= st_a * w
                jo = step_jj[c]
                fo = step_fw[c]
                ro = step_amp[c] * 0.5 * (1.0 + cos(pi * jo / fo)) if jo < fo and fo > 0 else 0.0
                if ro == 0.0:
                    step_amp[c] = st_a
                    step_jj[c] = cnt
                    step_fw[c] = fade_len
                else:
                    rn = st_a * 0.5 * (1.0 + cos(pi * cnt / fade_len)) if cnt < fade_len else 0.0
                    step_amp[c] = ro + rn
                    step_jj[c] = 0
                    step_fw[c] = fade_len
                p1 -= st_a
                p2 -= st_a
            m0 = sl0 * min(nn, 6)
            m1 = (p2 - p1) * min(nn, 6)
            for k in range(1, nbad + 1):
                t = k / nn
                t2 = t * t
                t3 = t2 * t
                yy = (
                    (2.0 * t3 - 3.0 * t2 + 1.0) * p0
                    + (t3 - 2.0 * t2 + t) * m0
                    + (-2.0 * t3 + 3.0 * t2) * p1
                    + (t3 - t2) * m1
                )
                # The cubic tangent can stick out past the two good samples
                # and that stick-out is a new click.
                lo = p0 if p0 < p1 else p1
                hi = p0 if p0 > p1 else p1
                if yy < lo:
                    yy = lo
                elif yy > hi:
                    yy = hi
                work[c, (s0 - 1 + k + bs) % bs] = yy

        src_i = step - latency
        if 0 <= src_i < n:
            rp = (wp - latency + bs) % bs
            for c in range(nch):
                out[src_i, c] = work[c, rp]
        wp = (wp + 1) % bs

    if y.shape[1] == nch:
        return out
    full = np.empty_like(y, dtype=np.float64)
    full[:, :nch] = out
    full[:, nch:] = y[:, nch:]
    return full


def apply_rigaudio_declick(path: Path) -> None:
    """Replace ``path`` with the same audio after De-Click Pro.

    Length, sample rate, and channel count stay put.
    """
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr <= 0 or len(y) == 0:
        return
    out_audio = _rigaudio_declick(y.astype(np.float64), sr).astype(np.float32)
    out = path.with_name(path.stem + "._rigaudio" + path.suffix)
    if out.exists():
        out.unlink()
    try:
        if path.suffix.lower() == ".flac":
            sf.write(str(out), out_audio, sr, format="FLAC")
        else:
            sf.write(str(out), out_audio, sr, subtype="FLOAT")
        out.replace(path)
    finally:
        out.unlink(missing_ok=True)


def apply_ffmpeg_adeclick(path: Path) -> None:
    """Replace ``path`` with the same audio after FFmpeg adeclick.

    Sample rate, channel count, and length stay put. A shorter or longer
    render is trimmed or padded so the stem still lines up with the original.
    """
    exe = _require_ffmpeg()
    info = sf.info(str(path))
    wav = path.with_name(path.stem + "._adeclick_tmp.wav")
    out = path.with_name(path.stem + "._adeclick_out" + path.suffix)
    if wav.exists():
        wav.unlink()
    if out.exists():
        out.unlink()
    cmd = [
        str(exe),
        "-hide_banner",
        "-nostats",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-map",
        "0:a:0",
        "-af",
        ADECLICK_FILTER,
        "-c:a",
        "pcm_f32le",
        str(wav),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not wav.is_file():
            lines = [line.strip() for line in (proc.stderr or "").splitlines() if line.strip()]
            tail = lines[-1] if lines else f"exit {proc.returncode}"
            raise RuntimeError(f"ffmpeg adeclick failed: {tail}")
        y, sr = sf.read(str(wav), always_2d=True, dtype="float32")
        if sr != info.samplerate:
            raise RuntimeError(
                f"ffmpeg adeclick changed the sample rate from {info.samplerate} to {sr}."
            )
        if y.shape[0] > info.frames:
            y = y[: info.frames]
        elif y.shape[0] < info.frames:
            y = np.pad(y, ((0, info.frames - y.shape[0]), (0, 0)))
        if y.shape[1] > info.channels:
            y = y[:, : info.channels]
        elif y.shape[1] < info.channels:
            y = np.pad(y, ((0, 0), (0, info.channels - y.shape[1])))
        if path.suffix.lower() == ".flac":
            sf.write(str(out), y, sr, format="FLAC")
        else:
            sf.write(str(out), y, sr, subtype="FLOAT")
        out.replace(path)
    finally:
        wav.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
