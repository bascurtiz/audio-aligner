#!/usr/bin/env python3
"""Drag acapella or instrumental sections, then write that layout.

Top lane is the Demucs vocal from the original (fixed).
Bottom lane is the processed stem, split into sections.
Drag a section sideways to move it on the song timeline.
Apply writes that layout, then re-processes that stem: élastique
against its Demucs split and loudness match. The other stem is left
as it is. Gap placement is not run again. The first apply keeps a copy
in _before_section_edit.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from PyQt6.QtCore import QEvent, QPoint, QSize, QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QCursor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from aca_gap_align import (
    HOP,
    SR_ANALYSIS,
    bandpass,
    find_active_segments,
    load_mono,
    merge_segments_across_loud_gaps,
    peak_norm,
    rms_env,
)
from check_alignment import checkpoint_sample_times, scan_folder
from window_chrome import (
    WIN_DEFAULT_H,
    WIN_DEFAULT_W,
    WIN_MIN_H,
    WIN_MIN_W,
    CustomTitleBar,
    apply_window_corner_preference,
    install_rounded_corner_watcher,
    prepare_dark_frameless_chrome,
)

try:
    import sounddevice as sd

    HAS_MEDIA = True
except ImportError:
    sd = None  # type: ignore[assignment]
    HAS_MEDIA = False

PEAK_HZ = 200
SEEK_STEP_SEC = 0.1
SEEK_FINE_SEC = 0.01
BG = "#1e1f26"
PANEL = "#262833"
FG = "#e6e8ef"
DIM = "#9aa0b4"
ACCENT = "#7c5cff"
DEMUCS = "#3d9e8f"
ACA = "#a855f7"
LANE_DEM = "#1a2424"
STRETCH_CUE = "#e6b15c"
SECTION_FILL = "#171920"
SECTION_EDGE = "#3a3d4a"
SECTION_DRAG_FILL = "#22242e"
SECTION_DRAG_EDGE = "#8d93a8"


class _EditorPlayback:
    """Mix on the PortAudio thread, the same way the stem player does.

    The UI only flips gains and swaps buffers. It never writes audio itself.
    """

    def __init__(self) -> None:
        self.sr = 0
        self.master = 0.85
        self.aca_gain = 1.0
        self.bed_kind = "vox"
        self.aca: np.ndarray | None = None
        self.aca_src: np.ndarray | None = None
        self.vox: np.ndarray | None = None
        self.inst: np.ndarray | None = None
        self.dinst: np.ndarray | None = None
        self._position = 0
        self._playing = False
        self._ended = False
        self._hold = False
        self.underruns = 0
        self._lock = threading.Lock()
        self._stream = None

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self._playing = playing
            if playing:
                self._ended = False

    def set_hold(self, hold: bool) -> None:
        with self._lock:
            self._hold = bool(hold)

    def is_held(self) -> bool:
        with self._lock:
            return self._hold

    def consume_ended(self) -> bool:
        with self._lock:
            ended = self._ended
            self._ended = False
            return ended

    def position_seconds(self) -> float:
        with self._lock:
            if not self.sr:
                return 0.0
            return self._position / self.sr

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._position = int(max(0.0, seconds) * self.sr) if self.sr else 0
            self._ended = False

    def set_master(self, value: float) -> None:
        with self._lock:
            self.master = value

    def set_gains(self, aca_gain: float, bed_kind: str) -> None:
        with self._lock:
            self.aca_gain = aca_gain
            self.bed_kind = bed_kind

    def set_aca(self, aca: np.ndarray | None) -> None:
        with self._lock:
            self.aca = aca

    def set_bed(self, kind: str, audio: np.ndarray | None) -> None:
        with self._lock:
            if kind == "vox":
                self.vox = audio
            elif kind == "inst":
                self.inst = audio
            elif kind == "dinst":
                self.dinst = audio
            elif kind == "aca":
                self.aca_src = audio

    def clear_buffers(self) -> None:
        with self._lock:
            self.aca = None
            self.aca_src = None
            self.vox = None
            self.inst = None
            self.dinst = None
            self._position = 0
            self._playing = False

    def callback(self, outdata, frames: int, _time, status) -> None:
        if status:
            self.underruns += 1
        with self._lock:
            pos = self._position
            playing = self._playing and not self._hold
            master = self.master
            aca_g = self.aca_gain
            kind = self.bed_kind
            aca = self.aca
            bed = (
                self.vox
                if kind == "vox"
                else self.inst
                if kind == "inst"
                else self.dinst
                if kind == "dinst"
                else self.aca_src
                if kind == "aca"
                else None
            )

        out = np.zeros((frames, 2), dtype=np.float32)
        n_end = len(aca) if aca is not None else 0
        if bed is not None:
            n_end = max(n_end, len(bed))
        if playing:
            if aca is not None and aca_g > 0.0 and pos < len(aca):
                take = min(frames, len(aca) - pos)
                out[:take] += aca[pos : pos + take] * (aca_g * master)
            if bed is not None and pos < len(bed):
                take = min(frames, len(bed) - pos)
                out[:take] += bed[pos : pos + take] * master
            np.clip(out, -1.0, 1.0, out=out)
            new_pos = pos + frames
            with self._lock:
                if self._position == pos:
                    self._position = new_pos
                if n_end and self._position >= n_end:
                    self._playing = False
                    self._position = n_end
                    self._ended = True
        outdata[:] = out

    def ensure_stream(self, sr: int) -> None:
        if sd is None:
            return
        if self._stream is not None and self.sr == sr:
            return
        self.close()
        self.sr = int(sr)
        self._stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype="float32",
            callback=self.callback,
            blocksize=1024,
        )
        self._stream.start()

    def flush(self) -> None:
        """Drop audio already queued, so the next sound starts at the playhead."""
        stream = self._stream
        if stream is None:
            return
        try:
            stream.abort()
            stream.start()
        except Exception:
            return

    def close(self) -> None:
        with self._lock:
            self._playing = False
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def _dim_wave(hex_color: str, toward: str, *, dim: bool) -> QColor:
    """Match the stem player: muted waves blend toward the lane background."""
    color = QColor(hex_color)
    if not dim:
        return color
    bg = QColor(toward)
    t = 0.72
    return QColor(
        int(color.red() * (1.0 - t) + bg.red() * t),
        int(color.green() * (1.0 - t) + bg.green() * t),
        int(color.blue() * (1.0 - t) + bg.blue() * t),
    )
CHIP_BG = "#2F3140"
BORDER = "#3a3d4d"
MUTE = "#e25c5c"
GUTTER = 76

_OPEN: list[QWidget] = []


def _sm_style(*, active: bool, danger: bool = False) -> str:
    if active:
        bg = MUTE if danger else ACCENT
        fg = FG
        hover = bg
    else:
        bg = CHIP_BG
        fg = DIM
        hover = "#36384A"
    return (
        f"QToolButton {{ background-color: {bg}; color: {fg}; border: 1px solid {BORDER}; "
        f"border-radius: 4px; font-weight: 700; font-size: 11px; padding: 0px; }}"
        f"QToolButton:hover {{ background-color: {hover}; }}"
        f"QToolButton:checked {{ background-color: {bg}; color: {fg}; }}"
    )


def _shortcut_key_chip(text: str, parent: QWidget) -> QLabel:
    chip = QLabel(text, parent)
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    chip.setStyleSheet(
        f"QLabel {{ background: {CHIP_BG}; color: {FG}; border: 1px solid {BORDER}; "
        f"border-radius: 3px; padding: 1px 6px; font-family: Consolas, 'Segoe UI'; font-size: 9pt; }}"
    )
    chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return chip


def _shortcut_group(keys: tuple[str, ...], label: str, *, join: str = "gap") -> QWidget:
    cell = QWidget()
    row = QHBoxLayout(cell)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    for i, key in enumerate(keys):
        if i > 0:
            if join == "plus":
                sep = QLabel("+", cell)
                sep.setStyleSheet(f"color: {DIM}; font-size: 9pt; background: transparent;")
                row.addWidget(sep)
            else:
                row.addSpacing(6)
        row.addWidget(_shortcut_key_chip(key, cell))
    action = QLabel(label, cell)
    action.setStyleSheet(f"color: {DIM}; font-size: 9pt; background: transparent;")
    row.addWidget(action)
    return cell


_ICON_SIZE = 14
_ICON_CACHE: dict[str, QIcon] = {}


def _format_time_ms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    mins, secs = divmod(total_s, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}:{ms:03d}"
    return f"{mins:02d}:{secs:02d}:{ms:03d}"


def _media_icon(kind: str) -> QIcon:
    cached = _ICON_CACHE.get(kind)
    if cached is not None:
        return cached
    size = _ICON_SIZE
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(FG))
    margin = max(2, size // 6)
    if kind == "pause":
        bar_w = max(2, size // 5)
        gap = max(2, size // 5)
        total = bar_w * 2 + gap
        x0 = (size - total) // 2
        y0 = max(2, size // 5)
        h = size - 2 * y0
        painter.drawRect(x0, y0, bar_w, h)
        painter.drawRect(x0 + bar_w + gap, y0, bar_w, h)
    elif kind == "stop":
        side = max(4, size - 2 * margin - 2)
        x0 = (size - side) // 2
        painter.drawRect(x0, x0, side, side)
    elif kind == "prev":
        bar_w = max(2, size // 5)
        gap = max(2, size // 7)
        tri_w = max(4, size - margin - bar_w - gap)
        x0 = margin + bar_w + gap
        painter.drawRect(margin, margin, bar_w, size - 2 * margin)
        painter.drawPolygon(
            QPolygon(
                [
                    QPoint(x0 + tri_w, margin),
                    QPoint(x0, size // 2),
                    QPoint(x0 + tri_w, size - margin),
                ]
            )
        )
    elif kind == "next":
        bar_w = max(2, size // 5)
        gap = max(2, size // 7)
        tri_w = max(4, size - margin - bar_w - gap)
        x0 = margin
        painter.drawPolygon(
            QPolygon(
                [
                    QPoint(x0, margin),
                    QPoint(x0 + tri_w, size // 2),
                    QPoint(x0, size - margin),
                ]
            )
        )
        painter.drawRect(x0 + tri_w + gap, margin, bar_w, size - 2 * margin)
    else:
        painter.drawPolygon(
            QPolygon(
                [
                    QPoint(margin, margin),
                    QPoint(size - margin, size // 2),
                    QPoint(margin, size - margin),
                ]
            )
        )
    painter.end()
    icon = QIcon(pix)
    _ICON_CACHE[kind] = icon
    return icon


def _transport_button(parent: QWidget, kind: str, tip: str, on_click) -> QToolButton:
    btn = QToolButton(parent)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedSize(32, 26)
    btn.setIcon(_media_icon(kind))
    btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
    btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
    btn.setAutoRaise(False)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setToolTip(tip)
    btn.setStyleSheet(
        f"QToolButton {{ background-color: {CHIP_BG}; border: 1px solid {BORDER}; "
        f"border-radius: 5px; padding: 0px; margin: 0px; }}"
        f"QToolButton:hover {{ background-color: #36384A; }}"
        f"QToolButton:pressed {{ background-color: #2A2C38; }}"
        f"QToolButton:disabled {{ background-color: {CHIP_BG}; border: 1px solid {BORDER}; }}"
    )
    btn.clicked.connect(on_click)
    return btn


@dataclass
class Section:
    src0: float
    src1: float
    dst0: float
    dst_dur: float | None = None

    @property
    def dur(self) -> float:
        return self.src1 - self.src0

    @property
    def out_dur(self) -> float:
        src = self.dur
        if self.dst_dur is None or self.dst_dur <= 0:
            return src
        return float(self.dst_dur)

    @property
    def dst1(self) -> float:
        return self.dst0 + self.out_dur

    def snap(self) -> tuple:
        return (
            round(self.src0, 6),
            round(self.src1, 6),
            round(self.dst0, 6),
            round(self.out_dur, 6),
        )


def section_from_snap(item: tuple) -> Section:
    src0, src1, dst0 = float(item[0]), float(item[1]), float(item[2])
    sec = Section(src0, src1, dst0)
    if len(item) >= 4:
        out = float(item[3])
        if abs(out - (src1 - src0)) > 1e-4:
            sec.dst_dur = out
    return sec


_MIN_SECTION_SEC = 0.4


def _quiet_cut(env: np.ndarray, src0: float, src1: float) -> float | None:
    """Split near the middle, on the quietest moment, leaving two draggable pieces."""
    dur = src1 - src0
    if dur < _MIN_SECTION_SEC * 2:
        return None
    margin = max(_MIN_SECTION_SEC, dur * 0.3)
    lo = src0 + margin
    hi = src1 - margin
    if hi <= lo + 0.02:
        return (src0 + src1) / 2.0
    hop_sec = HOP / SR_ANALYSIS
    i0 = max(0, int(lo / hop_sec))
    i1 = min(len(env), max(i0 + 1, int(hi / hop_sec)))
    if i0 >= len(env):
        return (src0 + src1) / 2.0
    rel = int(np.argmin(env[i0:i1]))
    cut = (i0 + rel) * hop_sec
    return min(max(cut, src0 + _MIN_SECTION_SEC), src1 - _MIN_SECTION_SEC)


def _more_sections(sections: list[Section], env: np.ndarray) -> list[Section] | None:
    """Split the longest phrases until there are about 1.5× as many."""
    if not sections:
        return None
    goal = (len(sections) * 3 + 1) // 2
    cur = list(sections)
    for _ in range(goal):
        if len(cur) >= goal:
            break
        order = sorted(range(len(cur)), key=lambda i: cur[i].dur, reverse=True)
        split_at = None
        cut = None
        for i in order:
            cut = _quiet_cut(env, cur[i].src0, cur[i].src1)
            if cut is not None:
                split_at = i
                break
        if split_at is None or cut is None:
            break
        sec = cur[split_at]
        frac = (cut - sec.src0) / max(1e-6, sec.dur)
        out_cut = sec.dst0 + frac * sec.out_dur
        left = Section(sec.src0, cut, sec.dst0)
        right = Section(cut, sec.src1, out_cut)
        if abs((out_cut - sec.dst0) - left.dur) > 1e-4:
            left.dst_dur = out_cut - sec.dst0
        if abs((sec.dst1 - out_cut) - right.dur) > 1e-4:
            right.dst_dur = sec.dst1 - out_cut
        cur[split_at : split_at + 1] = [left, right]
    if len(cur) == len(sections):
        return None
    return cur


def _fewer_sections(sections: list[Section]) -> list[Section] | None:
    """Join the closest phrases until there are about two thirds as many."""
    if len(sections) <= 1:
        return None
    goal = max(1, int(len(sections) / 1.5))
    if goal >= len(sections):
        goal = len(sections) - 1
    cur = sorted(sections, key=lambda s: (s.src0, s.src1))
    while len(cur) > goal:
        best = 0
        best_key = None
        for i in range(len(cur) - 1):
            gap = cur[i + 1].src0 - cur[i].src1
            key = (max(0.0, gap), cur[i].dur + cur[i + 1].dur)
            if best_key is None or key < best_key:
                best_key = key
                best = i
        a, b = cur[best], cur[best + 1]
        src0 = min(a.src0, b.src0)
        src1 = max(a.src1, b.src1)
        if a.src0 <= b.src0:
            dst = a.dst0
        else:
            dst = b.dst0
        joined = Section(src0, src1, dst)
        out = a.out_dur + b.out_dur
        if abs(out - joined.dur) > 1e-4:
            joined.dst_dur = out
        cur[best : best + 2] = [joined]
    return cur


def _peaks(y: np.ndarray, sr: int, hz: int = PEAK_HZ) -> np.ndarray:
    if y.ndim == 2:
        y = y.mean(axis=1)
    hop = max(1, int(sr / hz))
    n = len(y) // hop
    if n < 1:
        return np.zeros((1, 2), np.float32)
    sl = y[: n * hop].reshape(n, hop)
    return np.stack([sl.min(axis=1), sl.max(axis=1)], axis=1).astype(np.float32)


def _fade(chunk: np.ndarray, sr: int, fade_sec: float = 0.008) -> np.ndarray:
    n = len(chunk)
    fade = min(int(fade_sec * sr), n // 4)
    if fade < 2:
        return chunk
    out = chunk.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    if out.ndim == 1:
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    else:
        out[:fade] *= ramp[:, None]
        out[-fade:] *= ramp[::-1, None]
    return out


def _clock_cursor() -> QCursor:
    pix = QPixmap(22, 22)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QPen(QColor("#e6e8ef"), 1.4))
    painter.setBrush(QColor("#1e1f26"))
    painter.drawEllipse(1, 1, 18, 18)
    painter.drawLine(10, 10, 10, 5)
    painter.drawLine(10, 10, 14, 12)
    painter.end()
    return QCursor(pix, 10, 10)


_CLOCK = None


class PitchStretchError(RuntimeError):
    """Rubber Band could not change the length without changing the pitch."""


def _pad_or_trim(chunk: np.ndarray, n_out: int) -> np.ndarray:
    """Match a length by cutting or adding silence. The samples stay as they are."""
    if len(chunk) > n_out:
        return chunk[:n_out]
    if len(chunk) < n_out:
        return np.pad(chunk, ((0, n_out - len(chunk)), (0, 0)))
    return chunk


def _fit_chunk(chunk: np.ndarray, sr: int, n_out: int) -> np.ndarray:
    """Pitch-preserving fit of one section to a new length.

    Rubber Band ``--tempo`` changes the length and leaves the pitch alone.
    A missing Rubber Band raises instead of resampling, which would shift pitch.
    """
    if n_out < 8 or abs(len(chunk) - n_out) <= 2:
        return _pad_or_trim(chunk, n_out)
    rate = float(np.clip(len(chunk) / float(n_out), 0.5, 2.0))
    if abs(rate - 1.0) < 1e-4:
        return _pad_or_trim(chunk, n_out)
    try:
        from warp_align_fail_all import _ensure_rubberband_on_path

        _ensure_rubberband_on_path()
        import pyrubberband as pyrb

        audio = np.ascontiguousarray(chunk, dtype=np.float64)
        stretched = pyrb.time_stretch(audio, sr, rate)
        fitted = np.asarray(stretched, dtype=np.float32)
        if fitted.ndim == 1:
            fitted = fitted[:, None]
        if fitted.shape[1] != chunk.shape[1]:
            raise PitchStretchError("Rubber Band changed the channel count.")
    except PitchStretchError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface this instead of shifting pitch
        raise PitchStretchError(
            "Rubber Band could not stretch this section without changing the pitch."
        ) from exc
    return _pad_or_trim(fitted, n_out)


def render_moved(
    audio: np.ndarray,
    sr: int,
    sections: list[Section],
    n_frames: int,
    *,
    strict: bool = False,
    warnings: list[str] | None = None,
) -> np.ndarray:
    """Paste each section's source audio at its dragged start."""
    if audio.ndim == 1:
        audio = audio[:, None]
    channels = audio.shape[1]
    out = np.zeros((n_frames, channels), np.float32)
    for sec in sections:
        s0 = int(round(sec.src0 * sr))
        s1 = int(round(sec.src1 * sr))
        s0 = max(0, min(s0, len(audio)))
        s1 = max(s0, min(s1, len(audio)))
        chunk = _fade(audio[s0:s1], sr)
        if len(chunk) == 0:
            continue
        n_out = max(1, int(round(sec.out_dur * sr)))
        if abs(len(chunk) - n_out) > 2:
            try:
                chunk = _fit_chunk(chunk, sr, n_out)
            except PitchStretchError as exc:
                if strict:
                    raise
                if warnings is not None and not warnings:
                    warnings.append(
                        "Rubber Band is unavailable. Preview keeps the original pitch and length."
                    )
                chunk = _pad_or_trim(chunk, n_out)
                del exc
        d0 = int(round(sec.dst0 * sr))
        if d0 < 0:
            chunk = chunk[-d0:]
            d0 = 0
        d1 = min(n_frames, d0 + len(chunk))
        if d0 >= n_frames or d1 <= d0:
            continue
        out[d0:d1] += chunk[: d1 - d0]
    np.clip(out, -1.0, 1.0, out)
    return out


def _to_sr(y: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return y
    import librosa

    if y.ndim == 1:
        return librosa.resample(y, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)
    cols = [
        librosa.resample(y[:, c], orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)
        for c in range(y.shape[1])
    ]
    return np.stack(cols, axis=1)


class _ApplyWorker(QThread):
    step = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        folder: Path,
        aca_path: Path,
        audio: np.ndarray,
        sr: int,
        n_frames: int,
        sections: list[Section],
        declick: str = "rx",
        kind: str = "aca",
        tag_folder: bool = False,
    ) -> None:
        super().__init__()
        self.folder = folder
        self.aca_path = aca_path
        self.audio = audio
        self.sr = sr
        self.n_frames = n_frames
        self.sections = sections
        self.declick = declick
        self.kind = kind
        self.tag_folder = tag_folder

    def run(self) -> None:  # noqa: D102
        try:
            self.step.emit("write")
            out = render_moved(
                self.audio, self.sr, self.sections, self.n_frames, strict=True
            )
            tmp = self.aca_path.with_name(self.aca_path.stem + "._edit_tmp.flac")
            if tmp.exists():
                tmp.unlink()
            sf.write(str(tmp), out, self.sr, format="FLAC")
            tmp.replace(self.aca_path)

            from warp_align_reaper import (
                find_reaper,
                reprocess_edited_acapella,
                reprocess_edited_instrumental,
            )

            reaper_exe = find_reaper(None)
            if self.kind == "inst":
                payload = reprocess_edited_instrumental(
                    self.folder,
                    self.aca_path,
                    reaper_exe=reaper_exe,
                    on_step=self.step.emit,
                    tag_folder=self.tag_folder,
                )
            else:
                payload = reprocess_edited_acapella(
                    self.folder,
                    self.aca_path,
                    reaper_exe=reaper_exe,
                    declick=self.declick,
                    on_step=self.step.emit,
                    tag_folder=self.tag_folder,
                )
            self.finished_ok.emit(payload)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _Loader(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, folder: Path) -> None:
        super().__init__()
        self.folder = folder

    def run(self) -> None:  # noqa: D102
        try:
            self.loaded.emit(_load_song(self.folder))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def _audible_end_sec(y: np.ndarray, sr: int, thr: float = 0.008) -> float:
    """Last moment the waveform is still audible, plus a short release.

    Phrase detection stops on a bandpass dip. The vocal after that dip is
    still in the file, and the editor only draws inside a section, so the
    tail looked cut off.
    """
    mono = y.mean(axis=1) if y.ndim == 2 else y
    win = max(1, int(0.05 * sr))
    n = len(mono) // win
    if n < 1:
        return 0.0
    sl = mono[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(sl * sl, axis=1))
    idx = np.flatnonzero(rms >= thr)
    if len(idx) == 0:
        return 0.0
    last = (int(idx[-1]) + 1) * win / sr
    return min(len(mono) / sr, last + 0.12)


def _clock_marks(duration: float, spacing: float = 30.0) -> list[float]:
    """Output times of a fixed clock, including the start and not the end."""
    duration = float(duration)
    spacing = max(5.0, float(spacing))
    marks = [0.0]
    cursor = spacing
    while cursor < duration - spacing * 0.5:
        marks.append(float(cursor))
        cursor += spacing
    return marks


def _cue_clock(seconds: float) -> str:
    whole = max(0, int(round(float(seconds))))
    minutes, secs = divmod(whole, 60)
    return f"{minutes}:{secs:02d}"


def _step_marker_times(
    reference: np.ndarray,
    query: np.ndarray,
    file_sr: int,
    ruler: list[float],
) -> list[float]:
    """Where a re-align would plant a warp marker, on the 8-bar ruler."""
    if reference is None or query is None or len(ruler) < 1:
        return []
    from check_alignment import CHECKPOINT_CONFIRM_LAG_SEC, checkpoint_offsets
    from warp_align_fail_all import marks_where_lag_steps

    sr = 22050
    ref = reference.mean(axis=1) if reference.ndim == 2 else reference
    qry = query.mean(axis=1) if query.ndim == 2 else query
    ref = _to_sr(np.asarray(ref, dtype=np.float32), file_sr, sr)
    qry = _to_sr(np.asarray(qry, dtype=np.float32), file_sr, sr)
    starts = np.array([max(0.0, float(t) - 4.0) for t in ruler], dtype=float)
    points = checkpoint_offsets(
        ref, qry, sr, at_times=starts, confirm_sec=CHECKPOINT_CONFIRM_LAG_SEC
    )
    if len(points) < 2:
        return []
    times = np.array([p[0] for p in points], dtype=float)
    lags = np.array([p[1] for p in points], dtype=float)
    return marks_where_lag_steps(np.asarray(ruler, dtype=float), times, lags)


def _warp_cue_times(original: Path, duration: float) -> tuple[list[float], bool]:
    """Shared 8-bar ruler, and whether that bar line was used.

    A missing bar line puts the ticks on the 30 s clock.
    """
    marks, _note = checkpoint_sample_times(original)
    if marks is None:
        return _clock_marks(duration, 30.0), False
    shared = [0.0]
    for t in np.asarray(marks, dtype=float):
        t = float(t)
        if 1.0 < t < float(duration) - 1.0:
            shared.append(t)
    return shared, True


def _load_song(folder: Path) -> dict:
    aca_path, inst_path, orig_path, notes = scan_folder(folder)
    if not aca_path or not orig_path:
        raise RuntimeError(f"Need an acapella and an original.\n{notes}")
    orig_info = sf.info(str(orig_path))
    sr = int(orig_info.samplerate)
    n_frames = int(orig_info.frames)
    duration = n_frames / sr

    aca, aca_sr = sf.read(str(aca_path), always_2d=True, dtype="float32")
    aca = _to_sr(aca, aca_sr, sr)
    if len(aca) < n_frames:
        aca = np.pad(aca, ((0, n_frames - len(aca)), (0, 0)))
    else:
        aca = aca[:n_frames]

    inst = None
    if inst_path:
        inst_y, inst_sr = sf.read(str(inst_path), always_2d=True, dtype="float32")
        inst = _to_sr(inst_y, inst_sr, sr)
        if len(inst) < n_frames:
            inst = np.pad(inst, ((0, n_frames - len(inst)), (0, 0)))
        else:
            inst = inst[:n_frames]

    from demucs_vocals import ensure_demucs_stems_isolated

    vpath, dinst_path = ensure_demucs_stems_isolated(orig_path, folder=folder)
    vox, vox_sr = sf.read(str(vpath), always_2d=False, dtype="float32")
    if vox.ndim == 2:
        vox = vox.mean(axis=1).astype(np.float32)
    vox = _to_sr(vox, vox_sr, sr)
    if len(vox) < n_frames:
        vox = np.pad(vox, (0, n_frames - len(vox)))
    else:
        vox = vox[:n_frames]

    dinst, dinst_sr = sf.read(str(dinst_path), always_2d=False, dtype="float32")
    if dinst.ndim == 2:
        dinst = dinst.mean(axis=1).astype(np.float32)
    dinst = _to_sr(dinst, dinst_sr, sr)
    if len(dinst) < n_frames:
        dinst = np.pad(dinst, (0, n_frames - len(dinst)))
    else:
        dinst = dinst[:n_frames]

    aca_m = load_mono(aca_path, SR_ANALYSIS)
    aca_v = peak_norm(bandpass(aca_m, SR_ANALYSIS))
    segs = find_active_segments(
        aca_v,
        sr=SR_ANALYSIS,
        min_silence_sec=0.45,
        min_active_sec=0.7,
        silence_db=-48.0,
    )
    segs = merge_segments_across_loud_gaps(aca_m, segs, sr=SR_ANALYSIS)
    if not segs:
        segs = [(0.0, duration)]
    sections = [Section(a, b, a) for a, b in segs if b - a >= 0.15]
    if sections:
        heard = _audible_end_sec(aca, sr)
        tail = sections[-1]
        if heard > tail.src1 + 0.02:
            sections[-1] = Section(tail.src0, min(heard, duration), tail.dst0)

    inst_sections: list[Section] = []
    inst_v = None
    if inst is not None:
        inst_m = inst.mean(axis=1).astype(np.float32)
        inst_analysis = _to_sr(inst_m, sr, SR_ANALYSIS)
        inst_v = peak_norm(inst_analysis)
        inst_segs = find_active_segments(
            inst_v,
            sr=SR_ANALYSIS,
            min_silence_sec=0.45,
            min_active_sec=0.7,
            silence_db=-48.0,
        )
        if not inst_segs:
            inst_segs = [(0.0, duration)]
        inst_sections = [Section(a, b, a) for a, b in inst_segs if b - a >= 0.15]
        if inst_sections:
            heard = _audible_end_sec(inst, sr)
            tail = inst_sections[-1]
            if heard > tail.src1 + 0.02:
                inst_sections[-1] = Section(tail.src0, min(heard, duration), tail.dst0)

    bar_times, cue_on_bars = _warp_cue_times(orig_path, duration)
    aca_markers = _step_marker_times(vox, aca, sr, bar_times)
    inst_markers = _step_marker_times(dinst, inst, sr, bar_times) if inst is not None else []
    return {
        "folder": folder,
        "aca_path": aca_path,
        "sr": sr,
        "n_frames": n_frames,
        "duration": duration,
        "aca": aca,
        "inst_path": inst_path,
        "inst": inst,
        "inst_sections": inst_sections,
        "inst_v": inst_v,
        "vox": vox,
        "demucs_inst": dinst,
        "aca_peaks": _peaks(aca, sr),
        "vox_peaks": _peaks(vox, sr),
        "inst_peaks": _peaks(inst, sr) if inst is not None else None,
        "demucs_inst_peaks": _peaks(dinst, sr),
        "aca_v": aca_v,
        "sections": sections,
        "bar_times": bar_times,
        "cue_on_bars": cue_on_bars,
        "aca_markers": aca_markers,
        "inst_markers": inst_markers,
    }


class Timeline(QWidget):
    """Demucs vocal on top, draggable acapella sections below."""

    changed = pyqtSignal()
    seeked = pyqtSignal(float)
    dragFinished = pyqtSignal()
    mixChanged = pyqtSignal()
    splitAt = pyqtSignal(float)
    editBegan = pyqtSignal()
    menuHold = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.duration = 1.0
        self.view_start = 0.0
        self.view_span = 30.0
        self.stem = "aca"
        self.vox_peaks: np.ndarray | None = None
        self.aca_peaks: np.ndarray | None = None
        self._aca_ref_peaks: np.ndarray | None = None
        self._aca_out_peaks: np.ndarray | None = None
        self._inst_ref_peaks: np.ndarray | None = None
        self._inst_out_peaks: np.ndarray | None = None
        self.sections: list[Section] = []
        self.playhead = 0.0
        self.show_alignment = False
        self.show_bars = True
        self.bar_times: list[float] = []
        self.cue_on_bars = True
        self._cue_tip = ""
        self._drag: int | None = None
        self._drag_grab = 0.0
        self._drag_origin = 0.0
        self._pan: float | None = None
        self._pan_x = 0.0
        self._pan_start = 0.0
        self._hover_x: float | None = None
        self._drag_slip = False
        self._stretch: tuple[int, str] | None = None
        self._hover_edge: tuple[int, str] | None = None
        self.marker_times: list[float] = []
        self._aca_markers: list[float] = []
        self._inst_markers: list[float] = []
        self.solo = {"demucs": False, "output": False}
        self.mute = {"demucs": False, "output": False}
        self._demucs_solo = self._make_sm("S", "Solo Demucs vocal (1)")
        self._demucs_mute = self._make_sm("M", "Mute Demucs vocal (Shift+1)", danger=True)
        self._output_solo = self._make_sm("S", "Solo output acapella (2)")
        self._output_mute = self._make_sm("M", "Mute output acapella (Shift+2)", danger=True)
        self._demucs_solo.clicked.connect(lambda: self.toggle_solo("demucs"))
        self._demucs_mute.clicked.connect(lambda: self.toggle_mute("demucs"))
        self._output_solo.clicked.connect(lambda: self.toggle_solo("output"))
        self._output_mute.clicked.connect(lambda: self.toggle_mute("output"))
        self._sm_buttons = {
            "demucs": (self._demucs_solo, self._demucs_mute),
            "output": (self._output_solo, self._output_mute),
        }

    def set_song(
        self,
        duration: float,
        vox_peaks: np.ndarray,
        aca_peaks: np.ndarray,
        sections: list[Section],
        *,
        inst_ref_peaks: np.ndarray | None = None,
        inst_out_peaks: np.ndarray | None = None,
        bar_times: list[float] | None = None,
        cue_on_bars: bool = True,
        aca_markers: list[float] | None = None,
        inst_markers: list[float] | None = None,
    ) -> None:
        self.duration = max(duration, 0.1)
        self.view_span = self.duration
        self.view_start = 0.0
        self._aca_ref_peaks = vox_peaks
        self._aca_out_peaks = aca_peaks
        self._inst_ref_peaks = inst_ref_peaks
        self._inst_out_peaks = inst_out_peaks
        self.bar_times = [float(t) for t in (bar_times or [])]
        self.cue_on_bars = bool(cue_on_bars)
        self._aca_markers = [float(t) for t in (aca_markers or [])]
        self._inst_markers = [float(t) for t in (inst_markers or [])]
        self.sections = list(sections)
        self.playhead = 0.0
        self.set_stem("aca")

    def set_stem(self, stem: str) -> None:
        """Show the acapella pair or the instrumental pair."""
        self.stem = "inst" if stem == "inst" else "aca"
        if self.stem == "inst":
            self.vox_peaks = self._inst_ref_peaks
            self.aca_peaks = self._inst_out_peaks
            self._demucs_solo.setToolTip("Solo Demucs instrumental (1)")
            self._demucs_mute.setToolTip("Mute Demucs instrumental (Shift+1)")
            self._output_solo.setToolTip("Solo instrumental output (2)")
            self._output_mute.setToolTip("Mute instrumental output (Shift+2)")
            self.marker_times = self._inst_markers
        else:
            self.vox_peaks = self._aca_ref_peaks
            self.aca_peaks = self._aca_out_peaks
            self._demucs_solo.setToolTip("Solo Demucs vocal (1)")
            self._demucs_mute.setToolTip("Mute Demucs vocal (Shift+1)")
            self._output_solo.setToolTip("Solo output acapella (2)")
            self._output_mute.setToolTip("Mute output acapella (Shift+2)")
            self.marker_times = self._aca_markers
        self.update()

    def editing(self) -> bool:
        if self.stem == "inst":
            return self._inst_out_peaks is not None
        return self._aca_out_peaks is not None

    def set_playhead(self, t: float) -> None:
        self.playhead = t
        self.update()

    def _plot_w(self) -> float:
        return max(1.0, float(self.width() - GUTTER))

    def _x_to_t(self, x: float) -> float:
        return self.view_start + ((x - GUTTER) / self._plot_w()) * self.view_span

    def _t_to_x(self, t: float) -> float:
        return GUTTER + (t - self.view_start) / self.view_span * self._plot_w()

    def _make_sm(self, text: str, tip: str, *, danger: bool = False) -> QToolButton:
        btn = QToolButton(self)
        btn.setText(text)
        btn.setCheckable(True)
        btn.setFixedSize(28, 24)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tip)
        btn.setProperty("danger", danger)
        btn.setStyleSheet(_sm_style(active=False, danger=danger))
        return btn

    def audible(self, name: str) -> bool:
        if self.mute[name]:
            return False
        if any(self.solo.values()) and not self.solo[name]:
            return False
        return True

    def toggle_solo(self, name: str) -> None:
        enabling = not self.solo[name]
        for key in self.solo:
            self.solo[key] = enabling and key == name
        if enabling:
            self.mute[name] = False
        self._sync_sm()
        self.mixChanged.emit()
        self.update()

    def toggle_mute(self, name: str) -> None:
        self.mute[name] = not self.mute[name]
        if self.mute[name]:
            self.solo[name] = False
        self._sync_sm()
        self.mixChanged.emit()
        self.update()

    def _sync_sm(self) -> None:
        for name, (solo, mute) in self._sm_buttons.items():
            solo.blockSignals(True)
            mute.blockSignals(True)
            solo.setChecked(self.solo[name])
            mute.setChecked(self.mute[name])
            solo.setStyleSheet(_sm_style(active=self.solo[name]))
            mute.setStyleSheet(_sm_style(active=self.mute[name], danger=True))
            solo.blockSignals(False)
            mute.blockSignals(False)

    def _place_sm(self) -> None:
        ruler, dem_bot, aca_top, aca_bot = self._lanes()
        self._place_sm_pair(self._demucs_solo, self._demucs_mute, ruler, dem_bot)
        self._place_sm_pair(self._output_solo, self._output_mute, aca_top, aca_bot)

    def _place_sm_pair(self, solo: QToolButton, mute: QToolButton, top: int, bot: int) -> None:
        y = top + max(0, (bot - top - solo.height()) // 2)
        solo.move(8, y)
        mute.move(40, y)
        solo.show()
        mute.show()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_sm()

    def _lanes(self) -> tuple[int, int, int, int]:
        h = self.height()
        ruler = 22
        body = max(1, h - ruler)
        dem = ruler + int(body * 0.46)
        return ruler, dem, dem, h

    def _hit_section(self, x: float, y: float) -> int | None:
        if not self.editing():
            return None
        _r, _d, aca_top, aca_bot = self._lanes()
        if y < aca_top or y > aca_bot:
            return None
        t = self._x_to_t(x)
        for i in range(len(self.sections) - 1, -1, -1):
            sec = self.sections[i]
            if sec.dst0 <= t <= sec.dst1:
                return i
        return None

    def _clamp_dst(self, index: int, dst: float) -> float:
        sec = self.sections[index]
        return max(0.0, min(self.duration - sec.out_dur, dst))

    def _split_partner(self, index: int, side: str) -> tuple[int, str] | None:
        """The other section that meets this edge at a split."""
        sec = self.sections[index]
        t = sec.dst1 if side == "right" else sec.dst0
        want = "left" if side == "right" else "right"
        for j, other in enumerate(self.sections):
            if j == index:
                continue
            other_t = other.dst0 if want == "left" else other.dst1
            if abs(other_t - t) <= 0.02:
                return (j, want)
        return None

    def _hit_edge(self, x: float, y: float) -> tuple[int, str] | None:
        if not self.editing():
            return None
        _r, _d, aca_top, aca_bot = self._lanes()
        if y < aca_top or y > aca_bot:
            return None
        near: list[tuple[float, int, str, float]] = []
        for i, sec in enumerate(self.sections):
            for side, edge in (("left", sec.dst0), ("right", sec.dst1)):
                dx = abs(x - self._t_to_x(edge))
                if dx <= 8.0:
                    near.append((dx, i, side, edge))
        if not near:
            return None
        _dx, index, side, edge = min(near)
        partner = self._split_partner(index, side)
        if partner is None:
            return (index, side)
        split_x = self._t_to_x(edge)
        if x > split_x:
            return (index, "left") if side == "left" else partner
        return (index, "right") if side == "right" else partner

    def _clock(self) -> QCursor:
        global _CLOCK
        if _CLOCK is None:
            _CLOCK = _clock_cursor()
        return _CLOCK

    def _apply_stretch(self, x: float) -> None:
        if self._stretch is None:
            return
        idx, side = self._stretch
        sec = self.sections[idx]
        t = self._x_to_t(x)
        src = max(0.05, sec.dur)
        if side == "right":
            raw = t - self._stretch_dst0
            dur = min(max(raw, src / 2.0), src * 2.0)
            dur = min(dur, max(0.05, self.duration - sec.dst0))
            sec.dst_dur = None if abs(dur - src) < 1e-3 else dur
        else:
            right = self._stretch_dst0 + self._stretch_dur
            raw = right - t
            dur = min(max(raw, src / 2.0), src * 2.0)
            dst0 = right - dur
            if dst0 < 0.0:
                dur += dst0
                dst0 = 0.0
            sec.dst0 = dst0
            sec.dst_dur = None if abs(dur - src) < 1e-3 else dur
        self.changed.emit()
        self.update()

    def wheelEvent(self, event) -> None:  # noqa: N802
        t = self._x_to_t(float(event.position().x()))
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        span = min(self.duration, max(2.0, self.view_span * factor))
        frac = 0.0 if self.view_span <= 0 else (t - self.view_start) / self.view_span
        self.view_span = span
        self.view_start = min(max(0.0, t - frac * span), max(0.0, self.duration - span))
        self.changed.emit()
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        x = float(event.position().x())
        y = float(event.position().y())
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            edge = self._hit_edge(x, y) if event.button() == Qt.MouseButton.LeftButton else None
            if edge is not None:
                idx, side = edge
                sec = self.sections[idx]
                self.editBegan.emit()
                self._stretch = edge
                self._stretch_dst0 = sec.dst0
                self._stretch_dur = sec.out_dur
                self.setCursor(self._clock())
                return
            self._pan = self.view_start
            self._pan_x = x
            self._pan_start = self.view_start
            return
        if event.button() == Qt.MouseButton.RightButton:
            if x < GUTTER:
                return
            t = max(0.0, min(self.duration, self._x_to_t(x)))
            self.seeked.emit(t)
            self.repaint()
            if not self.editing():
                return
            self.menuHold.emit(True)
            menu = QMenu(self)
            act = QAction("Split from here", menu)
            act.triggered.connect(lambda _checked=False, tt=t: self.splitAt.emit(tt))
            menu.addAction(act)
            try:
                menu.exec(event.globalPosition().toPoint())
            finally:
                self.menuHold.emit(False)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if x < GUTTER:
            return
        hit = self._hit_section(x, y)
        if hit is not None:
            self.editBegan.emit()
            self._drag = hit
            self._drag_slip = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            sec = self.sections[hit]
            if self._drag_slip:
                self._drag_origin = sec.src0
                self._drag_span = sec.dur
                self._drag_grab = self._x_to_t(x)
            else:
                self._drag_origin = sec.dst0
                self._drag_grab = self._x_to_t(x) - sec.dst0
            return
        self.seeked.emit(max(0.0, min(self.duration, self._x_to_t(x))))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x = float(event.position().x())
        self._set_hover_x(x)
        alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        if self._stretch is not None:
            self._apply_stretch(x)
            return
        if self._pan is not None:
            dx = x - self._pan_x
            self.view_start = min(
                max(0.0, self._pan_start - dx / self._plot_w() * self.view_span),
                max(0.0, self.duration - self.view_span),
            )
            self.changed.emit()
            self.update()
            return
        if self._drag is None:
            self._hover_cue(event)
            edge = self._hit_edge(x, float(event.position().y())) if alt else None
            self._hover_edge = edge
            if edge is not None:
                self.setCursor(self._clock())
            else:
                hit = self._hit_section(x, float(event.position().y()))
                self.setCursor(
                    Qt.CursorShape.SizeHorCursor if hit is not None else Qt.CursorShape.ArrowCursor
                )
            if alt:
                self.update()
            return
        t = self._x_to_t(x)
        sec = self.sections[self._drag]
        if self._drag_slip:
            delta = t - self._drag_grab
            span = self._drag_span
            src0 = self._drag_origin - delta
            src0 = max(0.0, min(src0, max(0.0, self.duration - span)))
            sec.src0 = src0
            sec.src1 = src0 + span
        else:
            delta = (t - self._drag_grab) - self._drag_origin
            sec.dst0 = self._clamp_dst(self._drag, self._drag_origin + delta)
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        dragged = self._drag is not None or self._stretch is not None
        self._drag = None
        self._drag_slip = False
        self._stretch = None
        self._pan = None
        alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        x = float(event.position().x())
        y = float(event.position().y())
        self._hover_edge = self._hit_edge(x, y) if alt else None
        self.update()
        if dragged:
            self.dragFinished.emit()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hover_x = None
        self._cue_tip = ""
        QToolTip.hideText()
        self._hover_edge = None
        if self._stretch is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Alt:
            self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Alt and self._stretch is None:
            self._hover_edge = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
        super().keyReleaseEvent(event)

    def _set_hover_x(self, x: float) -> None:
        nxt = x if x >= GUTTER else None
        if nxt == self._hover_x:
            return
        self._hover_x = nxt
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG))
        w = self.width()
        h = self.height()
        ruler, dem_bot, aca_top, aca_bot = self._lanes()
        p.setClipRect(GUTTER, 0, max(1, w - GUTTER), h)
        p.fillRect(GUTTER, ruler, w - GUTTER, dem_bot - ruler, QColor(LANE_DEM))
        p.fillRect(GUTTER, aca_top, w - GUTTER, aca_bot - aca_top, QColor(PANEL))
        vox_color = _dim_wave(DEMUCS, LANE_DEM, dim=not self.audible("demucs"))
        self._draw_wave(p, self.vox_peaks, ruler, dem_bot, vox_color, None)
        if self.editing():
            self._draw_sections(p, aca_top, aca_bot)
        else:
            out_color = _dim_wave(ACA, PANEL, dim=not self.audible("output"))
            self._draw_wave(p, self.aca_peaks, aca_top, aca_bot, out_color, None)
        if self.show_alignment:
            self._draw_alignment(p, ruler, h)
        self._draw_ruler(p, ruler)
        self._draw_warp_cues(p, ruler, h)
        if self._hover_x is not None:
            hx = int(round(self._hover_x))
            if GUTTER <= hx <= w:
                p.setPen(QPen(QColor("#6a7080"), 1))
                p.drawLine(hx, ruler, hx, h)
        if 0 <= self.playhead <= self.duration:
            x = int(self._t_to_x(self.playhead))
            if GUTTER <= x <= w:
                p.setPen(QPen(QColor("#d0d4de"), 1))
                p.drawLine(x, ruler, x, h)
        p.setPen(QColor(DIM))
        p.setFont(QFont("Segoe UI", 8))
        if self.stem == "inst":
            ref_name = "Demucs instrumental"
        else:
            ref_name = "Demucs vocal"
        out_name = "Output — drag a section" if self.editing() else "Instrumental output"
        p.drawText(GUTTER + 8, ruler + 14, ref_name)
        p.drawText(GUTTER + 8, aca_top + 14, out_name)
        p.end()

    def _alignment_step(self) -> float:
        """Seconds between lines. Zooming in brings the lines closer, down to 0.1 s."""
        target = self.view_span / self._plot_w() * 72.0
        for cand in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60):
            if cand >= target * 0.75:
                return float(cand)
        return 60.0

    def _draw_alignment(self, p: QPainter, ruler_h: int, bottom: int) -> None:
        step = self._alignment_step()
        if step <= 0:
            return
        p.setPen(QPen(QColor(210, 216, 232, 78), 1))
        t = float(np.ceil((self.view_start - 1e-6) / step) * step)
        end = self.view_start + self.view_span
        while t <= end + 1e-6:
            x = int(round(self._t_to_x(t)))
            if x >= GUTTER:
                p.drawLine(x, ruler_h, x, bottom)
            t += step

    def _draw_warp_cues(self, p: QPainter, ruler_h: int, bottom: int) -> None:
        end = self.view_start + self.view_span
        tick = QColor(STRETCH_CUE)
        tick.setAlpha(90)
        p.setPen(QPen(tick, 1))
        for t in self.bar_times:
            if t < self.view_start - 0.01 or t > end + 0.01:
                continue
            x = int(round(self._t_to_x(float(t))))
            if GUTTER <= x <= self.width():
                p.drawLine(x, ruler_h - 6, x, ruler_h)
        if not self.show_bars or not self.marker_times:
            return
        line = QColor(STRETCH_CUE)
        line.setAlpha(72)
        pen = QPen(line, 1)
        for t in self.marker_times:
            if t < self.view_start - 0.01 or t > end + 0.01:
                continue
            x = int(round(self._t_to_x(float(t))))
            if x < GUTTER or x > self.width():
                continue
            p.setPen(pen)
            p.drawLine(x, ruler_h, x, bottom)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(STRETCH_CUE))
            p.drawPolygon(
                QPolygon([QPoint(x, ruler_h), QPoint(x - 4, ruler_h - 7), QPoint(x + 4, ruler_h - 7)])
            )
        p.setBrush(Qt.BrushStyle.NoBrush)

    def _hover_cue(self, event) -> None:
        tip = self._cue_tip_at(float(event.position().x()))
        if tip == self._cue_tip:
            return
        self._cue_tip = tip
        if tip:
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)
        else:
            QToolTip.hideText()

    def _cue_tip_at(self, x: float) -> str:
        if not self.show_bars:
            return ""
        hit = self._near_cue(self.marker_times, x)
        if hit is None:
            return ""
        clock = _cue_clock(hit)
        return f"Warp marker at {clock}."

    def _near_cue(self, times: list[float], x: float) -> float | None:
        best: float | None = None
        best_dx = 7.0
        for t in times:
            dx = abs(self._t_to_x(float(t)) - x)
            if dx <= best_dx:
                best = float(t)
                best_dx = dx
        return best

    def _draw_ruler(self, p: QPainter, ruler_h: int) -> None:
        p.setPen(QColor("#3a3d4d"))
        p.drawLine(GUTTER, ruler_h, self.width(), ruler_h)
        span = self.view_span
        step = 1.0
        for cand in (1, 2, 5, 10, 15, 30, 60):
            if span / cand < 12:
                step = float(cand)
                break
        p.setPen(QColor(DIM))
        p.setFont(QFont("Segoe UI", 8))
        t = np.ceil(self.view_start / step) * step
        while t < self.view_start + span:
            x = int(self._t_to_x(float(t)))
            p.drawLine(x, ruler_h - 6, x, ruler_h)
            m, s = divmod(int(t), 60)
            p.drawText(x + 3, 14, f"{m}:{s:02d}")
            t += step

    def _draw_wave(
        self,
        p: QPainter,
        peaks: np.ndarray | None,
        top: int,
        bot: int,
        color: QColor,
        src_dst: tuple[float, float, float, float] | None,
    ) -> None:
        if peaks is None or len(peaks) == 0:
            return
        mid = (top + bot) / 2
        amp = max(4.0, (bot - top) * 0.42)
        p.setPen(QPen(color, 1))
        n = len(peaks)
        if src_dst is None:
            t0, t1 = self.view_start, self.view_start + self.view_span
            i0 = max(0, int(t0 * PEAK_HZ))
            i1 = min(n, int(t1 * PEAK_HZ) + 1)
            for i in range(i0, i1):
                tt = i / PEAK_HZ
                x = int(self._t_to_x(tt))
                lo, hi = float(peaks[i, 0]), float(peaks[i, 1])
                p.drawLine(x, int(mid - hi * amp), x, int(mid - lo * amp))
            return
        src0, src1, dst0, out_dur = src_dst
        scale = out_dur / max(1e-6, src1 - src0)
        i0 = max(0, int(src0 * PEAK_HZ))
        i1 = min(n, int(src1 * PEAK_HZ) + 1)
        for i in range(i0, i1):
            src_t = i / PEAK_HZ
            dst_t = dst0 + (src_t - src0) * scale
            if dst_t < self.view_start or dst_t > self.view_start + self.view_span:
                continue
            x = int(self._t_to_x(dst_t))
            lo, hi = float(peaks[i, 0]), float(peaks[i, 1])
            p.drawLine(x, int(mid - hi * amp), x, int(mid - lo * amp))

    def _draw_sections(self, p: QPainter, top: int, bot: int) -> None:
        hot = self._stretch if self._stretch is not None else self._hover_edge
        for i, sec in enumerate(self.sections):
            x0 = self._t_to_x(sec.dst0)
            x1 = self._t_to_x(sec.dst1)
            if x1 < 0 or x0 > self.width():
                continue
            muted = not self.audible("output")
            wave = _dim_wave(ACA, PANEL, dim=muted)
            held = i == self._drag or (hot is not None and hot[0] == i)
            fill = QColor(SECTION_DRAG_FILL if held else SECTION_FILL)
            edge = QColor(SECTION_EDGE)
            if muted:
                edge.setAlpha(90)
            elif held:
                edge = QColor(SECTION_DRAG_EDGE)
            p.fillRect(int(x0), top + 4, max(1, int(x1 - x0)), bot - top - 8, fill)
            p.setPen(QPen(edge, 1))
            p.drawRect(int(x0), top + 4, max(1, int(x1 - x0)), bot - top - 8)
            self._draw_wave(
                p,
                self.aca_peaks,
                top + 6,
                bot - 4,
                wave,
                (sec.src0, sec.src1, sec.dst0, sec.out_dur),
            )
        if hot is None or not (0 <= hot[0] < len(self.sections)):
            return
        sec = self.sections[hot[0]]
        where = "top" if hot[1] == "right" else "bottom"
        ex = sec.dst1 if hot[1] == "right" else sec.dst0
        self._draw_clock(p, self._t_to_x(ex), top, bot, where)

    def _draw_clock(self, p: QPainter, x: float, top: int, bot: int, where: str) -> None:
        cx = int(x)
        cy = top + 10 if where == "top" else bot - 10
        r = 6
        p.setPen(QPen(QColor("#e6e8ef"), 1.2))
        p.setBrush(QColor("#1e1f26"))
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)
        p.drawLine(cx, cy, cx, cy - 4)
        p.drawLine(cx, cy, cx + 3, cy + 2)


class SectionEditor(QWidget):
    def __init__(
        self,
        folder: Path,
        *,
        declick: str = "rx",
        on_apply_begin=None,
        on_apply_step=None,
        on_apply_scored=None,
        on_apply_end=None,
        tag_folders=None,
    ) -> None:
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.folder = folder
        self._declick = declick or "rx"
        self._on_apply_begin = on_apply_begin
        self._on_apply_step = on_apply_step
        self._on_apply_scored = on_apply_scored
        self._on_apply_end = on_apply_end
        self._tag_folders_enabled = tag_folders
        self.setObjectName("AppRoot")
        self.setWindowTitle("Audio Aligner - Editor")
        self.resize(WIN_DEFAULT_W, WIN_DEFAULT_H)
        self.setMinimumSize(WIN_MIN_W, WIN_MIN_H)
        self._custom_maximized = False
        self._restore_geometry = None
        self._center_pending = True
        prepare_dark_frameless_chrome(self)
        install_rounded_corner_watcher(self)
        self.setStyleSheet(
            f"QWidget#AppRoot, QWidget#AppBody {{ background: {BG}; color: {FG}; "
            f"font-family: 'Segoe UI'; font-size: 12px; }}"
            f"QWidget#TitleBar {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}"
            f"QLabel#FolderChip {{ color: {FG}; background: {CHIP_BG}; "
            f"padding: 4px 8px; border-radius: 4px; }}"
            f"QLabel#StatusLabel {{ color: {DIM}; background: transparent; }}"
            f"QPushButton {{ background: {CHIP_BG}; color: {FG}; border: 1px solid {BORDER}; "
            f"border-radius: 6px; padding: 6px 14px; min-height: 26px; }}"
            f"QPushButton:hover {{ background: #36384A; }}"
            f"QPushButton:pressed {{ background: #2A2C38; }}"
            f"QPushButton:disabled {{ color: #7a8199; }}"
            f"QRadioButton {{ color: {FG}; background: transparent; spacing: 8px; }}"
            f"QRadioButton::indicator {{ width: 14px; height: 14px; border-radius: 7px; "
            f"border: 1px solid {BORDER}; background: {CHIP_BG}; }}"
            f"QRadioButton::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}"
            f"QCheckBox {{ color: {FG}; spacing: 8px; background: transparent; }}"
            f"QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; "
            f"border: 1px solid {BORDER}; background: {CHIP_BG}; }}"
            f"QCheckBox::indicator:checked {{ background: {ACCENT}; border: 1px solid {ACCENT}; }}"
            f"QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}"
            f"QScrollBar::handle:horizontal {{ background: #3a3d4d; border-radius: 4px; min-width: 24px; }}"
            f"QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}"
            f"QMenu {{ background: {PANEL}; color: {FG}; border: 1px solid {BORDER}; padding: 4px; }}"
            f"QMenu::item {{ padding: 6px 16px; border-radius: 4px; }}"
            f"QMenu::item:selected {{ background: #36384A; }}"
            f"QSlider#VolumeSlider::groove:horizontal {{ height: 4px; background: {BORDER}; "
            f"border-radius: 2px; }}"
            f"QSlider#VolumeSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}"
            f"QSlider#VolumeSlider::add-page:horizontal {{ background: {BORDER}; border-radius: 2px; }}"
            f"QSlider#VolumeSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; "
            f"background: #d9d2ff; border: 2px solid {ACCENT}; border-radius: 7px; }}"
        )
        self._data: dict | None = None
        self._initial: dict[str, tuple] = {"aca": (), "inst": ()}
        self._section_bank: dict[str, list[Section]] = {"aca": [], "inst": []}
        self._aca_play: np.ndarray | None = None
        self._aca_key: tuple | None = None
        self._vox_st: np.ndarray | None = None
        self._inst_st: np.ndarray | None = None
        self._dinst_st: np.ndarray | None = None
        self._aca_st: np.ndarray | None = None
        self._play_sr = 0
        self._play = _EditorPlayback()
        self._cue: float | None = None

        self.timeline = Timeline()
        self.timeline.changed.connect(self._on_timeline_changed)
        self.timeline.seeked.connect(self._seek)
        self.timeline.dragFinished.connect(self._on_drag_finished)
        self.timeline.splitAt.connect(self._split_at)
        self.timeline.menuHold.connect(self._play.set_hold)
        self.timeline.editBegan.connect(self._note_edit)
        self.timeline.dragFinished.connect(self._commit_drag_undo)
        self._undo: dict[str, list[tuple]] = {"aca": [], "inst": []}
        self._pending_undo: tuple | None = None
        self.scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.scroll.valueChanged.connect(self._on_scroll)

        self._stem_group = QButtonGroup(self)
        self._stem_group.setExclusive(True)
        self.aca_radio = QRadioButton("Acapella")
        self.inst_radio = QRadioButton("Instrumental")
        self.aca_radio.setChecked(True)
        self._stem_group.addButton(self.aca_radio, 0)
        self._stem_group.addButton(self.inst_radio, 1)
        self.aca_radio.setToolTip(
            "Shows the Demucs vocal and the acapella.\n"
            "Apply writes this stem."
        )
        self.inst_radio.setToolTip(
            "Shows the Demucs instrumental and its output.\n"
            "Apply writes this stem."
        )
        self._stem_group.idClicked.connect(self._on_stem_changed)
        self.status = QLabel("Loading Demucs vocal and acapella…")
        self.status.setObjectName("StatusLabel")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.with_demucs = QCheckBox("Play against Demucs vocal")
        self.with_demucs.setChecked(True)
        self.with_demucs.setToolTip(
            "On plays the acapella against the Demucs vocal.\n"
            "Off plays the acapella with the instrumental."
        )
        self.with_demucs.toggled.connect(lambda _checked: self._sync_gains())
        self.align_lines = QCheckBox("Alignment lines")
        self.align_lines.setToolTip(
            "Lines mark the same moment on both lanes.\n"
            "Drag a section and the wave moves against them."
        )
        self.align_lines.toggled.connect(self._toggle_alignment)
        self.bar_lines = QCheckBox("Warp markers")
        self.bar_lines.setChecked(True)
        self.bar_lines.setStyleSheet(f"QCheckBox {{ color: {STRETCH_CUE}; background: transparent; }}")
        self.bar_lines.setToolTip(
            "Shows a marker where the lag has stepped.\n"
            "Faint ticks on the ruler are every 8 bars.\n"
            "A straight song has no marker."
        )
        self.bar_lines.toggled.connect(self._toggle_bars)
        self.timeline.mixChanged.connect(self._sync_gains)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self._reset)
        self.sections_lbl = QLabel("Sections")
        self.sections_lbl.setStyleSheet(f"color: {FG}; background: transparent;")
        self.fewer_btn = QPushButton("-")
        self.fewer_btn.setEnabled(False)
        self.fewer_btn.setToolTip("Join nearby phrases into longer sections.")
        self.fewer_btn.clicked.connect(lambda: self._retarget_sections(-1))
        self.more_btn = QPushButton("+")
        self.more_btn.setEnabled(False)
        self.more_btn.setToolTip("Split the longest phrases into shorter sections.")
        self.more_btn.clicked.connect(lambda: self._retarget_sections(1))
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setToolTip(
            "Write the dragged sections, then re-align this acapella.\n"
            "The instrumental is not changed."
        )
        self.apply_btn.clicked.connect(self._apply)

        row = QHBoxLayout()
        row.addWidget(self.with_demucs)
        row.addWidget(self.align_lines)
        row.addWidget(self.bar_lines)
        row.addWidget(self.status, 1)
        row.addWidget(self.sections_lbl)
        row.addWidget(self.fewer_btn)
        row.addWidget(self.more_btn)
        row.addWidget(self.reset_btn)
        row.addWidget(self.apply_btn)

        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        icon = Path(__file__).resolve().parent / "icon.png"
        if not icon.is_file():
            icon = Path(__file__).resolve().parent / "logo.ico"
        self.title_bar = CustomTitleBar(self, title="Audio Aligner - Editor", icon_path=icon)
        shell.addWidget(self.title_bar)

        body = QWidget()
        body.setObjectName("AppBody")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.folder_lbl = QLabel(folder.name)
        self.folder_lbl.setObjectName("FolderChip")
        self.folder_lbl.setToolTip("Open folder in Explorer")
        self.folder_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self.folder_lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.folder_lbl.installEventFilter(self)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self.folder_lbl, 0)
        header.addSpacing(14)
        header.addWidget(self.aca_radio)
        header.addWidget(self.inst_radio)
        header.addStretch(1)
        self.time_lbl = QLabel("00:00:000")
        self.time_lbl.setStyleSheet(
            f"color: {DIM}; font-family: Consolas, 'Cascadia Mono', monospace; "
            f"font-size: 14px; background: transparent;"
        )
        header.addWidget(self.time_lbl)
        self.prev_btn = _transport_button(
            body, "prev", "Seek −0.1 s", lambda: self._seek_relative(-SEEK_STEP_SEC)
        )
        self.prev_btn.setAutoRepeat(True)
        self.prev_btn.setAutoRepeatDelay(250)
        self.prev_btn.setAutoRepeatInterval(80)
        self.transport_play = _transport_button(body, "play", "Play / Pause (Space)", self._toggle_play)
        self.stop_btn = _transport_button(body, "stop", "Stop", self._stop)
        self.next_btn = _transport_button(
            body, "next", "Seek +0.1 s", lambda: self._seek_relative(SEEK_STEP_SEC)
        )
        self.next_btn.setAutoRepeat(True)
        self.next_btn.setAutoRepeatDelay(250)
        self.next_btn.setAutoRepeatInterval(80)
        for btn in (self.prev_btn, self.transport_play, self.stop_btn, self.next_btn):
            btn.setEnabled(False)
            header.addWidget(btn)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setObjectName("VolumeSlider")
        self.volume.setRange(0, 100)
        self.volume.setValue(85)
        self.volume.setFixedWidth(120)
        self.volume.setToolTip("Volume")
        self.volume.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.volume.valueChanged.connect(self._on_volume)
        header.addWidget(self.volume)
        lay.addLayout(header)
        lay.addWidget(self.timeline, 1)
        lay.addWidget(self.scroll)
        lay.addLayout(row)
        lay.addWidget(self._shortcuts_bar())
        shell.addWidget(body, 1)

        if HAS_MEDIA:
            self._tick = QTimer(self)
            self._tick.setInterval(33)
            self._tick.timeout.connect(self._on_audio_tick)

        for widget in (
            self.fewer_btn,
            self.more_btn,
            self.aca_radio,
            self.inst_radio,
            self.with_demucs,
            self.align_lines,
            self.bar_lines,
            self.reset_btn,
            self.apply_btn,
            self.scroll,
        ):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._bind_keys()

        self._busy = False
        self._apply_worker: _ApplyWorker | None = None
        self._loader: _Loader | None = None
        self._start_loader(self._on_loaded)

    def _shortcuts_bar(self) -> QWidget:
        self._shortcuts_host = QWidget()
        self._shortcuts_layout = QHBoxLayout(self._shortcuts_host)
        self._shortcuts_layout.setContentsMargins(4, 2, 4, 0)
        self._shortcuts_layout.setSpacing(12)
        self._populate_shortcuts()
        return self._shortcuts_host

    def _populate_shortcuts(self) -> None:
        lay = self._shortcuts_layout
        while lay.count():
            item = lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        # The last three only fit once the window is maximized.
        groups = (
            (("Space",), "Play / Pause", "plus", False),
            (("1", "2"), "Solo", "gap", False),
            (("Shift+1", "Shift+2"), "Mute", "gap", False),
            (("<-", "->"), "Seek +/-0.1s", "gap", False),
            (("Shift+<-", "Shift+->"), "+/-10ms", "gap", True),
            (("S",), "Split section", "plus", False),
            (("Ctrl+Z",), "Undo", "plus", False),
            (("Wheel",), "Zoom", "plus", False),
            (("Alt", "drag"), "Pan", "plus", True),
            (("Shift", "drag"), "Fine move", "plus", True),
        )
        maximized = bool(self.windowState() & Qt.WindowState.WindowMaximized)
        visible = [group for group in groups if maximized or not group[3]]
        lay.addStretch(1)
        for i, (keys, label, join, _extra) in enumerate(visible):
            if i:
                lay.addStretch(1)
            lay.addWidget(_shortcut_group(keys, label, join=join))
        lay.addStretch(1)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return
        if hasattr(self, "_shortcuts_layout"):
            self._populate_shortcuts()

    def _bind_keys(self) -> None:
        self._shortcuts: list[QShortcut] = []

        def _sc(key, handler, *, repeat: bool = True) -> None:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.setAutoRepeat(repeat)
            shortcut.activated.connect(handler)
            self._shortcuts.append(shortcut)

        _sc(Qt.Key.Key_Space, self._toggle_play, repeat=False)
        _sc(Qt.Key.Key_Left, lambda: self._seek_relative(-SEEK_STEP_SEC))
        _sc(Qt.Key.Key_Right, lambda: self._seek_relative(SEEK_STEP_SEC))
        _sc("Shift+Left", lambda: self._seek_relative(-SEEK_FINE_SEC))
        _sc("Shift+Right", lambda: self._seek_relative(SEEK_FINE_SEC))
        _sc("S", self._split_at_playhead, repeat=False)
        _sc(QKeySequence.StandardKey.Undo, self._undo_edit, repeat=False)
        _sc("1", lambda: self.timeline.toggle_solo("demucs"), repeat=False)
        _sc("2", lambda: self.timeline.toggle_solo("output"), repeat=False)
        _sc("Shift+1", lambda: self.timeline.toggle_mute("demucs"), repeat=False)
        _sc("Shift+2", lambda: self.timeline.toggle_mute("output"), repeat=False)

    def _section_snapshot(self) -> tuple:
        return tuple(s.snap() for s in self.timeline.sections)

    def _sections_dirty(self) -> bool:
        if self._data is None:
            return False
        return self._section_snapshot() != self._initial.get(self.timeline.stem, ())

    def _undos(self) -> list[tuple]:
        return self._undo.setdefault(self.timeline.stem, [])

    def _push_undo(self) -> None:
        stack = self._undos()
        stack.append(self._section_snapshot())
        if len(stack) > 50:
            del stack[:-50]

    def _remember_sections(self) -> None:
        self._section_bank[self.timeline.stem] = list(self.timeline.sections)

    def _sync_edit_buttons(self) -> None:
        on = not self._busy and self._sections_dirty() and self.timeline.editing()
        self.reset_btn.setEnabled(on)
        self.apply_btn.setEnabled(on)
        ready = self.timeline.editing() and not self._busy and self._data is not None
        n = len(self.timeline.sections)
        self.fewer_btn.setEnabled(ready and n > 1)
        self.more_btn.setEnabled(ready and n > 0)

    def _note_edit(self) -> None:
        self._pending_undo = self._section_snapshot()

    def _commit_drag_undo(self) -> None:
        pending = self._pending_undo
        self._pending_undo = None
        if pending is None or pending == self._section_snapshot():
            return
        stack = self._undos()
        stack.append(pending)
        if len(stack) > 50:
            del stack[:-50]

    def _retarget_sections(self, direction: int) -> None:
        if self._busy or self._data is None or self.timeline._drag is not None or not self.timeline.editing():
            return
        env_key = "inst_v" if self.timeline.stem == "inst" else "aca_v"
        shaped = self._data.get(env_key)
        if shaped is None:
            return
        env = rms_env(shaped)
        current = self.timeline.sections
        if direction > 0:
            nxt = _more_sections(current, env)
            stuck = "These sections are already short enough to drag one at a time."
        else:
            nxt = _fewer_sections(current)
            stuck = "Already one section."
        if not nxt:
            self.status.setText(stuck)
            return
        self._push_undo()
        self.timeline.sections = nxt
        self._invalidate_aca_play()
        if self._is_playing():
            self._ensure_aca_play()
        self.timeline.update()
        self._sync_edit_buttons()
        word = "sections" if len(nxt) != 1 else "section"
        self.status.setText(f"{len(nxt)} {word}.")

    def _restore_sections(self, snap: tuple) -> None:
        self.timeline.sections = [section_from_snap(item) for item in snap]
        self._invalidate_aca_play()
        if self._is_playing() and self._data is not None:
            self._ensure_aca_play()
        self.timeline.update()

    def _undo_edit(self) -> None:
        if self._busy:
            return
        stack = self._undos()
        if not stack:
            self.status.setText("Nothing to undo.")
            return
        self._restore_sections(stack.pop())
        self._sync_edit_buttons()
        self.status.setText("Undid the last edit.")

    def _on_loaded(self, data: dict) -> None:
        self._halt_audio()
        self._vox_st = None
        self._inst_st = None
        self._dinst_st = None
        self._aca_st = None
        self._play.clear_buffers()
        self._data = data
        self._play.sr = int(data["sr"])
        self._play_sr = self._play.sr
        self._cue = None
        self._invalidate_aca_play()
        self._undo = {"aca": [], "inst": []}
        self._pending_undo = None
        aca_sections = list(data["sections"])
        inst_sections = list(data.get("inst_sections") or [])
        self._section_bank = {"aca": aca_sections, "inst": inst_sections}

        def _snap(sections: list[Section]) -> tuple:
            return tuple(s.snap() for s in sections)

        self._initial = {"aca": _snap(aca_sections), "inst": _snap(inst_sections)}
        self.timeline.set_song(
            data["duration"],
            data["vox_peaks"],
            data["aca_peaks"],
            data["sections"],
            inst_ref_peaks=data.get("demucs_inst_peaks"),
            inst_out_peaks=data.get("inst_peaks"),
            bar_times=list(data.get("bar_times") or []),
            cue_on_bars=bool(data.get("cue_on_bars", True)),
            aca_markers=list(data.get("aca_markers") or []),
            inst_markers=list(data.get("inst_markers") or []),
        )
        self._set_stem_choice("aca")
        self._apply_pair_chrome()
        self._sync_scroll()
        self._sync_transport()
        self._sync_edit_buttons()
        n = len(data["sections"])
        self.status.setText(
            f"{n} sections. Drag them onto the Demucs vocal."
        )
        if not HAS_MEDIA:
            self.status.setText(self.status.text() + " Playback needs sounddevice.")

    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.status.setText(message)
        QMessageBox.warning(self, "Could not open editor", message)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Alt:
            self.timeline.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Alt:
            self.timeline.keyReleaseEvent(event)
            return
        super().keyReleaseEvent(event)

    def _on_timeline_changed(self) -> None:
        self._sync_scroll()
        self._sync_edit_buttons()
        if self.timeline._stretch is not None:
            idx = self.timeline._stretch[0]
            sec = self.timeline.sections[idx]
            pct = 100.0 * sec.dur / max(1e-6, sec.out_dur)
            self.status.setText(f"Section {idx + 1} at {pct:.0f}% speed.")
            return
        if self.timeline._drag is not None:
            sec = self.timeline.sections[self.timeline._drag]
            m, s = divmod(sec.dst0, 60)
            self.status.setText(f"Section {self.timeline._drag + 1} at {int(m)}:{s:04.1f}")

    def _sync_scroll(self) -> None:
        span = self.timeline.view_span
        dur = self.timeline.duration
        self.scroll.blockSignals(True)
        max_ms = int(max(0.0, dur - span) * 1000)
        self.scroll.setRange(0, max_ms)
        self.scroll.setPageStep(int(span * 1000))
        self.scroll.setValue(int(self.timeline.view_start * 1000))
        self.scroll.blockSignals(False)

    def _on_scroll(self, value: int) -> None:
        self.timeline.view_start = value / 1000.0
        self.timeline.update()

    def _set_stem_choice(self, stem: str) -> None:
        button = self.inst_radio if stem == "inst" else self.aca_radio
        self._stem_group.blockSignals(True)
        button.setChecked(True)
        self._stem_group.blockSignals(False)

    def _on_stem_changed(self, _checked_id: int = 0) -> None:
        stem = "inst" if self._stem_group.checkedId() == 1 else "aca"
        if self._data is None:
            return
        if stem == "inst" and self._data.get("inst") is None:
            self._set_stem_choice("aca")
            self.status.setText("This folder has no instrumental.")
            return
        if stem != self.timeline.stem:
            self._remember_sections()
        self.timeline.set_stem(stem)
        self.timeline.sections = list(self._section_bank.get(stem) or [])
        self._apply_pair_chrome()
        self._invalidate_aca_play()
        if self._is_playing():
            self._ensure_beds()
            self._ensure_aca_play()
        self._sync_gains()
        self._sync_edit_buttons()
        n = len(self.timeline.sections)
        if stem == "inst":
            self.status.setText(f"{n} sections. Drag them onto the Demucs instrumental.")
        else:
            self.status.setText(f"{n} sections. Drag them onto the Demucs vocal.")

    def _apply_pair_chrome(self) -> None:
        if self.timeline.stem == "inst":
            self.with_demucs.setText("Play against Demucs instrumental")
            self.with_demucs.setToolTip(
                "On plays the instrumental against the Demucs instrumental.\n"
                "Off plays the instrumental with the acapella."
            )
            self.align_lines.setToolTip(
                "Lines mark the same moment on both lanes.\n"
                "Drag a section and the wave moves against them."
            )
            self.apply_btn.setToolTip(
                "Write the dragged sections, then re-align this instrumental.\n"
                "The acapella is not changed."
            )
            return
        self.with_demucs.setText("Play against Demucs vocal")
        self.with_demucs.setToolTip(
            "On plays the acapella against the Demucs vocal.\n"
            "Off plays the acapella with the instrumental."
        )
        self.align_lines.setToolTip(
            "Lines mark the same moment on both lanes.\n"
            "Drag a section and the wave moves against them."
        )
        self.apply_btn.setToolTip(
            "Write the dragged sections, then re-align this acapella.\n"
            "The instrumental is not changed."
        )

    def _mix_gains(self) -> tuple[float, str | None]:
        """Output gain, and which reference to add under it."""
        tl = self.timeline
        out = 1.0 if tl.audible("output") else 0.0
        hear_dem = tl.audible("demucs") and (
            self.with_demucs.isChecked() or tl.solo["demucs"]
        )
        if tl.stem == "inst":
            if hear_dem and self._data is not None and self._data.get("demucs_inst") is not None:
                return out, "dinst"
            if (
                not self.with_demucs.isChecked()
                and not any(tl.solo.values())
                and self._data is not None
                and self._data.get("aca") is not None
            ):
                return out, "aca"
            return out, None
        if hear_dem:
            return out, "vox"
        if not self.with_demucs.isChecked() and not any(tl.solo.values()) and self._data is not None:
            if self._data.get("inst") is not None:
                return out, "inst"
        return out, None

    def _sync_gains(self) -> None:
        aca, bed = self._mix_gains()
        self._play.set_gains(aca, bed or "")

    def _as_stereo(self, audio: np.ndarray | None) -> np.ndarray | None:
        if audio is None:
            return None
        if audio.ndim == 1:
            audio = np.repeat(audio[:, None], 2, axis=1)
        elif audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        else:
            audio = audio[:, :2]
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _match_play_rate(self, audio: np.ndarray | None) -> np.ndarray | None:
        if audio is None or self._data is None or self._play_sr in (0, int(self._data["sr"])):
            return audio
        return self._as_stereo(_to_sr(audio, int(self._data["sr"]), self._play_sr))

    def _ensure_aca_play(self) -> None:
        assert self._data is not None
        key = (self.timeline.stem, self._section_snapshot())
        if self._aca_play is not None and key == self._aca_key:
            return
        data = self._data
        if self.timeline.stem == "inst":
            src = data.get("inst")
        else:
            src = data.get("aca")
        if src is None:
            self._aca_play = None
            self._aca_key = key
            self._play.set_aca(None)
            return
        notes: list[str] = []
        moved = render_moved(
            src, data["sr"], self.timeline.sections, data["n_frames"], warnings=notes
        )
        if notes:
            self.status.setText(notes[0])
        moved = np.ascontiguousarray(self._match_play_rate(self._as_stereo(moved)), dtype=np.float32)
        self._aca_play = moved
        self._aca_key = key
        self._play.set_aca(moved)

    def _ensure_beds(self) -> None:
        if self._data is None:
            return
        if self._vox_st is None:
            vox = self._match_play_rate(self._as_stereo(self._data.get("vox")))
            self._vox_st = None if vox is None else np.ascontiguousarray(vox, dtype=np.float32)
            self._play.set_bed("vox", self._vox_st)
        if self._inst_st is None:
            inst = self._match_play_rate(self._as_stereo(self._data.get("inst")))
            self._inst_st = None if inst is None else np.ascontiguousarray(inst, dtype=np.float32)
            self._play.set_bed("inst", self._inst_st)
        if self._dinst_st is None:
            dinst = self._match_play_rate(self._as_stereo(self._data.get("demucs_inst")))
            self._dinst_st = None if dinst is None else np.ascontiguousarray(dinst, dtype=np.float32)
            self._play.set_bed("dinst", self._dinst_st)
        if self._aca_st is None:
            aca = self._match_play_rate(self._as_stereo(self._data.get("aca")))
            self._aca_st = None if aca is None else np.ascontiguousarray(aca, dtype=np.float32)
            self._play.set_bed("aca", self._aca_st)

    def _invalidate_aca_play(self) -> None:
        self._aca_play = None
        self._aca_key = None

    def _is_playing(self) -> bool:
        return self._play.playing

    def _set_transport_playing(self, playing: bool) -> None:
        self.transport_play.setIcon(_media_icon("pause" if playing else "play"))

    def _start_playback(self) -> None:
        assert self._data is not None
        self._play_sr = int(self._data["sr"])
        self._ensure_beds()
        self._ensure_aca_play()
        self._sync_gains()
        self._play.ensure_stream(self._play_sr)
        self._restore_cue()
        self._play.set_playing(True)
        if not self._tick.isActive():
            self._tick.start()
        self._set_transport_playing(True)

    def _pause_playback(self) -> None:
        self._play.set_playing(False)
        self._set_transport_playing(False)
        self._restore_cue()

    def _restore_cue(self) -> None:
        """Put the playhead back on the last click, so Play starts there again."""
        if self._cue is None:
            return
        self._play.seek(self._cue)
        self._play.flush()
        self.timeline.set_playhead(self._cue)
        self._show_time(self._cue)

    def _toggle_play(self) -> None:
        if self._busy or not HAS_MEDIA or self._data is None:
            return
        if self._is_playing():
            self._pause_playback()
            return
        self._start_playback()

    def _seek_relative(self, delta: float) -> None:
        if self._data is None or self._busy:
            return
        dur = float(self._data["duration"])
        t = max(0.0, min(dur, self.timeline.playhead + delta))
        self._seek(t)
        self._reveal_playhead()

    def _reveal_playhead(self) -> None:
        tl = self.timeline
        t = tl.playhead
        if tl.view_start - 1e-3 <= t <= tl.view_start + tl.view_span + 1e-3:
            return
        new_start = t - tl.view_span * 0.35
        tl.view_start = min(max(0.0, new_start), max(0.0, tl.duration - tl.view_span))
        tl.update()
        self._sync_scroll()

    def _on_drag_finished(self, *_args) -> None:
        self._invalidate_aca_play()
        if self._is_playing() and self._data is not None:
            self._ensure_aca_play()

    def _on_audio_tick(self) -> None:
        if self._data is None or self._play.is_held():
            return
        if self._play.consume_ended():
            self._set_transport_playing(False)
            if self._cue is None:
                self._play.seek(0.0)
                self.timeline.set_playhead(0.0)
                self._show_time(0.0)
            else:
                self._restore_cue()
            return
        if not self._is_playing():
            return
        t = self._play.position_seconds()
        self.timeline.set_playhead(t)
        self._show_time(t)

    def _seek(self, t: float) -> None:
        if self._data is not None:
            t = max(0.0, min(float(self._data["duration"]), t))
            self._cue = t
            self._play.seek(t)
        self.timeline.set_playhead(t)
        self._show_time(t)
        m, s = divmod(t, 60)
        self.status.setText(f"Playhead {int(m)}:{s:05.2f}")

    def _split_at_playhead(self) -> None:
        self._split_at(self.timeline.playhead)

    def _split_at(self, t: float) -> None:
        if self._data is None or self._busy or not self.timeline.editing():
            return
        t = max(0.0, min(float(self._data["duration"]), t))
        self.timeline.set_playhead(t)
        self._show_time(t)
        sections = self.timeline.sections
        idx = next(
            (i for i, sec in enumerate(sections) if sec.dst0 < t < sec.dst1),
            None,
        )
        if idx is None:
            self.status.setText("Put the playhead inside a section, then press S to split it.")
            return
        sec = sections[idx]
        if t - sec.dst0 < 0.05 or sec.dst1 - t < 0.05:
            self.status.setText("Move the playhead further inside the section before splitting.")
            return
        self._push_undo()
        span = max(1e-6, sec.out_dur)
        frac = (t - sec.dst0) / span
        src_cut = sec.src0 + frac * sec.dur
        left = Section(sec.src0, src_cut, sec.dst0)
        right = Section(src_cut, sec.src1, t)
        left_out, right_out = t - sec.dst0, sec.dst1 - t
        if abs(left_out - left.dur) > 1e-4:
            left.dst_dur = left_out
        if abs(right_out - right.dur) > 1e-4:
            right.dst_dur = right_out
        sections[idx : idx + 1] = [left, right]
        self._invalidate_aca_play()
        if self._is_playing():
            self._ensure_aca_play()
        self.timeline.update()
        self._sync_edit_buttons()
        m, s = divmod(t, 60)
        self.status.setText(
            f"Split section {idx + 1} at {int(m)}:{s:04.1f}. Drag either piece, including over other audio."
        )

    def _toggle_alignment(self, on: bool) -> None:
        self.timeline.show_alignment = bool(on)
        self.timeline.update()

    def _toggle_bars(self, on: bool) -> None:
        self.timeline.show_bars = bool(on)
        self.timeline._cue_tip = ""
        self.timeline.update()

    def _reset(self) -> None:
        initial = self._initial.get(self.timeline.stem, ())
        if self._section_snapshot() != initial:
            self._push_undo()
        self.timeline.sections = [section_from_snap(item) for item in initial]
        self._invalidate_aca_play()
        if self._is_playing():
            self._ensure_aca_play()
        self.timeline.update()
        self._sync_edit_buttons()
        name = "instrumental" if self.timeline.stem == "inst" else "acapella"
        self.status.setText(f"Sections restored to the current {name}.")

    def _show_time(self, t: float) -> None:
        self.time_lbl.setText(_format_time_ms(t))

    def _sync_transport(self) -> None:
        can = not self._busy and self._data is not None
        self.prev_btn.setEnabled(can)
        self.next_btn.setEnabled(can)
        self.stop_btn.setEnabled(can)
        self.transport_play.setEnabled(can and HAS_MEDIA)

    def _halt_audio(self) -> None:
        self._play.close()
        if HAS_MEDIA and getattr(self, "_tick", None) is not None:
            self._tick.stop()
        self._set_transport_playing(False)

    def _stop(self) -> None:
        if self._data is None or self._busy:
            return
        self._cue = None
        self._play.set_playing(False)
        self._play.seek(0.0)
        self._play.flush()
        self._set_transport_playing(False)
        self.timeline.set_playhead(0.0)
        self._show_time(0.0)
        self.status.setText("Playhead 0:00.00")

    def _on_volume(self, value: int) -> None:
        self._play.set_master(value / 100.0)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._sync_transport()
        self._sync_edit_buttons()
        self.timeline.setEnabled(not busy)

    def _apply(self) -> None:
        if self._data is None or self._busy or not self._sections_dirty() or not self.timeline.editing():
            return
        stem = self.timeline.stem
        if stem == "inst":
            dest: Path | None = self._data.get("inst_path")
            audio = self._data.get("inst")
            prompt = (
                "Write these section positions, then re-align that instrumental "
                "with élastique against the Demucs instrumental, match its loudness, "
                "and score it again. The acapella is left as it is.\n\n"
            )
        else:
            dest = self._data.get("aca_path")
            audio = self._data.get("aca")
            prompt = (
                "Write these section positions, then re-align that acapella "
                "with élastique against the Demucs vocal, match its loudness, "
                "and score it again. The instrumental is left as it is.\n\n"
            )
        if dest is None or audio is None:
            self.status.setText("This stem is missing.")
            return
        if self._tag_folders():
            folder_line = (
                "If the new score passes, the folder is renamed from _[fail] to _[pass]. "
                "If it fails, the other way. Close REAPER first if it is open."
            )
        else:
            folder_line = "The folder name is left as it is. Close REAPER first if it is open."
        answer = QMessageBox.question(
            self,
            "Apply section edit",
            prompt + f"{dest.name}\n" + folder_line,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self._on_apply_begin is not None and not self._on_apply_begin(self.folder.name, stem):
            QMessageBox.information(
                self,
                "Busy",
                "Wait for the current run to finish before applying this edit.",
            )
            return
        data = self._data
        bak = self.folder / "_before_section_edit"
        bak.mkdir(exist_ok=True)
        saved = bak / dest.name
        if not saved.exists():
            shutil.copy2(dest, saved)
        self._halt_audio()
        self._remember_sections()
        sections = [
            Section(s.src0, s.src1, s.dst0, s.dst_dur) for s in self.timeline.sections
        ]
        self._applied_stem = stem
        self._set_busy(True)
        self.status.setText("Starting re-process…")
        worker = _ApplyWorker(
            self.folder,
            dest,
            audio,
            data["sr"],
            data["n_frames"],
            sections,
            self._declick,
            stem,
            self._tag_folders(),
        )
        worker.step.connect(self._relay_apply_step)
        worker.finished_ok.connect(self._on_apply_done)
        worker.failed.connect(self._on_apply_failed)
        worker.finished.connect(lambda worker=worker: self._retire_thread(worker, "_apply_worker"))
        self._apply_worker = worker
        worker.start()

    def _start_loader(self, on_loaded) -> None:
        loader = _Loader(self.folder)
        loader.loaded.connect(on_loaded)
        loader.failed.connect(self._on_failed)
        loader.finished.connect(lambda loader=loader: self._retire_thread(loader, "_loader"))
        self._loader = loader
        loader.start()

    def _retire_thread(self, thread: QThread, attr: str) -> None:
        if getattr(self, attr) is thread:
            setattr(self, attr, None)
        thread.deleteLater()

    def _relay_apply_step(self, step_id: str) -> None:
        if self._on_apply_step is not None:
            self._on_apply_step(step_id)

    def _on_apply_done(self, payload: dict) -> None:
        try:
            if self._on_apply_scored is not None:
                self._on_apply_scored(payload)
        finally:
            if self._on_apply_end is not None:
                self._on_apply_end()
        self._invalidate_aca_play()
        self._apply_notes = str(payload.get("notes") or "")
        aca = str(payload.get("aca_verdict") or "").upper()
        inst = str(payload.get("inst_verdict") or "").upper()
        self.status.setText(
            f"Scored again. Acapella {aca or '—'}, instrumental {inst or '—'}. Reloading…"
        )
        self._apply_summary = (
            f"Re-aligned. Acapella {aca or '—'}, instrumental {inst or '—'}."
        )
        self._follow_renamed_folder(payload)
        self._start_loader(self._on_reloaded)

    def _tag_folders(self) -> bool:
        fn = self._tag_folders_enabled
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001 — a settings read must not block the edit
            return False

    def _follow_renamed_folder(self, payload: dict) -> None:
        moved = str(payload.get("moved_to") or "")
        if not moved:
            return
        new_folder = Path(moved)
        if not new_folder.is_dir() or new_folder == self.folder:
            return
        self.folder = new_folder
        self.folder_lbl.setText(new_folder.name)

    def _on_reloaded(self, data: dict) -> None:
        applied = getattr(self, "_applied_stem", "aca")
        other = "inst" if applied == "aca" else "aca"
        self._remember_sections()
        kept = None
        other_sections = self._section_bank.get(other) or []
        other_snap = tuple(s.snap() for s in other_sections)
        if other_snap != self._initial.get(other, ()):
            kept = (
                other,
                list(other_sections),
                list(self._undo.get(other, [])),
                self._initial.get(other, ()),
            )
        view = self.timeline.stem
        self._on_loaded(data)
        if kept is not None:
            stem, sections, undo, initial = kept
            self._section_bank[stem] = sections
            self._undo[stem] = undo
            self._initial[stem] = initial
        if view == "inst" and self._data is not None and self._data.get("inst") is not None:
            self._set_stem_choice("inst")
            self.timeline.set_stem("inst")
            self.timeline.sections = list(self._section_bank.get("inst") or [])
            self._apply_pair_chrome()
        self._set_busy(False)
        self.status.setText(getattr(self, "_apply_summary", "Re-aligned from the edited sections."))

    def _on_apply_failed(self, message: str) -> None:
        if self._on_apply_end is not None:
            self._on_apply_end()
        self._set_busy(False)
        self.status.setText(message)
        QMessageBox.warning(self, "Re-process failed", message)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._after_first_show)

    def _after_first_show(self) -> None:
        apply_window_corner_preference(self)
        if not self._center_pending:
            return
        self._center_pending = False
        screen = self.screen()
        if screen is None:
            return
        ag = screen.availableGeometry()
        g = self.frameGeometry()
        self.move(
            ag.x() + max(0, (ag.width() - g.width()) // 2),
            ag.y() + max(0, (ag.height() - g.height()) // 2),
        )

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is getattr(self, "folder_lbl", None) and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                import os

                os.startfile(self.folder)  # noqa: S606 — intentional Explorer open
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:  # noqa: N802
        apply = self._apply_worker
        loader = self._loader
        if self._busy or (apply is not None and apply.isRunning()):
            event.ignore()
            return
        if loader is not None and loader.isRunning():
            loader.wait()
            QApplication.processEvents()
        self._halt_audio()
        super().closeEvent(event)


def open_section_editor(
    folder: Path,
    parent: QWidget | None = None,
    *,
    declick: str = "rx",
    on_apply_begin=None,
    on_apply_step=None,
    on_apply_scored=None,
    on_apply_end=None,
    tag_folders=None,
) -> SectionEditor:
    # Stay a top-level window. Parenting onto the aligner creates a native
    # child that can take down the host when the frameless frame is shown.
    del parent
    win = SectionEditor(
        folder,
        declick=declick,
        on_apply_begin=on_apply_begin,
        on_apply_step=on_apply_step,
        on_apply_scored=on_apply_scored,
        on_apply_end=on_apply_end,
        tag_folders=tag_folders,
    )
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    win.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
    _OPEN.append(win)
    win.destroyed.connect(lambda _obj=None, w=win: _OPEN.remove(w) if w in _OPEN else None)
    win.show()
    win.raise_()
    win.activateWindow()
    return win
