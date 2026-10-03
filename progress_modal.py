"""Busy / progress modal — stepper inspired by stem-separation UIs, STEM-organizer theme."""
from __future__ import annotations

import math
import time
from typing import Optional, Sequence

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

COLORS = {
    "bg": "#1e1f26",
    "panel": "#262833",
    "panel2": "#2F3140",
    "fg": "#e6e8ef",
    "fg_dim": "#9aa0b4",
    "accent": "#7c5cff",
    "accent_hov": "#9077ff",
    "log_bg": "#15161c",
    "log_fg": "#d6dae8",
    "border": "#3a3d4d",
    "text_mute": "#7a8199",
    "done": "#7c5cff",
    "active": "#9077ff",
    "score": "#60A5FA",  # instrumental blue (tag)
    "final": "#7ee0a0",  # pass green (score / align)
    "acapella": "#a855f7",  # player acapella purple (scan)
    "facebook": "#1877F2",  # Demucs
    "silence": "#C8CCD8",  # light gray
    "beat": "#ecc990",  # master clock
    "loudness": "#ff7a7a",  # same red as a fail verdict
    "icon_mute": "#555866",
}

FONT_FAMILY = "Segoe UI"


def format_eta(seconds: float | None) -> str:
    """Right-hand progress label. None means there is not enough time yet."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "ETA —"
    whole = int(round(seconds))
    if whole < 60:
        return f"ETA {whole}s"
    minutes, sec = divmod(whole, 60)
    if minutes < 60:
        return f"ETA {minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"ETA {hours}h {minutes:02d}m"

# (step_id, short label, headline while active)
ALIGN_STEPS: list[tuple[str, str, str]] = [
    ("demucs", "SPLIT", "Preparing Demucs vocal and instrumental"),
    ("silence", "SILENCE", "Putting rests back in the acapella"),
    ("align", "ALIGN", "Stretching both stems to the Demucs splits"),
    ("loudness", "LOUDNESS", "Matching loudness to the Demucs splits"),
    ("score", "SCORE", "Scoring alignment"),
]

def edit_steps(stem: str) -> list[tuple[str, str, str]]:
    """Headlines for a section edit. ``stem`` is ``aca`` or ``inst``."""
    if stem == "inst":
        name, ref = "instrumental", "Demucs instrumental"
    else:
        name, ref = "acapella", "Demucs vocal"
    return [
        ("write", "WRITE", f"Writing the edited {name}"),
        ("align", "ALIGN", f"Warping the {name} to the {ref}"),
        ("loudness", "LOUDNESS", f"Matching {name} loudness"),
        ("score", "SCORE", f"Scoring the edited {name}"),
    ]

CHECK_STEPS: list[tuple[str, str, str]] = [
    ("scan", "SCAN", "Reading stems"),
    ("score", "SCORE", "Checking alignment"),
]

# Repair stays on the stretch. Renaming the folder is too brief to be its own step.
ALIGN_TRAILING = {
    "repair": "Correcting drift the first warp left behind",
}

def _step_color(step_id: str, *, is_last: bool) -> str:
    """Per-step colors for check and align steppers."""
    by_id = {
        "scan": COLORS["acapella"],
        "write": COLORS["silence"],
        "score": COLORS["final"],
        "tag": COLORS["score"],
        "demucs": COLORS["facebook"],
        "beat": COLORS["beat"],
        "silence": COLORS["silence"],
        "align": COLORS["final"],
        "loudness": COLORS["loudness"],
    }
    if step_id in by_id:
        return by_id[step_id]
    if is_last:
        return COLORS["score"]
    return COLORS["accent"]


ICON_BASE_SIZE = int(110 * 0.75)  # 75% of the previous full size

_SPLIT_AMPS = (0.32, 0.56, 0.94, 0.44, 0.76, 0.34)
_SILENCE_LEFT = (0.58, 0.96, 0.46)
_SILENCE_RIGHT = (0.74, 0.42, 0.9)
_ALIGN_AMPS = (0.36, 0.86, 0.48, 1.0, 0.4)
_LOUD_START = (0.42, 0.96, 0.34, 0.72, 0.5)
_LOUD_TARGET = 0.68


def _pen(color: str, width: float, alpha: int = 255) -> QPen:
    c = QColor(color)
    c.setAlpha(max(0, min(255, alpha)))
    pen = QPen(c)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw_bars(
    p: QPainter,
    color: str,
    cx: float,
    cy: float,
    amps: Sequence[float],
    spacing: float,
    max_h: float,
    *,
    width: float = 2.5,
    alpha: int = 255,
) -> None:
    if alpha <= 0:
        return
    p.setPen(_pen(color, width, alpha))
    p.setBrush(Qt.BrushStyle.NoBrush)
    x0 = cx - spacing * (len(amps) - 1) / 2
    for i, amp in enumerate(amps):
        h = max(2.6, float(amp) * max_h)
        x = x0 + i * spacing
        p.drawLine(QPointF(x, cy - h / 2), QPointF(x, cy + h / 2))


class _ProcessGlyph(QWidget):
    """Painted icon that acts out one align step.

    ``t`` runs 0 → 1: the before pose eases into the after pose.
    Pending steps hold t=0 in mute. Finished steps hold t=1 in color.
    """

    def __init__(self, step_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._step_id = step_id
        self._state = "pending"
        self._t = 0.0
        self.setFixedSize(ICON_BASE_SIZE, ICON_BASE_SIZE)
        self.setStyleSheet("background: transparent; border: none;")

    def set_visual(self, state: str, t: float) -> None:
        if state == "pending":
            t = 0.0
        elif state == "done":
            t = 1.0
        else:
            t = max(0.0, min(1.0, t))
        if state == self._state and abs(t - self._t) < 0.004:
            return
        self._state = state
        self._t = t
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_tile(p)
        scene = {
            "write": self._paint_silence,
            "demucs": self._paint_split,
            "beat": self._paint_beat,
            "silence": self._paint_silence,
            "align": self._paint_align,
            "loudness": self._paint_loudness,
            "score": self._paint_score,
        }.get(self._step_id)
        if scene is not None:
            scene(p)
        p.end()

    def _accent(self) -> str:
        if self._state == "pending":
            return COLORS["icon_mute"]
        return _step_color(self._step_id, is_last=False)

    def _muted(self) -> str:
        return COLORS["icon_mute"] if self._state == "pending" else COLORS["fg_dim"]

    def _paint_tile(self, p: QPainter) -> None:
        rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        clip = QPainterPath()
        clip.addRoundedRect(rect, 18, 18)
        p.setClipPath(clip)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#1E2028"))
        p.drawRoundedRect(rect, 18, 18)
        if self._state == "active":
            glow = QColor(self._accent())
            glow.setAlpha(28)
            p.setBrush(glow)
            p.drawRoundedRect(rect, 18, 18)

    def _paint_split(self, p: QPainter) -> None:
        t = self._t
        s = float(self.width())
        cx = cy = s / 2
        accent = self._accent()
        inst = self._muted() if self._state == "pending" else COLORS["score"]
        lift = 18.0 * t
        _draw_bars(p, inst, cx, cy + lift, _SPLIT_AMPS, 7.0, 15, width=2.3)
        _draw_bars(p, accent, cx, cy - lift, _SPLIT_AMPS, 7.0, 15, width=2.3)
        if t > 0.18:
            alpha = int(230 * min(1.0, (t - 0.18) / 0.55))
            p.setPen(_pen(accent, 2.15, alpha))
            reach = 14.0 + lift
            p.drawLine(QPointF(cx + 18, cy - reach), QPointF(cx - 18, cy + reach))

    def _paint_beat(self, p: QPainter) -> None:
        """Four beats in a bar. The downbeat is the tall mark. A playhead walks them."""
        t = self._t
        s = float(self.width())
        cx = cy = s / 2
        color = self._accent()
        heights = (26.0, 14.0, 14.0, 14.0)
        spacing = 14.0
        x0 = cx - spacing * 1.5
        play = (t * 4.0) % 4.0
        for i, height in enumerate(heights):
            hot = abs(play - i) < 0.55 or (i == 0 and play > 3.45)
            grown = height + (5.0 if hot and self._state == "active" else 0.0)
            width = 2.8 if i == 0 else 2.2
            alpha = 255 if hot or self._state != "active" else 150
            p.setPen(_pen(color, width, alpha))
            x = x0 + i * spacing
            p.drawLine(QPointF(x, cy + 10), QPointF(x, cy + 10 - grown))

    def _paint_silence(self, p: QPainter) -> None:
        t = self._t
        s = float(self.width())
        cx = cy = s / 2
        color = self._accent()
        shift = 10.0 * t
        _draw_bars(p, color, cx - 8 - shift, cy, _SILENCE_LEFT, 6.2, 24)
        _draw_bars(p, color, cx + 8 + shift, cy, _SILENCE_RIGHT, 6.2, 24)
        if t > 0.12:
            alpha = int(240 * min(1.0, (t - 0.12) / 0.45))
            p.setPen(_pen(color, 2.15, alpha))
            half = 2.0 + 8.0 * t
            p.drawLine(QPointF(cx - half, cy), QPointF(cx + half, cy))

    def _paint_align(self, p: QPainter) -> None:
        t = self._t
        s = float(self.width())
        cx = s / 2
        accent = self._accent()
        ref = self._muted()
        spacing = 6.5
        _draw_bars(p, ref, cx, s * 0.36, _ALIGN_AMPS, spacing, 16, width=2.2)
        moving_spacing = spacing * (1.0 + 0.22 * (1.0 - t))
        _draw_bars(
            p,
            accent,
            cx + (1.0 - t) * 12.0,
            s * 0.66,
            _ALIGN_AMPS,
            moving_spacing,
            16,
        )

    def _paint_loudness(self, p: QPainter) -> None:
        t = self._t
        s = float(self.width())
        cx = s / 2
        accent = self._accent()
        line = self._muted() if self._state == "pending" else "#ffb0b0"
        baseline = s * 0.74
        max_h = 38.0
        spacing = 8.0
        target_y = baseline - _LOUD_TARGET * max_h
        p.setPen(_pen(line, 1.35))
        pen = p.pen()
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([1.6, 2.4])
        p.setPen(pen)
        p.drawLine(QPointF(16, target_y), QPointF(s - 18, target_y))
        p.setPen(_pen(line, 1.35))
        p.drawLine(QPointF(s - 18, target_y - 3.5), QPointF(s - 18, target_y + 3.5))

        p.setPen(_pen(accent, 3.0))
        x0 = cx - spacing * (len(_LOUD_START) - 1) / 2
        for i, start in enumerate(_LOUD_START):
            h = (start + (_LOUD_TARGET - start) * t) * max_h
            x = x0 + i * spacing
            p.drawLine(QPointF(x, baseline), QPointF(x, baseline - h))

    def _paint_score(self, p: QPainter) -> None:
        """A ring fills, then a check draws. Not the align bars."""
        t = self._t
        s = float(self.width())
        cx = cy = s / 2
        accent = self._accent()
        radius = 22.0
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(self._muted(), 2.2))
        p.drawEllipse(rect)
        if self._state != "pending" and t > 0.01:
            span = int(round(-360 * 16 * min(1.0, t)))
            p.setPen(_pen(accent, 2.4))
            p.drawArc(rect, 90 * 16, span)
        if self._state == "pending" or t <= 0.5:
            return
        u = min(1.0, (t - 0.5) / 0.5)
        p.setPen(_pen(accent, 2.6))
        start = QPointF(cx - 8, cy + 1)
        knee = QPointF(cx - 2, cy + 8)
        end = QPointF(cx + 10, cy - 7)
        short = min(1.0, u / 0.4)
        p.drawLine(
            start,
            QPointF(
                start.x() + (knee.x() - start.x()) * short,
                start.y() + (knee.y() - start.y()) * short,
            ),
        )
        if u > 0.4:
            long = (u - 0.4) / 0.6
            p.drawLine(
                knee,
                QPointF(
                    knee.x() + (end.x() - knee.x()) * long,
                    knee.y() + (end.y() - knee.y()) * long,
                ),
            )


class _Stepper(QWidget):
    def __init__(
        self,
        steps: Sequence[tuple[str, str, str]],
        parent: Optional[QWidget] = None,
        *,
        with_icons: bool = False,
    ) -> None:
        super().__init__(parent)
        self._steps = list(steps)
        self._active = 0  # index of current; len(steps) means all done
        self._with_icons = with_icons
        self._phase = 0.0
        self._glyphs: list[Optional[_ProcessGlyph]] = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self._bars: list[QFrame] = []
        self._labels: list[QLabel] = []
        for sid, short, _head in self._steps:
            col = QVBoxLayout()
            col.setSpacing(8)
            col.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            glyph: Optional[_ProcessGlyph] = None
            if with_icons:
                glyph = _ProcessGlyph(sid)
                col.addWidget(glyph, 0, Qt.AlignmentFlag.AlignHCenter)
            self._glyphs.append(glyph)

            bar = QFrame()
            bar.setFixedHeight(3)
            bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            lbl = QLabel(short)
            lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            f = QFont(FONT_FAMILY)
            f.setPixelSize(11)
            f.setBold(True)
            f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
            lbl.setFont(f)
            col.addWidget(bar)
            col.addWidget(lbl)
            wrap = QWidget()
            wrap.setLayout(col)
            lay.addWidget(wrap, 1)
            self._bars.append(bar)
            self._labels.append(lbl)

        self._pulse = QTimer(self)
        self._pulse.setInterval(40)
        self._pulse.timeout.connect(self._on_pulse)
        if with_icons:
            self._sync_glyphs()
            self._pulse.start()

        self._paint()

    def hideEvent(self, event) -> None:  # noqa: N802
        if self._with_icons:
            self._pulse.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._with_icons and self._active < len(self._steps):
            self._pulse.start()

    def _motion(self) -> float:
        """0 at the before pose, 1 at the after pose, easing through the loop."""
        return (1.0 - math.cos(self._phase)) * 0.5

    def _on_pulse(self) -> None:
        if not self._with_icons or not self.isVisible():
            return
        n = len(self._steps)
        if self._active >= n:
            self._pulse.stop()
            return
        # ~3.6s for a full before → after → before cycle
        self._phase = (self._phase + 0.07) % (2 * math.pi)
        glyph = self._glyphs[self._active]
        if glyph is not None:
            glyph.set_visual("active", self._motion())

    def _sync_glyphs(self) -> None:
        if not self._with_icons:
            return
        n = len(self._steps)
        for i, glyph in enumerate(self._glyphs):
            if glyph is None:
                continue
            if self._active >= n or i < self._active:
                glyph.set_visual("done", 1.0)
            elif i == self._active:
                glyph.set_visual("active", self._motion())
            else:
                glyph.set_visual("pending", 0.0)

    def active_index(self) -> int:
        return self._active

    def set_active(self, index: int) -> None:
        self._active = max(0, min(index, len(self._steps)))
        self._phase = 0.0
        self._paint()
        if self._with_icons:
            self._sync_glyphs()
            if self._active < len(self._steps):
                if not self._pulse.isActive():
                    self._pulse.start()
            else:
                self._pulse.stop()

    def _paint(self) -> None:
        last_i = len(self._steps) - 1
        for i, (bar, lbl) in enumerate(zip(self._bars, self._labels)):
            sid = self._steps[i][0]
            tone = _step_color(sid, is_last=(i == last_i))
            if i < self._active:
                color = tone
                text = tone
            elif i == self._active and self._active < len(self._steps):
                color = tone
                text = COLORS["log_fg"]
            else:
                color = COLORS["border"]
                text = COLORS["text_mute"]
            bar.setStyleSheet(
                f"QFrame {{ background-color: {color}; border: none; border-radius: 1px; }}"
            )
            lbl.setStyleSheet(
                f"color: {text}; background: transparent; border: none;"
            )


class ProgressModal(QDialog):
    """Non-blocking busy card with a step strip. Parent stays usable after Hide."""

    stop_requested = pyqtSignal()

    def __init__(
        self,
        parent: QWidget,
        *,
        mode: str = "align",
        folder_total: int = 1,
        gaps_cut: bool = True,
        stem: str = "aca",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ProgressModal")
        self.setWindowTitle("Processing")
        self.setModal(False)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        self._mode = mode if mode in {"check", "edit", "align"} else "align"
        if self._mode == "check":
            self._steps = list(CHECK_STEPS)
        elif self._mode == "edit":
            self._steps = edit_steps(stem)
        else:
            self._steps = list(ALIGN_STEPS)
            if not gaps_cut:
                self._steps = [step for step in self._steps if step[0] != "silence"]
        self._id_to_index = {sid: i for i, (sid, _, _) in enumerate(self._steps)}
        self._folder_total = max(1, folder_total)
        self._folder_index = 1
        self._folder_name = ""
        self._overlay: Optional[QWidget] = None
        self._closing = False
        self._within = 0.0
        self._song_started: float | None = None
        self._song_durations: list[float] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)

        shell = QFrame()
        shell.setObjectName("ProgressCard")
        shell.setStyleSheet(
            f"""
            QFrame#ProgressCard {{
                background-color: {COLORS["log_bg"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 16px;
            }}
            """
        )
        body = QVBoxLayout(shell)
        body.setContentsMargins(36, 28, 36, 28)
        body.setSpacing(0)

        self._stepper = _Stepper(
            self._steps, with_icons=(self._mode in {"align", "edit"})
        )
        body.addWidget(self._stepper)
        body.addSpacing(28)

        self._headline = QLabel("Starting…")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._headline.setWordWrap(True)
        hf = QFont(FONT_FAMILY)
        hf.setPixelSize(22)
        hf.setBold(True)
        self._headline.setFont(hf)
        self._headline.setStyleSheet(
            f"color: {COLORS['log_fg']}; background: transparent; border: none;"
        )
        body.addWidget(self._headline)
        body.addSpacing(10)

        self._folder_lbl = QLabel("")
        self._folder_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._folder_lbl.setWordWrap(True)
        ff = QFont(FONT_FAMILY)
        ff.setPixelSize(12)
        self._folder_lbl.setFont(ff)
        self._folder_lbl.setStyleSheet(
            f"color: {COLORS['text_mute']}; background: transparent; border: none;"
        )
        body.addWidget(self._folder_lbl)
        body.addSpacing(22)

        job_row = QHBoxLayout()
        job_row.setContentsMargins(0, 0, 0, 0)
        job_row.setSpacing(8)
        self._job = QProgressBar()
        self._job.setRange(0, 100)
        self._job.setValue(0)
        self._job.setTextVisible(False)
        self._job.setFixedHeight(3)
        self._job.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._job.setToolTip("How far the current song has got.")
        self._job.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                border-radius: 1px;
                background-color: {COLORS["panel2"]};
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS["accent"]};
                border-radius: 1px;
            }}
            """
        )
        self._job_pct = QLabel("0%")
        self._job_pct.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        pct_font = QFont(FONT_FAMILY)
        pct_font.setPixelSize(11)
        pct_font.setBold(True)
        self._job_pct.setFont(pct_font)
        self._job_pct.setStyleSheet(
            f"color: {COLORS['log_fg']}; background: transparent; border: none;"
        )
        readout = QFontMetrics(pct_font)
        self._job_pct.setFixedWidth(readout.horizontalAdvance("100%") + 2)
        self._job_eta = QLabel("ETA —")
        self._job_eta.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )
        self._job_eta.setFont(pct_font)
        self._job_eta.setStyleSheet(
            f"color: {COLORS['text_mute']}; background: transparent; border: none;"
        )
        self._job_eta.setFixedWidth(readout.horizontalAdvance("ETA 59m 59s") + 2)
        self._job_eta.setToolTip("Estimated time left on this song.")
        job_row.addWidget(self._job_pct, 0, Qt.AlignmentFlag.AlignVCenter)
        job_row.addWidget(self._job, 1, Qt.AlignmentFlag.AlignVCenter)
        job_row.addWidget(self._job_eta, 0, Qt.AlignmentFlag.AlignVCenter)
        body.addLayout(job_row)
        self._eta_timer = QTimer(self)
        self._eta_timer.setInterval(1000)
        self._eta_timer.timeout.connect(self._refresh_file_eta)
        self._eta_timer.start()
        body.addSpacing(16)

        btns = QHBoxLayout()
        btns.setSpacing(12)
        self.hide_btn = QPushButton("Hide")
        self.hide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hide_btn.setMinimumHeight(36)
        self.hide_btn.setStyleSheet(self._btn_style(outline=COLORS["border"], fg=COLORS["log_fg"]))
        self.hide_btn.clicked.connect(self._on_hide)

        self.stop_btn = QPushButton("Stop after this song")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setStyleSheet(
            self._btn_style(outline=COLORS["accent"], fg=COLORS["accent"])
        )
        self.stop_btn.clicked.connect(self._on_stop)
        if self._mode == "edit":
            self.stop_btn.hide()

        btns.addWidget(self.hide_btn, 1)
        btns.addWidget(self.stop_btn, 1)
        body.addLayout(btns)

        outer.addWidget(shell)
        self.setFixedWidth(640 if len(self._steps) >= 5 else 520)
        self.adjustSize()

    @staticmethod
    def _btn_style(*, outline: str, fg: str) -> str:
        return f"""
        QPushButton {{
            background-color: transparent;
            color: {fg};
            border: 1px solid {outline};
            border-radius: 18px;
            padding: 8px 16px;
            font-family: "{FONT_FAMILY}";
            font-size: 13px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {COLORS["panel2"]};
        }}
        QPushButton:disabled {{
            color: {COLORS["text_mute"]};
            border-color: {COLORS["border"]};
        }}
        """

    def open_over(self, host: QWidget) -> None:
        self._show_overlay(host)
        hg = host.frameGeometry()
        self.adjustSize()
        dg = self.frameGeometry()
        self.move(
            hg.x() + max(0, (hg.width() - dg.width()) // 2),
            hg.y() + max(0, (hg.height() - dg.height()) // 2),
        )
        self.show()
        self.raise_()
        self.activateWindow()

    def reveal(self, host: QWidget) -> None:
        """Show the card again after Hide, without resetting the current stage."""
        if self._closing:
            return
        self.open_over(host)

    def _show_overlay(self, host: QWidget) -> None:
        self._clear_overlay()
        overlay = QWidget(host)
        overlay.setObjectName("ProgressDimmer")
        overlay.setStyleSheet(
            "QWidget#ProgressDimmer { background-color: rgba(21, 22, 28, 160); }"
        )
        overlay.setGeometry(host.rect())
        overlay.show()
        overlay.raise_()
        self._overlay = overlay

    def _clear_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None

    def set_folder(self, name: str, index: int, total: int) -> None:
        self._folder_name = name
        self._folder_index = max(1, index)
        self._folder_total = max(1, total)
        # Reset step strip for the new song
        self._stepper.set_active(0)
        first = self._steps[0]
        self._headline.setText(first[2])
        self._refresh_folder_label()
        self._song_started = time.monotonic()
        self._set_within(0.0)

    def set_step(self, step_id: str) -> None:
        if step_id == "repair" and self._mode == "edit":
            self._headline.setText(ALIGN_TRAILING["repair"])
            self._refresh_folder_label()
            return
        if step_id == "repair" and self._mode == "align":
            # The correction is still the stretch. Loudness has not started.
            self._headline.setText(ALIGN_TRAILING["repair"])
            self._refresh_folder_label()
            align_idx = self._id_to_index.get("align")
            if align_idx is not None:
                self._stepper.set_active(align_idx)
            self._note_step("align")
            return

        idx = self._id_to_index.get(step_id)
        if idx is None:
            return
        self._stepper.set_active(idx)
        self._headline.setText(self._steps[idx][2])
        self._refresh_folder_label()
        self._note_step(step_id)

    def note_song_done(self) -> None:
        if self._song_started is not None:
            self._song_durations.append(time.monotonic() - self._song_started)
            self._song_started = None
        self._set_within(1.0)

    def fraction(self) -> float:
        return self._within

    def _progress_step_ids(self) -> list[str]:
        ids = [sid for sid, _, _ in self._steps]
        if self._mode == "align":
            if "score" not in ids:
                ids.append("score")
        return ids

    def _note_step(self, step_id: str) -> None:
        ids = self._progress_step_ids()
        try:
            index = ids.index(step_id)
        except ValueError:
            return
        # Count the stage in progress, so scoring is near the end of this song.
        self._set_within((index + 1) / len(ids))

    def _set_within(self, within: float) -> None:
        within = max(0.0, min(1.0, within))
        self._within = within
        pct = max(0, min(100, int(round(100.0 * within))))
        self._job.setValue(pct)
        self._job_pct.setText(f"{pct}%")
        self._refresh_file_eta()

    def _file_eta_sec(self) -> float | None:
        if self._within >= 0.999:
            return 0.0
        if self._song_started is None:
            return None
        elapsed = time.monotonic() - self._song_started
        if self._song_durations:
            typical = sum(self._song_durations) / len(self._song_durations)
            remaining = typical - elapsed
            if remaining >= 1.0:
                return remaining
        if self._within < 0.05 or elapsed < 1.0:
            return None
        remaining = elapsed * (1.0 - self._within) / self._within
        if remaining < 1.0:
            return None
        return remaining

    def _refresh_file_eta(self) -> None:
        self._job_eta.setText(format_eta(self._file_eta_sec()))

    def mark_stopping(self) -> None:
        self.stop_btn.setEnabled(False)
        self._headline.setText("Stop requested — finishing the current song…")

    def _refresh_folder_label(self) -> None:
        if self._folder_total > 1:
            prefix = f"Song {self._folder_index} of {self._folder_total}"
        else:
            prefix = "This song"
        name = self._folder_name or "…"
        self._folder_lbl.setText(f"{prefix}  ·  {name}")

    def _on_hide(self) -> None:
        self._clear_overlay()
        self.hide()

    def _on_stop(self) -> None:
        self.mark_stopping()
        self.stop_requested.emit()

    def finish(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._eta_timer.stop()
        self._clear_overlay()
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._clear_overlay()
        super().closeEvent(event)


def steps_for_mode(mode: str) -> list[tuple[str, str, str]]:
    return list(CHECK_STEPS if mode == "check" else ALIGN_STEPS)
