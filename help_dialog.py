"""Help "?" icon + About dialog — STEM-organizer Classify style (PyQt6)."""
from __future__ import annotations

import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Union

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# Tokens match align_gui / STEM-organizer theme.py
COLORS = {
    "bg": "#1e1f26",
    "panel": "#262833",
    "panel2": "#2F3140",
    "fg_dim": "#9aa0b4",
    "accent": "#7c5cff",
    "accent_hov": "#9077ff",
    "log_bg": "#15161c",
    "log_fg": "#d6dae8",
    "border": "#3a3d4d",
    "text_mute": "#7a8199",
}

STEM_CHIP = {
    "acapella": "#a855f7",
    "vocals": "#a855f7",
    "instrumental": "#60A5FA",
    "original": "#9aa0b4",
    "pass": "#7ee0a0",
    "fail": "#ff7a7a",
    "skip": "#7a8199",
}

FONT_FAMILY = "Segoe UI"
APP_VERSION = "1.0.0"

HELP_DIALOG_WIDTH = 820
HELP_ABOUT_ICON_PX = 112
HELP_HEADING_PX = 21
HELP_BODY_PX = 14
HELP_SECTION_TITLE_PX = 12
HELP_FOOTER_PX = 12

HelpSectionBody = Union[str, Sequence[str]]


class InfoIcon(QWidget):
    """Small "?" badge — clickable, hover-bright ring (STEM-organizer InfoIcon)."""

    def __init__(
        self,
        parent: QWidget,
        on_click: Optional[Callable[[], None]] = None,
        *,
        size: int = 18,
    ) -> None:
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Open help for this panel.")
        self._on_click = on_click
        self._hover = False

    def set_on_click(self, cb: Callable[[], None]) -> None:
        self._on_click = cb

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._on_click is not None:
            self._on_click()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        dim = QColor(COLORS["fg_dim"])
        bright = QColor(COLORS["log_fg"])
        ring = bright if self._hover else dim
        text = bright if self._hover else dim
        w, h = self.width(), self.height()
        r = min(w, h) / 2 - 1
        from PyQt6.QtCore import QRectF

        rect = QRectF(w / 2 - r, h / 2 - r, 2 * r, 2 * r)
        p.setPen(QPen(ring, 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(rect)
        p.translate(0, -2)
        p.setPen(text)
        font = p.font()
        font.setBold(True)
        font.setPointSize(max(7, int(r * 1.1)))
        p.setFont(font)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "?")


def _style_label(lbl: QLabel, px: int, color: str, *, bold: bool = False) -> None:
    weight = "600" if bold else "400"
    lbl.setStyleSheet(
        f"""
        QLabel {{
            color: {color};
            font-family: "{FONT_FAMILY}";
            font-size: {px}px;
            font-weight: {weight};
            background: transparent;
            border: none;
        }}
        """
    )


def _stem_chip(parent: QWidget, label: str, *, min_width: int = 0) -> QLabel:
    key = label.strip().lower()
    bg = STEM_CHIP.get(key, COLORS["panel2"])
    font = QFont("Arial")
    font.setBold(True)
    font.setPixelSize(11)
    fm = QFontMetrics(font)
    w = max(min_width, fm.horizontalAdvance(key) + 16)
    h = max(fm.height() + 4, 17)
    chip = QLabel(key, parent)
    chip.setObjectName("HelpStemChip")
    chip.setFont(font)
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    chip.setFixedSize(w, h)
    chip.setStyleSheet(
        f"""
        QLabel#HelpStemChip {{
            background-color: {bg};
            color: {COLORS["log_fg"]};
            border: none;
            padding: 0px;
        }}
        """
    )
    return chip


def _parse_legend(body: HelpSectionBody) -> Optional[list[tuple[str, str]]]:
    if isinstance(body, str):
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    else:
        lines = [str(ln).strip() for ln in body if str(ln).strip()]
    if not lines:
        return None
    out: list[tuple[str, str]] = []
    for ln in lines:
        if " — " not in ln and " - " not in ln:
            return None
        sep = " — " if " — " in ln else " - "
        stem, desc = ln.split(sep, 1)
        stem = stem.strip().lower()
        if stem not in STEM_CHIP:
            return None
        out.append((stem, desc.strip()))
    return out


def _section_card(
    parent: QWidget,
    title: str,
    body: HelpSectionBody,
    *,
    text_max: int,
) -> QFrame:
    card = QFrame(parent)
    card.setObjectName("HelpSection")
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    card.setStyleSheet(
        f"""
        QFrame#HelpSection {{
            background-color: {COLORS["panel2"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 10px;
        }}
        """
    )
    lay = QVBoxLayout(card)
    lay.setContentsMargins(14, 10, 14, 11)
    lay.setSpacing(4)

    hdr = QLabel(title.upper())
    _style_label(hdr, HELP_SECTION_TITLE_PX, COLORS["accent_hov"], bold=True)
    lay.addWidget(hdr)

    legend = _parse_legend(body)
    if legend is not None:
        chip_w = max(
            QFontMetrics(QFont("Arial", 11, QFont.Weight.Bold)).horizontalAdvance(s)
            for s, _ in legend
        ) + 16
        desc_max = max(120, text_max - chip_w - 12)
        for stem, desc in legend:
            row = QHBoxLayout()
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(12)
            row.addWidget(_stem_chip(card, stem, min_width=chip_w), 0, Qt.AlignmentFlag.AlignTop)
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setMaximumWidth(desc_max)
            _style_label(desc_lbl, HELP_BODY_PX, COLORS["fg_dim"])
            row.addWidget(desc_lbl, 1)
            lay.addLayout(row)
    else:
        if isinstance(body, str):
            text = body
        else:
            text = "\n".join(str(ln) for ln in body)
        body_lbl = QLabel()
        body_lbl.setWordWrap(True)
        body_lbl.setMaximumWidth(text_max)
        body_lbl.setTextFormat(Qt.TextFormat.RichText)
        body_lbl.setOpenExternalLinks(True)
        # Keep bullet-ish lines; convert plain newlines
        html_lines = []
        for ln in text.splitlines():
            html_lines.append(ln if ln.startswith("•") or ln.startswith("<") else ln)
        dim = COLORS["fg_dim"]
        body_lbl.setText(
            f'<div style="color:{dim}; font-family:{FONT_FAMILY}; '
            f'font-size:{HELP_BODY_PX}px; line-height:1.35;">'
            + "<br/>".join(html_lines)
            + "</div>"
        )
        _style_label(body_lbl, HELP_BODY_PX, COLORS["fg_dim"])
        lay.addWidget(body_lbl)
    return card


@contextmanager
def _dim_behind(parent: QWidget) -> Iterator[None]:
    host = parent.window() if parent is not None else None
    overlay: Optional[QWidget] = None
    if host is not None:
        overlay = QWidget(host)
        overlay.setObjectName("HelpDimmer")
        overlay.setStyleSheet(
            f"QWidget#HelpDimmer {{ background-color: rgba(21, 22, 28, 160); }}"
        )
        overlay.setGeometry(host.rect())
        overlay.show()
        overlay.raise_()
    try:
        yield
    finally:
        if overlay is not None:
            overlay.hide()
            overlay.deleteLater()


class _RepoLink(QLabel):
    def __init__(self, text: str, url: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._url = url
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        _style_label(self, HELP_BODY_PX, COLORS["accent"])

    def enterEvent(self, event) -> None:  # noqa: N802
        _style_label(self, HELP_BODY_PX, COLORS["accent_hov"])
        f = self.font()
        f.setUnderline(True)
        self.setFont(f)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        _style_label(self, HELP_BODY_PX, COLORS["accent"])
        f = self.font()
        f.setUnderline(False)
        self.setFont(f)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._url:
            webbrowser.open(self._url)
        super().mousePressEvent(event)


def show_help_dialog(
    parent: QWidget,
    *,
    title: str = "Help",
    heading: str = "",
    version_line: str = "",
    intro: str = "",
    sections: Optional[Sequence[tuple[str, HelpSectionBody]]] = None,
    footer_note: str = "Hover over individual controls for more detail.",
    header_icon: Optional[Path] = None,
    repo_url: Optional[str] = None,
    width: int = HELP_DIALOG_WIDTH,
) -> None:
    """STEM-organizer Classify-style About / help popup."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setModal(True)
    dlg.setWindowFlags(
        Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint
    )
    dlg.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    host = parent.window() if parent is not None else None
    dlg_w = int(width)
    if host is not None:
        dlg_w = min(dlg_w, max(640, host.frameGeometry().width() - 80))
    dlg.setFixedWidth(dlg_w)
    text_max = max(480, dlg_w - 24 - 44 - 10 - 16 - 28)

    outer = QVBoxLayout(dlg)
    outer.setContentsMargins(12, 12, 12, 12)

    shell = QFrame()
    shell.setObjectName("HelpCard")
    shell.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
    shell.setStyleSheet(
        f"""
        QFrame#HelpCard {{
            background-color: {COLORS["panel"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 14px;
        }}
        """
    )
    layout = QVBoxLayout(shell)
    layout.setContentsMargins(22, 28, 22, 26)
    layout.setSpacing(0)

    body_host = QWidget()
    body_host.setObjectName("HelpScrollBody")
    body_host.setStyleSheet("QWidget#HelpScrollBody { background: transparent; }")
    body_lay = QVBoxLayout(body_host)
    body_lay.setContentsMargins(0, 0, 16, 0)
    body_lay.setSpacing(0)

    # Header sits on the card, not in the scrollbar column, so the icon and
    # lines share the card's center. The scroll body is inset on the right.
    header = QWidget()
    header.setStyleSheet("background: transparent; border: none;")
    header_lay = QVBoxLayout(header)
    header_lay.setContentsMargins(0, 0, 0, 0)
    header_lay.setSpacing(0)

    if header_icon is not None and header_icon.exists():
        pix = QPixmap(str(header_icon))
        if not pix.isNull():
            scaled = pix.scaled(
                HELP_ABOUT_ICON_PX,
                HELP_ABOUT_ICON_PX,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_lbl = QLabel(header)
            icon_lbl.setPixmap(scaled)
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            icon_lbl.setStyleSheet("background: transparent; border: none;")
            dlg._help_icon_pix = scaled  # type: ignore[attr-defined]
            header_lay.addWidget(icon_lbl, 0, Qt.AlignmentFlag.AlignHCenter)
            header_lay.addSpacing(6)

    if heading:
        head = QLabel(heading, header)
        head.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        _style_label(head, HELP_HEADING_PX, COLORS["log_fg"], bold=True)
        header_lay.addWidget(head, 0, Qt.AlignmentFlag.AlignHCenter)

    if version_line:
        header_lay.addSpacing(8)
        ver = QLabel(version_line, header)
        ver.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        _style_label(ver, 12, COLORS["log_fg"], bold=True)
        header_lay.addWidget(ver, 0, Qt.AlignmentFlag.AlignHCenter)

    if intro:
        header_lay.addSpacing(4)
        ilbl = QLabel(intro, header)
        ilbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        _style_label(ilbl, HELP_BODY_PX, COLORS["fg_dim"])
        header_lay.addWidget(ilbl, 0, Qt.AlignmentFlag.AlignHCenter)

    if repo_url:
        header_lay.addSpacing(4)
        header_lay.addWidget(
            _RepoLink("View on GitHub", repo_url, parent=header),
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )

    if header_lay.count():
        layout.addWidget(header)
        layout.addSpacing(22)

    for i, (sec_title, sec_body) in enumerate(sections or []):
        if i:
            body_lay.addSpacing(10)
        body_lay.addWidget(_section_card(body_host, sec_title, sec_body, text_max=text_max))

    scroll = QScrollArea()
    scroll.setWidget(body_host)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    scroll.setStyleSheet(
        f"""
        QScrollArea {{ background: transparent; border: none; }}
        QScrollBar:vertical {{
            background: {COLORS["panel"]};
            width: 10px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {COLORS["border"]};
            border-radius: 4px;
            min-height: 24px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """
    )
    layout.addWidget(scroll)

    footer = QHBoxLayout()
    footer.setContentsMargins(0, 40, 16, 0)
    footer.setSpacing(12)
    if footer_note:
        fn = QLabel(footer_note)
        _style_label(fn, HELP_FOOTER_PX, COLORS["fg_dim"])
        footer.addWidget(fn, 1)
    else:
        footer.addStretch(1)

    close_btn = QPushButton("Close")
    close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
    close_btn.setMinimumWidth(72)
    close_btn.setMinimumHeight(30)
    close_btn.setStyleSheet(
        f"""
        QPushButton {{
            background-color: {COLORS["accent"]};
            color: {COLORS["log_fg"]};
            border: 1px solid {COLORS["accent_hov"]};
            border-radius: 6px;
            padding: 6px 16px;
            font-family: "{FONT_FAMILY}";
            font-size: 12px;
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {COLORS["accent_hov"]}; }}
        """
    )
    close_btn.clicked.connect(dlg.accept)
    footer.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.addLayout(footer)
    outer.addWidget(shell)

    body_host.adjustSize()
    content_h = max(body_host.sizeHint().height(), body_host.minimumSizeHint().height())
    max_body = 820
    if host is not None:
        max_body = max(400, int(host.frameGeometry().height() * 0.92) - 120)
    scroll.setFixedHeight(min(content_h + 4, max_body))
    dlg.adjustSize()

    if host is not None:
        hg = host.frameGeometry()
        dg = dlg.frameGeometry()
        dlg.move(
            hg.x() + max(0, (hg.width() - dg.width()) // 2),
            hg.y() + max(0, (hg.height() - dg.height()) // 2),
        )

    with _dim_behind(parent):
        dlg.exec()


def show_align_help(parent: QWidget) -> None:
    """About / help for Audio Aligner (Classify-tab equivalent)."""
    root = Path(__file__).resolve().parent
    logo = root / "icon.png"
    if not logo.exists():
        logo = root / "logo.ico"

    show_help_dialog(
        parent,
        title="About Audio Aligner",
        heading="Audio Aligner",
        version_line=f"v{APP_VERSION}",
        intro="Split it. Sync it. Match it.",
        header_icon=logo if logo.exists() else None,
        sections=[
            (
                "What it does",
                [
                    "Audio Aligner takes song folders that already include an original mix plus backup stems, "
                    "and brings the vocals and instrumental into sync with that mix.",
                    "Late starts, drifting tempo, and missing rests in the acapella get corrected automatically — "
                    "without the pumping, gating, or robotic stretch artifacts you hear from crude tools.",
                    "When a song lines up, you can audition it in the player. The folder is renamed in place with _[pass] or _[fail].",
                ],
            ),
            (
                "How it works",
                [
                    "1. Listens to how your stems sit against the original — where they start, and whether they drift",
                    "2. When silences were cut between vocal phrases, puts those rests back. "
                    "When the rests are still in the file, that step is skipped and the vocal follows the instrumental clock",
                    "3. Gently time-aligns both stems so they stay in lockstep with the mix (pitch stays natural)",
                    "4. Balances loudness so vocals and instrumental feel like they belong in the same mix",
                    "5. Double-checks the result and renames the folder _[pass] or _[fail] in place",
                ],
            ),
            (
                "Engines",
                [
                    "• REAPER élastique (recommended) — studio-grade stretch for the cleanest, most musical result",
                    "• Rubber Band — solid built-in alternative when REAPER isn’t available",
                ],
            ),
            (
                "De-click",
                [
                    "• Auto-detect RX 11 De-click — cleans clicks on the vocal when that plugin is installed. If it is not, the acapella is left as the stretch wrote it",
                    "• Off — leaves the acapella as the stretch wrote it",
                    "RX runs after loudness, so the level match does not raise the click again.",
                    "The instrumental is not de-clicked. Drum attacks read as clicks.",
                ],
            ),
            (
                "What each folder needs",
                [
                    "acapella — Your vocal stem (kept safe under _backup_before_align)",
                    "instrumental — Your instrumental stem (same backup folder)",
                    "original — The full mix that everything should lock to (at the song folder root)",
                ],
            ),
            (
                "Results at a glance",
                [
                    "pass — Ready to use — timing and levels look good",
                    "fail — Still off — worth a listen and a re-run with different settings",
                    "skip — Missing a stem or the original, so it couldn’t be processed",
                ],
            ),
            (
                "Useful controls",
                [
                    "Check alignment scores the stems as they are now: pass if they sit inside the margins, fail if they do not. It does not warp. "
                    "Dry run previews what would happen without changing your files. "
                    "Silences cut between vocal phrases is on for acapellas that had the rests removed. "
                    "Turn it off when the vocal still has its original rests: phrase placement is skipped and the vocal follows the instrumental. "
                    "Tag folder _[pass] / _[fail] renames each song in place after the check. "
                    "The Aca and Inst columns are the acapella against the Demucs vocal and the instrumental against the Demucs instrumental. "
                    "The folder is tagged fail when either stem fails. "
                    "P and F in the player rename that song folder; those two columns stay the measured checks. "
                    "The bottom bar then shows how many are marked fail, and accuracy is the rest. "
                    "The thresholds under Options set how strict the final “pass” check is — "
                    "tighter values mean fewer false wins; looser values forgive harder material. "
                    "Maximum front pad covers songs that start much later than the original.",
                ],
            ),
        ],
    )
