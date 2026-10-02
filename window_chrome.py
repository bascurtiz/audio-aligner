"""Frameless chrome for Audio Aligner — matches STEM-organizer / player look.

PyQt6 port of the essentials from stem_organizer.widgets.titlebar:
custom title bar, rounded corners (SetWindowRgn), dark fill, edge resize.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QMouseEvent, QPalette
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

# Match STEM-organizer theme tokens
COLORS = {
    "bg": "#1e1f26",
    "panel2": "#2F3140",
    "fg": "#e6e8ef",
    "fg_dim": "#9aa0b4",
    "danger": "#e25c5c",
    "log_fg": "#d6dae8",
    "border": "#3a3d4d",
}
TITLE_BAR_HEIGHT = 36
TITLE_ICON_SIZE = 22
TITLE_LABEL_FONT_PX = 13
FONT_FAMILY = "Segoe UI"
WINDOW_CORNER_RADIUS = 12
WIN_DEFAULT_W = 1280
WIN_DEFAULT_H = 1018
WIN_MIN_W = 1040
WIN_MIN_H = 720
RESIZE_BORDER = 10

if sys.platform == "win32":
    try:
        _user32 = ctypes.WinDLL("user32")
        _gdi32 = ctypes.WinDLL("gdi32")
    except OSError:
        _user32 = None
        _gdi32 = None
    try:
        _dwmapi = ctypes.WinDLL("dwmapi")
    except OSError:
        _dwmapi = None
else:
    _user32 = None
    _gdi32 = None
    _dwmapi = None

DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWCP_DONOTROUND = 1
DWMWCP_ROUND = 2
WM_NCHITTEST = 0x0084
WM_NCCALCSIZE = 0x0083
WM_NCACTIVATE = 0x0086
WM_NCLBUTTONDBLCLK = 0x00A3
WVR_REDRAW = 0x0100
HTCAPTION = 2
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17
GWL_STYLE = -16
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
SWP_NOACTIVATE = 0x0010

_GLYPH_MIN = "\u2212"
_GLYPH_MAX = "\u25a1"
_GLYPH_RESTORE = "\u2750"
_GLYPH_CLOSE = "\u00d7"


def prepare_dark_frameless_chrome(window: QWidget) -> None:
    bg = QColor(COLORS["bg"])
    pal = window.palette()
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        pal.setColor(group, QPalette.ColorRole.Window, bg)
        pal.setColor(group, QPalette.ColorRole.Base, bg)
    window.setPalette(pal)
    window.setAutoFillBackground(True)
    window.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)


def is_window_filled(window: QWidget) -> bool:
    if getattr(window, "_custom_maximized", False):
        return True
    try:
        if window.isMaximized():
            return True
    except Exception:
        pass
    return False


def _win_colorref_from_hex(hex_color: str) -> int:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b << 16) | (g << 8) | r


def apply_rounded_window_region(
    window: QWidget,
    *,
    maximized: Optional[bool] = None,
    radius: int = WINDOW_CORNER_RADIUS,
) -> None:
    if _user32 is None or _gdi32 is None or sys.platform != "win32":
        return
    if getattr(window, "_rounding_corners", False):
        return
    try:
        hwnd = int(window.winId())
        if hwnd == 0:
            return
        cur = wintypes.RECT()
        if _user32.GetWindowRect(hwnd, ctypes.byref(cur)):
            w = max(int(cur.right - cur.left), 1)
            h = max(int(cur.bottom - cur.top), 1)
        else:
            dpr = float(window.devicePixelRatioF())
            w = max(int(round(window.width() * dpr)), 1)
            h = max(int(round(window.height() * dpr)), 1)
        if w < 2 or h < 2:
            return
        is_max = is_window_filled(window) if maximized is None else bool(maximized)
        dpr = float(window.devicePixelRatioF())
        pr = max(1, int(round(radius * dpr))) if radius > 0 else 0
        key = (w, h, is_max, int(pr), "rgn")
        if getattr(window, "_round_corner_key", None) == key:
            return
        window._rounding_corners = True  # type: ignore[attr-defined]
        try:
            if is_max or radius <= 0:
                _user32.SetWindowRgn(hwnd, 0, True)
            else:
                hrgn = _gdi32.CreateRoundRectRgn(0, 0, w + 1, h + 1, pr * 2, pr * 2)
                if not hrgn:
                    return
                if not _user32.SetWindowRgn(hwnd, hrgn, True):
                    _gdi32.DeleteObject(hrgn)
                    return
            window._round_corner_key = key  # type: ignore[attr-defined]
        finally:
            window._rounding_corners = False  # type: ignore[attr-defined]
    except Exception:
        try:
            window._rounding_corners = False  # type: ignore[attr-defined]
        except Exception:
            pass


def apply_window_corner_preference(window: QWidget, radius: int = WINDOW_CORNER_RADIUS) -> None:
    if sys.platform != "win32":
        return
    try:
        if not window.isVisible():
            return
    except Exception:
        return
    is_max = is_window_filled(window)
    apply_rounded_window_region(window, maximized=is_max, radius=radius)
    if _dwmapi is None:
        return
    try:
        hwnd = int(window.winId())
        if hwnd == 0:
            return
        if is_max and _user32 is not None:
            _user32.SetWindowRgn(hwnd, 0, True)
        pref = ctypes.c_int(DWMWCP_DONOTROUND if is_max else DWMWCP_ROUND)
        _dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(pref), ctypes.sizeof(pref)
        )
        border = ctypes.c_uint(_win_colorref_from_hex(COLORS["border"]))
        _dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_BORDER_COLOR, ctypes.byref(border), ctypes.sizeof(border)
        )
    except Exception:
        pass


def install_rounded_corner_watcher(window: QWidget, *, radius: int = WINDOW_CORNER_RADIUS) -> None:
    timer = QTimer(window)
    timer.setSingleShot(True)
    timer.setInterval(80)

    def _apply() -> None:
        if getattr(window, "_rounding_corners", False):
            return
        try:
            if not window.isVisible():
                return
        except RuntimeError:
            return
        apply_window_corner_preference(window, radius)

    timer.timeout.connect(_apply)

    class CornerFilter(QObject):
        def eventFilter(self, obj, event):  # noqa: N802
            et = event.type()
            if et in (
                QEvent.Type.Resize,
                QEvent.Type.WindowStateChange,
                QEvent.Type.Show,
                QEvent.Type.WindowActivate,
            ):
                if getattr(window, "_resize_active", False):
                    return False
                timer.start()
            return False

    filt = CornerFilter(window)
    window.installEventFilter(filt)
    window._corner_filter = filt  # type: ignore[attr-defined]
    window._corner_timer = timer  # type: ignore[attr-defined]


def enable_win32_thick_frame(window: QWidget) -> None:
    if _user32 is None or sys.platform != "win32" or not window.isVisible():
        return
    try:
        hwnd = int(window.winId())
        if hwnd == 0:
            return
        if getattr(window, "_win32_thick_frame", False):
            style = int(_user32.GetWindowLongW(hwnd, GWL_STYLE))
            if style & WS_THICKFRAME:
                return
        style = int(_user32.GetWindowLongW(hwnd, GWL_STYLE))
        style |= WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
        window._win32_thick_frame = True  # type: ignore[attr-defined]
        _user32.SetWindowLongW(hwnd, GWL_STYLE, style)
        _user32.SetWindowPos(
            hwnd,
            0,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE,
        )
    except Exception:
        pass


def nchittest_resize(window: QWidget, *, screen_pos: tuple[int, int]) -> Optional[int]:
    if is_window_filled(window):
        return None
    pos = window.mapFromGlobal(QPoint(screen_pos[0], screen_pos[1]))
    x, y = pos.x(), pos.y()
    w, h = window.width(), window.height()
    if w < 2 or h < 2:
        return None
    b = RESIZE_BORDER
    left = x < b
    right = x >= w - b
    top = y < b
    bottom = y >= h - b
    if top and left:
        return HTTOPLEFT
    if top and right:
        return HTTOPRIGHT
    if bottom and left:
        return HTBOTTOMLEFT
    if bottom and right:
        return HTBOTTOMRIGHT
    if left:
        return HTLEFT
    if right:
        return HTRIGHT
    if top:
        return HTTOP
    if bottom:
        return HTBOTTOM
    return None


def handle_native_frame_message(window: QWidget, message: int, lparam: int) -> Optional[tuple]:
    """Return (True, result) if handled, else None."""
    if message == WM_NCCALCSIZE:
        return True, 0
    if message == WM_NCACTIVATE:
        return True, 1
    if message == WM_NCHITTEST:
        try:
            if is_window_filled(window):
                return None
            sx = ctypes.c_short(lparam & 0xFFFF).value
            sy = ctypes.c_short((lparam >> 16) & 0xFFFF).value
            hit = nchittest_resize(window, screen_pos=(sx, sy))
            if hit is not None:
                return True, int(hit)
        except Exception:
            return None
        return None
    if message == WM_NCLBUTTONDBLCLK:
        return True, 0
    return None


def toggle_maximize(window: QWidget) -> None:
    """Qt maximize / restore (square when filled; rounded when normal)."""
    if is_window_filled(window):
        window._custom_maximized = False  # type: ignore[attr-defined]
        window.showNormal()
        if getattr(window, "_restore_geometry", None) is not None:
            window.setGeometry(window._restore_geometry)  # type: ignore[attr-defined]
    else:
        window._restore_geometry = window.geometry()  # type: ignore[attr-defined]
        window._custom_maximized = True  # type: ignore[attr-defined]
        window.showMaximized()
    apply_window_corner_preference(window)


def _make_title_button(text: str, *, danger: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("TitleClose" if danger else "TitleBtn")
    btn.setFixedSize(46, TITLE_BAR_HEIGHT)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setFlat(True)
    font = QFont(FONT_FAMILY)
    font.setPixelSize(14)
    btn.setFont(font)
    if danger:
        btn.setStyleSheet(
            f"""
            QPushButton#TitleClose {{
                background: transparent; color: {COLORS['fg_dim']}; border: none;
            }}
            QPushButton#TitleClose:hover {{
                background-color: {COLORS['danger']}; color: {COLORS['log_fg']};
            }}
            """
        )
    else:
        btn.setStyleSheet(
            f"""
            QPushButton#TitleBtn {{
                background: transparent; color: {COLORS['fg_dim']}; border: none;
            }}
            QPushButton#TitleBtn:hover {{
                background-color: {COLORS['panel2']}; color: {COLORS['fg']};
            }}
            """
        )
    return btn


class CustomTitleBar(QWidget):
    """Icon + title + min/max/close — same layout as STEM-organizer player."""

    def __init__(
        self,
        parent_window: QWidget,
        *,
        title: str = "Audio Aligner",
        icon_path: Optional[Path] = None,
        height: int = TITLE_BAR_HEIGHT,
    ) -> None:
        super().__init__(parent_window)
        self.setObjectName("TitleBar")
        self._win = parent_window
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.close_requested: Callable[[], None] = parent_window.close
        self.minimize_requested: Callable[[], None] = parent_window.showMinimized
        self.maximize_requested: Callable[[], None] = lambda: toggle_maximize(parent_window)

        self._drag_press_pos: Optional[QPoint] = None
        self._drag_start_pos: Optional[QPoint] = None
        self._dragging = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        self._icon_lbl = QLabel()
        self._icon_lbl.setFixedSize(TITLE_ICON_SIZE, TITLE_ICON_SIZE)
        if icon_path is not None and icon_path.is_file():
            self._icon_lbl.setPixmap(QIcon(str(icon_path)).pixmap(TITLE_ICON_SIZE, TITLE_ICON_SIZE))
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._icon_lbl)
        layout.addSpacing(8)

        self._title_lbl = QLabel(title)
        self._title_lbl.setObjectName("Title")
        self._title_lbl.setStyleSheet(
            f"color: {COLORS['fg_dim']}; background: transparent; "
            f"font-size: {TITLE_LABEL_FONT_PX}px;"
        )
        title_font = QFont(FONT_FAMILY)
        title_font.setPixelSize(TITLE_LABEL_FONT_PX)
        self._title_lbl.setFont(title_font)
        layout.addWidget(self._title_lbl)
        layout.addStretch(1)

        self.min_btn = _make_title_button(_GLYPH_MIN)
        self.max_btn = _make_title_button(_GLYPH_MAX)
        self.close_btn = _make_title_button(_GLYPH_CLOSE, danger=True)
        self.min_btn.setToolTip("Minimize")
        self.max_btn.setToolTip("Maximize / restore")
        self.close_btn.setToolTip("Close")
        self.min_btn.clicked.connect(lambda: self.minimize_requested())
        self.max_btn.clicked.connect(lambda: self.maximize_requested())
        self.close_btn.clicked.connect(lambda: self.close_requested())
        for btn in (self.min_btn, self.max_btn, self.close_btn):
            layout.addWidget(btn)

        self.setMouseTracking(True)
        self._win.setMouseTracking(True)
        self._win.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self._win and event.type() == QEvent.Type.WindowStateChange:
            filled = is_window_filled(self._win)
            self.max_btn.setText(_GLYPH_RESTORE if filled else _GLYPH_MAX)
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_press_pos = event.globalPosition().toPoint()
        self._drag_start_pos = self._win.pos()
        self._dragging = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._dragging or self._drag_press_pos is None or self._drag_start_pos is None:
            return
        delta = event.globalPosition().toPoint() - self._drag_press_pos
        if is_window_filled(self._win):
            toggle_maximize(self._win)
            self._drag_press_pos = event.globalPosition().toPoint()
            self._drag_start_pos = self._win.pos()
            return
        self._win.move(self._drag_start_pos + delta)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._dragging = False
        self._drag_press_pos = None
        self._drag_start_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        event.accept()
