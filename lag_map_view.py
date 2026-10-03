"""Lag map for the selected result row."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

_BG = "#15161c"
_FG = "#e6e8ef"
_DIM = "#9aa0b4"
_ACCENT = "#7c5cff"
_INST = "#60A5FA"
_LINE = "#ecc990"
_LOW = "#e25c5c"
_GAP = "#3d6f8f"
_BEAT = "#6d7388"


class LagMapView(QWidget):
    """Estimated lag, the robust line, beats, gaps, confidence, and warp markers."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._report: dict | None = None
        self.setMinimumHeight(168)
        self.setAutoFillBackground(False)

    def set_report(self, report: dict | None) -> None:
        self._report = report if isinstance(report, dict) else None
        self.update()

    def clear(self) -> None:
        self.set_report(None)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor(_BG))
        report = self._report or {}
        times = [float(v) for v in report.get("times") or []]
        lags = [float(v) for v in report.get("lags") or []]
        if len(times) < 2 or len(lags) != len(times):
            painter.setPen(QColor(_DIM))
            font = QFont("Segoe UI")
            font.setPixelSize(13)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Select a row with an alignment map")
            painter.end()
            return

        font = QFont("Segoe UI")
        font.setPixelSize(11)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        hi_label, lo_label, t0, t1, lo, hi, offset, slope, inst, inst_lags = self._scales(
            times, lags, report
        )
        gutter = max(metrics.horizontalAdvance(hi_label), metrics.horizontalAdvance(lo_label)) + 10
        legend_h = self._legend_height(painter, rect.width(), gutter)
        right = 10
        marker_hang = 10
        bottom = marker_hang + metrics.height() + 4
        top = legend_h + 4
        left = gutter
        plot = rect.adjusted(left, top, -right, -bottom)
        if plot.width() < 20 or plot.height() < 20:
            painter.end()
            return
        def x_of(t: float) -> float:
            return plot.left() + (t - t0) / (t1 - t0) * plot.width()

        def y_of(lag: float) -> float:
            return plot.bottom() - (lag - lo) / (hi - lo) * plot.height()

        for span in report.get("low_conf_spans") or []:
            if not isinstance(span, (list, tuple)) or len(span) < 2:
                continue
            a, b = x_of(float(span[0])), x_of(float(span[1]))
            band = QColor(_LOW)
            band.setAlpha(36)
            painter.fillRect(int(min(a, b)), plot.top(), int(abs(b - a)) or 1, plot.height(), band)
        for span in report.get("gap_inserts") or []:
            if not isinstance(span, (list, tuple)) or len(span) < 2:
                continue
            a, b = x_of(float(span[0])), x_of(float(span[1]))
            band = QColor(_GAP)
            band.setAlpha(70)
            painter.fillRect(int(min(a, b)), plot.top(), int(abs(b - a)) or 1, plot.height(), band)

        painter.setPen(QPen(QColor(_BEAT), 1))
        for beat in report.get("beats") or []:
            x = x_of(float(beat))
            painter.drawLine(int(x), plot.bottom(), int(x), plot.bottom() - 7)
        painter.setPen(QPen(QColor(_FG), 1))
        for beat in report.get("downbeats") or []:
            x = x_of(float(beat))
            painter.drawLine(int(x), plot.bottom(), int(x), plot.bottom() - 14)

        painter.setPen(QPen(QColor(_LINE), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(int(x_of(t0)), int(y_of(offset + slope * t0)), int(x_of(t1)), int(y_of(offset + slope * t1)))

        inst_times = [float(v) for v in inst.get("times") or []]
        if len(inst_times) >= 2 and len(inst_lags) == len(inst_times):
            self._stroke(painter, inst_times, inst_lags, x_of, y_of, QColor(_INST))
        self._stroke(painter, times, lags, x_of, y_of, QColor(_ACCENT))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_LOW))
        for jump in report.get("jumps") or []:
            when = float(jump.get("time", 0.0))
            size = float(jump.get("size_sec", 0.0))
            y = y_of(offset + slope * when + size)
            painter.drawEllipse(int(x_of(when)) - 3, int(y) - 3, 6, 6)

        painter.setBrush(QColor(_LINE))
        painter.setPen(Qt.PenStyle.NoPen)
        for mark in report.get("markers") or []:
            x = x_of(float(mark))
            painter.drawPolygon(
                [
                    QPointF(x, plot.bottom()),
                    QPointF(x - 4, plot.bottom() + 8),
                    QPointF(x + 4, plot.bottom() + 8),
                ]
            )

        painter.setPen(QColor(_DIM))
        painter.setFont(font)
        self._draw_axis_label(painter, hi_label, left - 6, plot.top() + metrics.ascent())
        self._draw_axis_label(painter, lo_label, left - 6, plot.bottom())
        clock_y = rect.bottom() - 3
        start = _clock(t0)
        end = _clock(t1)
        start_w = metrics.horizontalAdvance(start)
        end_w = metrics.horizontalAdvance(end)
        painter.drawText(plot.left(), clock_y, start)
        end_x = plot.right() - end_w
        if end_x < plot.left() + start_w + 12:
            end_x = plot.left() + start_w + 12
        painter.drawText(int(end_x), clock_y, end)
        self._draw_legend(painter, rect.width(), left)
        painter.end()

    @staticmethod
    def _stroke(painter, times, lags, x_of, y_of, color: QColor) -> None:
        from PyQt6.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(x_of(times[0]), y_of(lags[0]))
        for t, lag in zip(times[1:], lags[1:]):
            path.lineTo(x_of(t), y_of(lag))
        painter.setPen(QPen(color, 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    @staticmethod
    def _scales(times, lags, report):
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            t1 = t0 + 1.0
        inst = report.get("inst_curve") or {}
        inst_lags = [float(v) for v in inst.get("lags") or []]
        offset = float(report.get("line_offset") if report.get("line_offset") is not None else report.get("offset_sec") or 0.0)
        slope = float(report.get("line_slope") if report.get("line_slope") is not None else report.get("drift") or 0.0)
        values = list(lags) + inst_lags + [offset + slope * t0, offset + slope * t1]
        lo = min(values)
        hi = max(values)
        if hi - lo < 0.02:
            mid = (hi + lo) / 2.0
            lo, hi = mid - 0.02, mid + 0.02
        pad = (hi - lo) * 0.12
        lo -= pad
        hi += pad
        return f"{hi * 1000:.0f}", f"{lo * 1000:.0f}", t0, t1, lo, hi, offset, slope, inst, inst_lags

    @staticmethod
    def _draw_axis_label(painter: QPainter, text: str, right: int, baseline: int) -> None:
        width = painter.fontMetrics().horizontalAdvance(text)
        painter.drawText(int(right - width), int(baseline), text)

    def _legend_height(self, painter: QPainter, width: int, origin: int) -> int:
        return 4 + len(self._legend_rows(painter, width, origin)) * 16

    def _legend_rows(self, painter: QPainter, width: int, origin: int = 8) -> list[list[tuple[str, str, str]]]:
        items = [
            ("vocal", "line", _ACCENT),
            ("inst", "line", _INST),
            ("drift", "dash", _LINE),
            ("jump", "dot", _LOW),
            ("marker", "up", _LINE),
            ("gap", "band", _GAP),
            ("beat", "tick", _BEAT),
        ]
        rows: list[list[tuple[str, str, str]]] = [[]]
        x = origin
        limit = max(origin + 40, width - 8)
        metrics = painter.fontMetrics()
        for item in items:
            need = 16 + metrics.horizontalAdvance(item[0]) + 14
            if rows[-1] and x + need > limit:
                rows.append([])
                x = origin
            rows[-1].append(item)
            x += need
        return rows

    def _draw_legend(self, painter: QPainter, width: int, origin: int = 8) -> None:
        rows = self._legend_rows(painter, width, origin)
        metrics = painter.fontMetrics()
        for r, row in enumerate(rows):
            x = origin
            y = 3 + r * 16
            for label, shape, color in row:
                self._swatch(painter, x, y + 3, shape, QColor(color))
                painter.setPen(QColor(color if shape != "band" else "#8eb4d4"))
                painter.drawText(x + 16, y + 12, label)
                x += 16 + metrics.horizontalAdvance(label) + 14

    @staticmethod
    def _swatch(painter: QPainter, x: int, y: int, shape: str, color: QColor) -> None:
        if shape == "line":
            painter.setPen(QPen(color, 1.6))
            painter.drawLine(x, y + 4, x + 12, y + 4)
        elif shape == "dash":
            painter.setPen(QPen(color, 1.4, Qt.PenStyle.DashLine))
            painter.drawLine(x, y + 4, x + 12, y + 4)
        elif shape == "dot":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(x + 3, y + 1, 6, 6)
        elif shape == "up":
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(
                [QPointF(x + 6, y), QPointF(x + 2, y + 8), QPointF(x + 10, y + 8)]
            )
        elif shape == "band":
            band = QColor(color)
            band.setAlpha(140)
            painter.fillRect(x, y + 1, 12, 8, band)
        else:
            painter.setPen(QPen(color, 1))
            painter.drawLine(x + 6, y + 8, x + 6, y)


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def metrics_text(report: dict | None) -> str:
    if not isinstance(report, dict) or not report:
        return "Offset, drift, error, and confidence appear here after a run."
    offset_ms = float(report.get("offset_sec") or 0.0) * 1000.0
    drift = float(report.get("drift_ms_per_min") or 0.0)
    p95 = float(report.get("p95_error_sec") or 0.0) * 1000.0
    jumps = int(report.get("n_discontinuities") or 0)
    conf = float(report.get("confidence") or 0.0)
    codes = [item.get("code", "") for item in report.get("failures") or [] if item.get("code")]
    code = codes[0] if codes else "none"
    profile = report.get("profile") or "default"
    return (
        f"{profile}   offset {offset_ms:+.0f} ms   drift {drift:+.0f} ms/min   "
        f"p95 {p95:.0f} ms   jumps {jumps}   conf {conf:.2f}   {code}"
    )


class LagMapPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(4)
        self.view = LagMapView()
        self.metrics = QLabel(metrics_text(None))
        self.metrics.setWordWrap(True)
        self.metrics.setStyleSheet("color: #9aa0b4; font-family: 'Segoe UI'; font-size: 12px;")
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.metrics)

    def set_report(self, report: dict | None) -> None:
        self.view.set_report(report)
        self.metrics.setText(metrics_text(report))

    def clear(self) -> None:
        self.set_report(None)
