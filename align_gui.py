#!/usr/bin/env python3
"""PyQt6 GUI for align-checker warp alignment.

Wraps:
  - warp_align_reaper.process_folder  (REAPER élastique, preferred)
  - warp_align_fail_all.warp_align_folder  (Rubber Band fallback)
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QPalette, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizeGrip,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from check_alignment import (
    DEFAULT_CORR_MIN,
    DEFAULT_DRIFT_MS,
    DEFAULT_WEAK_WINDOW_FRAC,
    DEFAULT_WINDOW_CORR_MIN,
    review_fail_parts,
    review_fail_text,
)
from help_dialog import InfoIcon, show_align_help
from progress_modal import format_eta
from warp_align_fail_all import DEFAULT_MAX_PAD_SEC, FAIL_ALL, find_backup_stems
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

APP_DIR = Path(__file__).resolve().parent
OUT_CSV = APP_DIR / "gui_align_results.csv"
SETTINGS_PATH = Path(os.environ.get("APPDATA") or APP_DIR) / "Audio Aligner" / "settings.json"
RESULTS_PATH = SETTINGS_PATH.with_name("results.json")
COMMENTS_PATH = SETTINGS_PATH.with_name("comments.json")
ICON_PATH = APP_DIR / "logo.ico"
PNG_ICON_PATH = APP_DIR / "icon.png"
PLAYER_LAUNCHER = APP_DIR / "launch_stem_player.py"
STEM_ORG_DEFAULT = Path(r"D:\github\STEM-organizer-BasCurtiz\stem-organizer")
PATH_ROLE = int(Qt.ItemDataRole.UserRole)
SEEN_NAME_ROLE = PATH_ROLE + 1
SORT_ROLE = SEEN_NAME_ROLE + 1
ROW_ROLE = SORT_ROLE + 1


def _json_ready(row: dict) -> dict:
    """Keep only values the results file can store."""
    out: dict = {}
    for key, value in row.items():
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, float) and not math.isfinite(value):
            value = None
        if value is None or isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
    return out
COL_FOLDER = 0
COL_ACA_VERDICT = 1
COL_INST_VERDICT = 2
COL_COMMENT = 3
COL_NOTES_ACA = 4
COL_NOTES_INST = 5
COL_CORR = 6
COL_DRIFT = 7
COL_ACA = 8
COL_INST = 9
_VERDICT_TAG = re.compile(r"_\[(pass|fail)\]$", re.IGNORECASE)

# STEM-organizer theme tokens (stem_organizer/theme.py COLORS / DARK)
COLORS = {
    "bg": "#1e1f26",
    "panel": "#262833",
    "panel2": "#2F3140",
    "fg": "#e6e8ef",
    "fg_dim": "#9aa0b4",
    "accent": "#7c5cff",
    "accent_hov": "#9077ff",
    "danger": "#e25c5c",
    "log_bg": "#15161c",
    "log_fg": "#d6dae8",
    "border": "#3a3d4d",
    "status_trough": "#343647",
    "scrollbar": "#2a2c38",
    "scrollbar_hover": "#3a3d4d",
    "text_mute": "#7a8199",
    "control_hover": "#36384A",
    "control_pressed": "#2A2C38",
    "active_row": "#44485f",
}
LOG_OK = "#7ee0a0"
LOG_ERR = "#ff7a7a"
LOG_WARN = "#ecc990"

VERDICT_COLORS = {
    "pass": QColor(LOG_OK),
    "fail": QColor(LOG_ERR),
    "skip": QColor(COLORS["fg_dim"]),
    "error": QColor(LOG_WARN),
    "dry_run": QColor(COLORS["accent_hov"]),
}

FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"


class SortItem(QTableWidgetItem):
    """Header sort uses SORT_ROLE so numbers stay numeric and names ignore case."""

    def __lt__(self, other: QTableWidgetItem) -> bool:  # noqa: N802
        left = self.data(SORT_ROLE)
        right = other.data(SORT_ROLE)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float(left) < float(right)
        return str(left if left is not None else self.text()).casefold() < str(
            right if right is not None else other.text()
        ).casefold()


def _strip_verdict_tag(name: str) -> str:
    return _VERDICT_TAG.sub("", name.strip()).strip()


def _verdict_from_folder_name(name: str) -> str | None:
    match = _VERDICT_TAG.search(name.strip())
    return match.group(1).lower() if match else None


def accuracy_bar_text(failed: int, total: int) -> str:
    """Failed count over processed rows. Accuracy is the share that are not fail."""
    if total <= 0:
        return ""
    pct = 100.0 * (total - failed) / total
    return f"Failed: {failed}/{total} = {pct:.0f}% accuracy"


def _folders_by_stem(parent: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    try:
        with os.scandir(parent) as it:
            for entry in it:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                stem = _strip_verdict_tag(entry.name).casefold()
                found.setdefault(stem, []).append(Path(entry.path))
    except OSError:
        return {}
    return found


def resolve_tagged_folder(stored: Path, index: dict[str, list[Path]] | None = None) -> Path | None:
    """Return the live song folder after a player P/F rename."""
    if stored.is_dir():
        return stored
    stem = _strip_verdict_tag(stored.name).casefold()
    if not stem:
        return None
    if index is None:
        index = _folders_by_stem(stored.parent)
    cands = index.get(stem) or []
    tagged = [p for p in cands if _verdict_from_folder_name(p.name)]
    if len(tagged) == 1:
        return tagged[0]
    if len(cands) == 1:
        return cands[0]
    return None


def list_song_folders(root: Path, only: str | None, limit: int | None) -> list[Path]:
    """Resolve work folders: single song dir, or children of a root."""
    if not root.is_dir():
        return []

    aca, inst, orig = find_backup_stems(root)
    if aca and inst and orig:
        folders = [root]
    else:
        folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))

    if only:
        needle = only.lower().strip()
        if needle:
            folders = [f for f in folders if needle in f.name.lower()]
    if limit is not None and limit > 0:
        folders = folders[:limit]
    return folders


def _check_block(notes: str, key: str) -> str:
    """The aca_check or inst_check clause, including the text inside its parentheses."""
    start = notes.find(f"{key}=")
    if start < 0:
        return ""
    rest = notes[start:]
    other = "inst_check=" if key == "aca_check" else "aca_check="
    end = rest.find(other)
    if end > 0:
        rest = rest[:end]
    return rest


def _stem_log_detail(notes: str, key: str) -> str:
    block = _check_block(notes, key)
    if not block:
        return ""
    if "offset_past_search" in block:
        return "past the 40 ms search"
    worst = re.search(r"worst=([\d.]+)ms", block)
    jump = re.search(r"jump=([\d.]+)ms", block)
    if not worst:
        return ""
    worst_ms = float(worst.group(1))
    jump_ms = float(jump.group(1)) if jump else 0.0
    return f"within {max(worst_ms, jump_ms):.0f} ms"


def _loudness_log(notes: str) -> str:
    match = re.search(
        r"loudness_match aca=([+-]?\d+(?:\.\d+)?)dB inst=([+-]?\d+(?:\.\d+)?)dB",
        notes,
    )
    if match:
        aca = float(match.group(1))
        inst = float(match.group(2))
        return f"Loudness      acapella {aca:+.1f} dB, instrumental {inst:+.1f} dB"
    if "loudness_match_skip" in notes:
        return "Loudness      skipped"
    return ""


def _repair_log(notes: str) -> str:
    parts: list[str] = []
    for label, key in (("acapella", "aca_repair"), ("instrumental", "inst_repair")):
        match = re.search(rf"{key}=(\S+)", notes)
        if not match:
            continue
        status = match.group(1)
        if status in {"ok", "0", "skip"}:
            continue
        if status == "unresolved":
            parts.append(f"{label} still past the search window")
            continue
        if status == "1":
            worst = re.search(rf"{key}=1(?:[^;]*?)worst=([\d.]+)ms", notes)
            if worst:
                parts.append(f"{label} corrected, was {float(worst.group(1)):.0f} ms off")
            else:
                parts.append(f"{label} corrected")
    if not parts:
        return ""
    return "Repair        " + ", ".join(parts)


def _beat_log(notes: str) -> str:
    if "beat_skip=" in notes:
        return "Beat          skipped"
    if "beat_ref=" not in notes and "phase=master" not in notes and "aca_locked_to=" not in notes:
        return ""
    replaced = re.search(r"beat_replaced=(\d+)", notes)
    n = int(replaced.group(1)) if replaced else 0
    if "aca_phase=master" in notes or "aca_locked_to=master_phase" in notes or "aca_locked_to=inst_beats" in notes:
        head = "acapella on the master phase"
    elif "phase=master" in notes:
        head = "master phase"
    else:
        head = "instrumental only"
    if n > 0:
        windows = "window" if n == 1 else "windows"
        return f"Beat          {head}, {n} {windows} moved"
    return f"Beat          {head}"


def _phrase_log(notes: str) -> str:
    if "aca_gaps=0" in notes:
        return "Phrases       kept as recorded"
    if "aca_gaps=1" in notes:
        return "Phrases       rests put back"
    return ""


def _pad_log(row: dict) -> str:
    bits: list[str] = []
    for key, label in (("aca_pad_sec", "acapella"), ("inst_pad_sec", "instrumental")):
        raw = row.get(key)
        if raw is None:
            continue
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            continue
        if abs(seconds) >= 0.05:
            bits.append(f"{label} {seconds:+.2f} s")
    if not bits:
        return ""
    return "Silence       " + ", ".join(bits)


def _warning_log(notes: str) -> list[str]:
    lines: list[str] = []
    if match := re.search(r"missing:([^;]+)", notes):
        lines.append(f"Missing       {match.group(1).replace(',', ', ')}")
    if "rename_failed" in notes or "rename_blocked_exists" in notes or "dest_exists" in notes:
        lines.append("Rename        the folder name could not be updated")
    if "repair_skip:" in notes:
        lines.append("Repair        skipped")
    if "post_scan:" in notes:
        lines.append("Check         could not read the rendered stems")
    if "stem_check:" in notes:
        lines.append("Check         scoring failed")
    return lines


def format_song_log(row: dict) -> list[str]:
    """Short lines for one finished song. The stored notes stay unchanged."""
    notes = str(row.get("notes") or "")
    fails = [part for part in review_fail_text(notes).split(" | ") if part]
    lines: list[str] = []
    for key, label, check_key in (
        ("aca_verdict", "Acapella", "aca_check"),
        ("inst_verdict", "Instrumental", "inst_check"),
    ):
        verdict = str(row.get(key) or "—").upper()
        explained = any(part.casefold().startswith(label.casefold()) for part in fails)
        if verdict == "FAIL" and explained:
            detail = ""
        else:
            detail = _stem_log_detail(notes, check_key)
            if verdict == "FAIL" and detail.startswith("within "):
                detail = detail.removeprefix("within ") + " off"
        gap = "  " if detail else ""
        lines.append(f"  {label:<14}{verdict}{gap}{detail}".rstrip())
    lines.extend(f"  {part[0].upper()}{part[1:].replace(': ', ' ', 1)}" for part in fails)
    for extra in (_loudness_log(notes), _repair_log(notes), _beat_log(notes), _phrase_log(notes), _pad_log(row)):
        if extra:
            lines.append(f"  {extra}")
    moved = str(row.get("moved_to") or "")
    tag = _verdict_from_folder_name(Path(moved).name) if moved else None
    if tag:
        lines.append(f"  Tagged        {tag}")
    lines.extend(f"  {line}" for line in _warning_log(notes))
    return lines


def format_done_log(counts: dict[str, int]) -> str:
    labels = {
        "pass": "passed",
        "fail": "failed",
        "skip": "skipped",
        "error": "errors",
        "dry_run": "dry run",
    }
    if not counts:
        return "Done."
    parts = [f"{count} {labels.get(name, name)}" for name, count in sorted(counts.items())]
    return "Done. " + ", ".join(parts) + "."


def list_check_folders(root: Path, only: str | None, limit: int | None) -> list[Path]:
    """Song folders to score as they are. Includes names that already end in _[pass]/_[fail]."""
    from check_alignment import scan_folder

    aca, inst, orig, _ = scan_folder(root)
    if aca and inst and orig:
        folders = [root]
    else:
        folders = sorted(
            p
            for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name != "_backup_before_align"
        )

    if only:
        needle = only.lower().strip()
        if needle:
            folders = [f for f in folders if needle in f.name.lower()]
    if limit is not None and limit > 0:
        folders = folders[:limit]
    return folders


class AlignWorker(QThread):
    log = pyqtSignal(str)
    folder_start = pyqtSignal(str, int, int)  # name, index, total
    step = pyqtSignal(str)  # step_id
    folder_done = pyqtSignal(dict, int, int)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, opts: dict, parent=None) -> None:
        super().__init__(parent)
        self.opts = opts
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def _on_step(self, step_id: str) -> None:
        self.step.emit(step_id)

    def run(self) -> None:
        if self.opts.get("mode") == "check":
            self._run_check()
            return
        try:
            opts = self.opts
            root = Path(opts["root"])
            folders = self._resolve_folders(list_song_folders)
            if not folders:
                self.failed.emit(f"No song folders found under:\n{root}")
                return

            engine = opts["engine"]
            from declick import resolve_declick

            declick_plan = resolve_declick(opts.get("declick") or "rx", host_rx=engine == "reaper")
            self.log.emit(f"Engine: {engine}")
            self.log.emit(f"De-click: {declick_plan.summary}")
            self.log.emit(f"Root:   {root}")
            self.log.emit("References: Demucs vocal, Demucs instrumental")
            self.log.emit(f"Songs: {len(folders)}" + ("   dry run" if opts["dry_run"] else ""))
            if opts.get("gaps_cut", True):
                self.log.emit("Acapella: rests were cut out, they will be put back")
            elif engine == "reaper":
                self.log.emit("Acapella: recorded rests kept, then aligned on the master phase")
            else:
                self.log.emit("Acapella: recorded rests kept")

            reaper_exe = None
            if engine == "reaper":
                from warp_align_reaper import find_reaper, process_folder

                reaper_path = Path(opts["reaper_exe"]) if opts.get("reaper_exe") else None
                reaper_exe = find_reaper(reaper_path)
                self.log.emit(f"REAPER: {reaper_exe}")
            else:
                from warp_align_fail_all import warp_align_folder

            rows: list[dict] = []
            total = len(folders)
            for i, folder in enumerate(folders, 1):
                if self._abort:
                    self.log.emit("Aborted by user.")
                    break

                self.log.emit(f"── [{i}/{total}] {folder.name}")
                self.folder_start.emit(folder.name, i, total)
                if engine == "reaper":
                    result = process_folder(
                        folder,
                        reaper_exe=reaper_exe,
                        max_pad_sec=opts["max_pad_sec"],
                        corr_min=opts["corr_min"],
                        drift_ms=opts["drift_ms"],
                        window_corr_min=opts["window_corr_min"],
                        weak_window_frac=opts["weak_window_frac"],
                        move_on_pass=opts["move_on_pass"] and not opts["dry_run"],
                        dry_run=opts["dry_run"],
                        use_demucs_vocals=True,
                        gaps_cut=bool(opts.get("gaps_cut", True)),
                        declick=opts.get("declick") or "rx",
                        on_step=self._on_step,
                    )
                else:
                    result = warp_align_folder(
                        folder,
                        max_pad_sec=opts["max_pad_sec"],
                        corr_min=opts["corr_min"],
                        drift_ms=opts["drift_ms"],
                        window_corr_min=opts["window_corr_min"],
                        weak_window_frac=opts["weak_window_frac"],
                        move_on_pass=opts["move_on_pass"] and not opts["dry_run"],
                        dry_run=opts["dry_run"],
                        gaps_cut=bool(opts.get("gaps_cut", True)),
                        declick=opts.get("declick") or "rx",
                        on_step=self._on_step,
                    )

                row = asdict(result)
                # Prefer post-move path so Play opens the live stems
                if result.moved_to:
                    row["path"] = result.moved_to
                else:
                    row["path"] = str(folder)
                rows.append(row)
                for line in format_song_log(row):
                    self.log.emit(line)
                self.log.emit("")
                self.folder_done.emit(row, i, total)

            if not opts.get("inplace_update"):
                csv_path = Path(opts["csv"])
                self._write_csv(csv_path, rows)
                self.log.emit(f"CSV: {csv_path}")
            counts: dict[str, int] = {}
            for r in rows:
                counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
            self.log.emit(format_done_log(counts))
            self.finished_ok.emit(rows)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    def _resolve_folders(self, scanner) -> list[Path]:
        """Use an explicit folder list when present; otherwise scan root."""
        explicit = self.opts.get("folders")
        if explicit:
            return [Path(p) for p in explicit if Path(p).is_dir()]
        root = Path(self.opts["root"])
        return scanner(root, self.opts.get("only"), self.opts.get("limit"))

    def _run_check(self) -> None:
        try:
            from check_alignment import (
                DEFAULT_MAX_SHIFT_SEC,
                DEFAULT_SR,
                DEFAULT_WINDOW_SEC,
                process_folder as check_folder,
            )

            opts = self.opts
            root = Path(opts["root"])
            folders = self._resolve_folders(list_check_folders)
            if not folders:
                self.failed.emit(f"No song folders found under:\n{root}")
                return

            rename = bool(opts.get("move_on_pass")) and not opts.get("dry_run")
            self.log.emit("Check alignment. No warp.")
            self.log.emit(f"Root:   {root}")
            self.log.emit(
                f"Songs: {len(folders)}" + ("   tagging folders" if rename else "")
            )

            rows: list[dict] = []
            total = len(folders)
            for i, folder in enumerate(folders, 1):
                if self._abort:
                    self.log.emit("Aborted by user.")
                    break
                self.log.emit(f"── [{i}/{total}] {folder.name}")
                self.folder_start.emit(folder.name, i, total)
                result = check_folder(
                    folder,
                    dry_run=not rename,
                    sr=DEFAULT_SR,
                    max_shift_sec=DEFAULT_MAX_SHIFT_SEC,
                    window_sec=DEFAULT_WINDOW_SEC,
                    corr_min=opts["corr_min"],
                    drift_ms=opts["drift_ms"],
                    window_corr_min=opts["window_corr_min"],
                    weak_window_frac=opts["weak_window_frac"],
                    skip_tagged=False,
                    on_step=self._on_step,
                )
                live = folder.with_name(result.renamed_to) if result.renamed_to else folder
                row = {
                    "folder": live.name,
                    "verdict": result.verdict,
                    "aca_verdict": result.aca_verdict,
                    "inst_verdict": result.inst_verdict,
                    "aca_corr": result.aca_corr,
                    "inst_corr": result.inst_corr,
                    "aca_check_drift_ms": result.aca_check_drift_ms,
                    "inst_check_drift_ms": result.inst_check_drift_ms,
                    "aca_lag_sec": result.aca_lag_sec,
                    "inst_lag_sec": result.inst_lag_sec,
                    "corr": result.corr,
                    "drift_ms": result.drift_ms,
                    "aca_pad_sec": None,
                    "inst_pad_sec": None,
                    "path": str(live),
                    "moved_to": str(live) if result.renamed_to else "",
                    "notes": result.notes,
                }
                rows.append(row)
                for line in format_song_log(row):
                    self.log.emit(line)
                self.log.emit("")
                self.folder_done.emit(row, i, total)

            if not opts.get("inplace_update"):
                csv_path = Path(opts["csv"])
                self._write_csv(csv_path, rows)
                self.log.emit(f"CSV: {csv_path}")
            counts: dict[str, int] = {}
            for r in rows:
                counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
            self.log.emit(format_done_log(counts))
            self.finished_ok.emit(rows)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("AppRoot")
        self.setWindowTitle("Audio Aligner")
        self.resize(WIN_DEFAULT_W, WIN_DEFAULT_H)
        self.setMinimumSize(WIN_MIN_W, WIN_MIN_H)
        self._custom_maximized = False
        self._restore_geometry = None
        self._center_pending = True
        self.worker: AlignWorker | None = None
        self._player_proc: subprocess.Popen | None = None
        self._inplace_update = False
        self._inplace_stem: str | None = None
        self._progress_modal = None
        self._editor_busy = False
        self._batch_busy = False
        self._filling_rows = False
        self._comments_by_stem: dict[str, str] = {}

        prepare_dark_frameless_chrome(self)
        install_rounded_corner_watcher(self)

        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        icon = PNG_ICON_PATH if PNG_ICON_PATH.is_file() else ICON_PATH
        self.title_bar = CustomTitleBar(self, title="Audio Aligner", icon_path=icon)
        shell.addWidget(self.title_bar)

        root = QWidget()
        root.setObjectName("AppBody")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 8)
        layout.setSpacing(12)
        shell.addWidget(root, stretch=1)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        subtitle = QLabel("Split it. Sync it. Match it.")
        subtitle.setObjectName("HeaderDesc")
        subtitle.setWordWrap(False)
        header.addWidget(subtitle, 0, Qt.AlignmentFlag.AlignVCenter)
        help_icon = InfoIcon(root, on_click=lambda: show_align_help(self))
        header.addWidget(help_icon, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch(1)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("MainSplit")
        splitter.setHandleWidth(0)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        top_layout.addWidget(self._build_paths())
        top_layout.addWidget(self._build_options())
        top_layout.addLayout(self._build_actions())
        splitter.addWidget(top)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(16)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setToolTip("Share of the queued songs finished, including the one in progress.")
        readout_font = QFont(FONT_FAMILY)
        readout_font.setPixelSize(12)
        readout_font.setBold(True)
        readout = QFontMetrics(readout_font)
        self.progress_pct = QLabel("0%")
        self.progress_pct.setObjectName("ProgressReadout")
        self.progress_pct.setFont(readout_font)
        self.progress_pct.setStyleSheet(
            f"color: {COLORS['log_fg']}; background: transparent; font-size: 12px; font-weight: 600;"
        )
        self.progress_eta = QLabel("ETA —")
        self.progress_eta.setObjectName("ProgressReadout")
        self.progress_eta.setFont(readout_font)
        self.progress_eta.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )
        eta_width = max(
            readout.horizontalAdvance(sample)
            for sample in ("ETA —", "ETA 59s", "ETA 59m 59s", "ETA 59h 59m")
        )
        readout_width = eta_width + 18
        self.progress_pct.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter
        )
        self.progress_pct.setFixedWidth(readout_width)
        self.progress_eta.setFixedWidth(readout_width)
        self.progress_eta.setStyleSheet(
            f"color: {COLORS['text_mute']}; background: transparent; font-size: 12px; font-weight: 600;"
        )
        self.progress_eta.setToolTip("Estimated time left for the rest of this run.")
        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(8)
        progress_row.addWidget(self.progress_pct, 0, Qt.AlignmentFlag.AlignVCenter)
        progress_row.addWidget(self.progress, 1, Qt.AlignmentFlag.AlignVCenter)
        progress_row.addWidget(self.progress_eta, 0, Qt.AlignmentFlag.AlignVCenter)
        bottom_layout.addLayout(progress_row)
        self._run_started: float | None = None
        self._song_started: float | None = None
        self._song_open = False
        self._batch_completed = 0
        self._batch_total = 0
        self._song_durations: list[float] = []
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_batch_progress)

        mid = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Folder",
                "Aca",
                "Inst",
                "Comment",
                "Notes Aca",
                "Notes Inst",
                "Corr",
                "Drift ms",
                "Aca pad",
                "Inst pad",
            ]
        )
        header_tips = {
            COL_FOLDER: "Song folder name (processed output path stored on the row).",
            COL_ACA_VERDICT: (
                "Acapella against the Demucs vocal.\n"
                "pass / fail uses the correlation and drift limits.\n"
                "The folder is tagged fail when either stem fails."
            ),
            COL_INST_VERDICT: (
                "Instrumental against the Demucs instrumental.\n"
                "pass / fail uses the correlation and drift limits.\n"
                "The folder is tagged fail when either stem fails."
            ),
            COL_CORR: "Global mix↔original correlation (higher is better).",
            COL_DRIFT: "Mix lag range across windows in milliseconds (lower is better).",
            COL_ACA: "Front pad applied to the acapella (seconds; + = silence inserted).",
            COL_INST: "Front pad applied to the instrumental (seconds).",
            COL_COMMENT: (
                "What you hear on this track.\n"
                "Click the cell and type. Saved with the row, kept when you re-check,\n"
                "and still there after a restart."
            ),
            COL_NOTES_ACA: (
                "Where the acapella is off, so you can listen to that spot.\n"
                "A stretch with no vocal in the stem is left out.\n"
                "Hover a cell to read the full line."
            ),
            COL_NOTES_INST: (
                "Where the instrumental is off, so you can listen to that spot.\n"
                "A stretch with no drums in the stem is left out.\n"
                "Hover a cell to read the full line."
            ),
        }
        for col, tip in header_tips.items():
            item = self.table.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setSectionsMovable(True)
        hdr.setMinimumSectionSize(52)
        hdr.setDefaultSectionSize(64)
        # Default widths matched to a 1280px window layout.
        for col, width in (
            (COL_FOLDER, 340),
            (COL_ACA_VERDICT, 52),
            (COL_INST_VERDICT, 52),
            (COL_CORR, 56),
            (COL_DRIFT, 68),
            (COL_ACA, 64),
            (COL_INST, 65),
            (COL_COMMENT, 180),
            (COL_NOTES_ACA, 220),
            (COL_NOTES_INST, 220),
        ):
            hdr.resizeSection(col, width)
        hdr.setStretchLastSection(False)
        hdr.setSortIndicatorShown(True)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.AnyKeyPressed
        )
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_context_menu)
        self.table.doubleClicked.connect(self._on_table_double_clicked)
        self.table.cellClicked.connect(self._on_comment_cell_clicked)
        self.table.itemChanged.connect(self._on_comment_edited)
        self.table.setToolTip(
            "Results for each processed folder.\n"
            "P or F in the player renames the folder.\n"
            "Aca and Inst stay the measured checks.\n"
            "Click a column header to sort; click again to reverse.\n"
            "Drag column edges to resize; drag headers to reorder.\n"
            "Right-click a row → Re-align / Re-check (keeps the list), Play, Edit, or Open folder.\n"
            "Double-click a row to play.\n"
            "Shift-click selects a range. Start align then redoes those rows only.\n"
            "Click Comment and type what you hear. The note stays on that track.\n"
            "Notes Aca and Notes Inst name the time ranges to review when that stem failed."
        )
        mid.addWidget(self.table)

        self.log_view = QTextEdit()
        self.log_view.setObjectName("LogView")
        self.log_view.setReadOnly(True)
        self.log_view.setAcceptRichText(False)
        log_font = QFont(FONT_MONO)
        log_font.setPixelSize(12)
        self.log_view.setFont(log_font)
        self.log_view.setToolTip(
            "Live run log — engine, pads, loudness gains, and errors.\n"
            "Use Clear log to empty it."
        )
        mid.addWidget(self.log_view)
        mid.setStretchFactor(0, 3)
        mid.setStretchFactor(1, 2)
        bottom_layout.addWidget(mid, stretch=1)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.status = QLabel("Ready")
        self.status.setObjectName("StatusLabel")
        self.accuracy = QLabel("")
        self.accuracy.setObjectName("StatusLabel")
        self.accuracy.setToolTip(
            "Folders tagged fail, out of every row in the list.\n"
            "A row with no tag counts as failed when either stem failed.\n"
            "Accuracy is the share that are not fail.\n"
            "Example: Failed: 53/526 = 90% accuracy."
        )
        status_row = QHBoxLayout()
        status_row.setContentsMargins(16, 4, 8, 8)
        status_row.addWidget(self.status)
        status_row.addStretch(1)
        status_row.addWidget(self.accuracy)
        grip = QSizeGrip(self)
        grip.setFixedSize(16, 16)
        status_row.addWidget(grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
        shell.addLayout(status_row)

        self._load_settings()
        self._load_comments()
        self._load_results()
        self._apply_style()
        self._on_engine_changed(self.engine.currentIndex())
        self._watch_settings()
        # Clear-log wired after log_view exists
        self.clear_btn.clicked.connect(self.log_view.clear)

        self._folder_watch = QTimer(self)
        self._folder_watch.setInterval(700)
        self._folder_watch.timeout.connect(self._sync_verdicts_from_folders)
        self._folder_watch.start()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._after_first_show)

    def _after_first_show(self) -> None:
        apply_window_corner_preference(self)
        if self._center_pending:
            self._center_pending = False
            screen = self.screen()
            if screen is not None:
                ag = screen.availableGeometry()
                g = self.frameGeometry()
                self.move(
                    ag.x() + max(0, (ag.width() - g.width()) // 2),
                    ag.y() + max(0, (ag.height() - g.height()) // 2),
                )

    def _build_paths(self) -> QGroupBox:
        box = QGroupBox("SOURCE")
        box.setObjectName("Section")
        form = QFormLayout(box)

        self.root_edit = QLineEdit(str(FAIL_ALL))
        self.root_edit.setToolTip(
            "Song folder (with stems + original), or a root of song folders.\n"
            "Each song needs _backup_before_align plus an original at the top level."
        )
        browse = QPushButton("Browse…")
        browse.setToolTip("Choose a song folder or library root.")
        browse.clicked.connect(self._browse_root)
        open_btn = QPushButton("Open")
        open_btn.setToolTip("Open this path in File Explorer.")
        open_btn.clicked.connect(self._open_root_folder)
        row = QHBoxLayout()
        row.addWidget(self.root_edit, stretch=1)
        row.addWidget(browse)
        row.addWidget(open_btn)
        form.addRow("Song / root folder", row)

        self.engine = QComboBox()
        self.engine.addItem("REAPER élastique (recommended)", "reaper")
        self.engine.addItem("Rubber Band", "rubberband")
        self.engine.setToolTip(
            "Time-stretch engine for residual warp after gap restore.\n"
            "REAPER élastique Pro is preferred; Rubber Band is the bundled fallback."
        )
        self.engine.currentIndexChanged.connect(self._on_engine_changed)
        form.addRow("Engine", self.engine)

        self.declick = QComboBox()
        self.declick.addItem("Auto-detect RX 11 De-click", "rx")
        self.declick.addItem("Off", "off")
        self.declick.setToolTip(
            "Removes clicks on the acapella.\n"
            "Auto-detect runs RX 11 De-click when that plugin is installed.\n"
            "When it is not, the acapella is left as the stretch wrote it.\n"
            "Off always leaves it that way.\n"
            "RX runs after loudness, so the level match does not raise the click again.\n"
            "The instrumental is left alone — a de-clicker treats drum hits as clicks."
        )
        form.addRow("De-click", self.declick)

        self.reaper_edit = QLineEdit("")
        self.reaper_edit.setPlaceholderText(r"Auto-detect  (e.g. C:\Program Files\REAPER (x64)\reaper.exe)")
        self.reaper_edit.setToolTip(
            "Path to reaper.exe.\n"
            "Leave empty to auto-detect the usual install locations."
        )
        reaper_browse = QPushButton("…")
        reaper_browse.setFixedWidth(36)
        reaper_browse.setToolTip("Browse for reaper.exe.")
        reaper_browse.clicked.connect(self._browse_reaper)
        self.reaper_row = QHBoxLayout()
        self.reaper_row.addWidget(self.reaper_edit, stretch=1)
        self.reaper_row.addWidget(reaper_browse)
        form.addRow("REAPER exe", self.reaper_row)

        self.only_edit = QLineEdit()
        self.only_edit.setPlaceholderText("Optional name filter, e.g. 0647")
        self.only_edit.setToolTip(
            "Only process folders whose name contains this text\n"
            "(case-insensitive). Leave empty for all."
        )
        form.addRow("Only (substring)", self.only_edit)

        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(0, 9999)
        self.limit_spin.setSpecialValueText("All")
        self.limit_spin.setValue(0)
        self.limit_spin.setToolTip(
            "Process at most this many folders (after the name filter).\n"
            "All = no limit."
        )
        form.addRow("Limit folders", self.limit_spin)
        return box

    def _build_options(self) -> QGroupBox:
        box = QGroupBox("OPTIONS")
        box.setObjectName("Section")
        form = QFormLayout(box)

        flags = QHBoxLayout()
        self.dry_run = QCheckBox("Dry run (analyze only)")
        self.dry_run.setToolTip(
            "Measure pads and drift only — do not write warped stems,\n"
            "apply loudness, or rename folders."
        )
        self.move_on_pass = QCheckBox("Tag folder _[pass] / _[fail]")
        self.move_on_pass.setChecked(True)
        self.move_on_pass.setToolTip(
            "After the post-align check, rename the song folder in place.\n"
            "Pass ends with _[pass]. Fail ends with _[fail].\n"
            "The folder stays in the same directory."
        )
        flags.addWidget(self.dry_run)
        flags.addWidget(self.move_on_pass)
        flags.addStretch(1)
        form.addRow(flags)

        self.gaps_cut = QCheckBox("Silences cut between vocal phrases")
        self.gaps_cut.setChecked(True)
        self.gaps_cut.setToolTip(
            "On when the acapella had the rests removed, so phrases sit back to back.\n"
            "Those rests are put back from the original vocal, and the vocal keeps\n"
            "that placement instead of following the instrumental beat.\n"
            "Off when the acapella still has its original rests. Phrase placement\n"
            "is skipped, and the vocal follows the instrumental clock."
        )
        form.addRow(self.gaps_cut)

        nums = QHBoxLayout()
        self.max_pad = QDoubleSpinBox()
        self.max_pad.setRange(1.0, 180.0)
        self.max_pad.setDecimals(1)
        self.max_pad.setSuffix(" s")
        self.max_pad.setValue(DEFAULT_MAX_PAD_SEC)
        self.max_pad.setToolTip(
            "Maximum front pad/trim searched when lining stems up to the original\n"
            "(global chroma offset). Raise for songs with a long pre-roll."
        )

        self.corr_min = QDoubleSpinBox()
        self.corr_min.setRange(0.0, 1.0)
        self.corr_min.setSingleStep(0.01)
        self.corr_min.setDecimals(2)
        self.corr_min.setValue(DEFAULT_CORR_MIN)
        self.corr_min.setToolTip(
            "Minimum correlation for the acapella against the Demucs vocal,\n"
            "and for the instrumental against the Demucs instrumental.\n"
            "Higher = stricter (more fails)."
        )

        self.drift_ms = QDoubleSpinBox()
        self.drift_ms.setRange(1.0, 2000.0)
        self.drift_ms.setDecimals(0)
        self.drift_ms.setSuffix(" ms")
        self.drift_ms.setValue(DEFAULT_DRIFT_MS)
        self.drift_ms.setToolTip(
            "Maximum offset of either stem against its Demucs reference, at\n"
            "points every 30 seconds through the song.\n"
            "<=5 ms is the target. 10–20 ms is suspect and still passes.\n"
            "Past 20 ms the stem is flagged.\n"
            "A sudden step past 20 ms between points also fails.\n"
            "A slow walk that stays inside 20 ms does not."
        )

        self.window_corr = QDoubleSpinBox()
        self.window_corr.setRange(0.0, 1.0)
        self.window_corr.setSingleStep(0.01)
        self.window_corr.setDecimals(2)
        self.window_corr.setValue(DEFAULT_WINDOW_CORR_MIN)
        self.window_corr.setToolTip(
            "Per-window correlation floor. Windows below this are “weak”\n"
            "and counted toward the weak-fraction fail limit."
        )

        self.weak_frac = QDoubleSpinBox()
        self.weak_frac.setRange(0.0, 1.0)
        self.weak_frac.setSingleStep(0.05)
        self.weak_frac.setDecimals(2)
        self.weak_frac.setValue(DEFAULT_WEAK_WINDOW_FRAC)
        self.weak_frac.setToolTip(
            "Fail if this fraction of windows are below Win corr.\n"
            "Example: 0.35 ≈ fail when more than ~35% of windows are weak."
        )

        tip_labels = {
            "MAXIMUM FRONT PAD": self.max_pad.toolTip(),
            "MINIMUM CORRELATION": self.corr_min.toolTip(),
            "MAXIMUM DRIFT": self.drift_ms.toolTip(),
            "WINDOW CORRELATION": self.window_corr.toolTip(),
            "WEAK WINDOW FRACTION": self.weak_frac.toolTip(),
        }
        for label, w in (
            ("MAXIMUM FRONT PAD", self.max_pad),
            ("MINIMUM CORRELATION", self.corr_min),
            ("MAXIMUM DRIFT", self.drift_ms),
            ("WINDOW CORRELATION", self.window_corr),
            ("WEAK WINDOW FRACTION", self.weak_frac),
        ):
            col = QVBoxLayout()
            lbl = QLabel(label)
            lbl.setObjectName("OptionFieldLabel")
            lbl.setToolTip(tip_labels[label])
            col.addWidget(lbl)
            col.addWidget(w)
            nums.addLayout(col)
        form.addRow(nums)
        return box

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start align")
        self.start_btn.setObjectName("primary")
        self.start_btn.setToolTip(
            "Run gap-aware align + warp + Demucs loudness match.\n"
            "Silences cut between vocal phrases controls whether rests are put back.\n"
            "With two or more rows selected, only those songs are redone.\n"
            "Otherwise the folders matched by Source / Only / Limit.\n"
            "Other rows stay in the list, comments included."
        )
        self.start_btn.clicked.connect(lambda: self._start("align"))
        self.check_btn = QPushButton("Check alignment")
        self.check_btn.setToolTip(
            "Score the current acapella + instrumental against the original.\n"
            "Pass when each stem stays within the offset limit at every checkpoint.\n"
            "Does not warp. Tag folder still renames _[pass] / _[fail] unless Dry run is on.\n"
            "With two or more rows selected, only those songs are checked."
        )
        self.check_btn.clicked.connect(lambda: self._start("check"))
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip(
            "Stop after the folder currently in progress finishes.\n"
            "Already-written stems are kept."
        )
        self.stop_btn.clicked.connect(self._stop)
        self.play_btn = QPushButton("♫ Play")
        self.play_btn.setObjectName("playBtn")
        self.play_btn.setToolTip(
            "Open Audio Aligner - Player for the selected results row\n"
            "(acapella + instrumental + original).\n"
            "Double-click a row to play as well."
        )
        self.play_btn.clicked.connect(self._play_selected)
        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setObjectName("editBtn")
        self.edit_btn.setToolTip(
            "Drag acapella sections against the Demucs vocal.\n"
            "Apply writes that layout, then re-aligns the acapella (élastique and loudness). The instrumental is left as it is."
        )
        self.edit_btn.clicked.connect(self._edit_selected)
        self.progress_btn = QPushButton("Show progress")
        self.progress_btn.setToolTip("Bring the progress window back.")
        self.progress_btn.setVisible(False)
        self.progress_btn.clicked.connect(self._show_progress)
        self.clear_btn = QPushButton("Clear log")
        self.clear_btn.setToolTip("Clear the log pane on the right.")
        row.addWidget(self.start_btn)
        row.addWidget(self.check_btn)
        row.addWidget(self.stop_btn)
        row.addWidget(self.play_btn)
        row.addWidget(self.edit_btn)
        row.addWidget(self.progress_btn)
        row.addStretch(1)
        row.addWidget(self.clear_btn)
        row.setContentsMargins(0, 0, 0, 10)
        return row

    def _apply_style(self) -> None:
        c = COLORS
        self.setStyleSheet(
            f"""
            QWidget#AppRoot, QWidget#AppBody {{
                background-color: {c['bg']};
                color: {c['log_fg']};
                font-family: "{FONT_FAMILY}";
                font-size: 12px;
            }}
            QWidget {{
                background-color: {c['bg']};
                color: {c['log_fg']};
                font-family: "{FONT_FAMILY}";
                font-size: 12px;
            }}
            QLabel {{
                background: transparent;
                color: {c['log_fg']};
            }}
            QWidget#TitleBar {{
                background-color: {c['bg']};
                border-bottom: 1px solid {c['border']};
            }}
            QLabel#HeaderDesc {{
                color: {c['fg_dim']};
                font-size: 12px;
            }}
            QLabel#Title {{
                font-size: 13px;
            }}
            QLabel#StatusLabel {{
                color: {c['fg_dim']};
                font-size: 12px;
                background: transparent;
            }}
            QLabel#OptionFieldLabel {{
                color: {c['fg_dim']};
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.4px;
                background: transparent;
            }}
            QPushButton#playBtn {{
                background-color: {c['panel2']};
                color: {c['log_fg']};
            }}
            QPushButton#playBtn:hover {{
                background-color: {c['control_hover']};
            }}
            QPushButton#editBtn {{
                background-color: {c['panel2']};
                color: {c['log_fg']};
            }}
            QPushButton#editBtn:hover {{
                background-color: {c['control_hover']};
            }}
            QGroupBox {{
                background-color: {c['panel']};
                border: 1px solid {c['border']};
                border-radius: 10px;
                margin-top: 14px;
                padding: 14px 10px 10px 10px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                top: 2px;
                padding: 0 6px;
                color: {c['fg_dim']};
                background-color: {c['bg']};
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 1px;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {c['panel2']};
                color: {c['log_fg']};
                border: 1px solid {c['border']};
                border-radius: 5px;
                padding: 4px 8px;
                min-height: 24px;
                selection-background-color: {c['accent']};
                selection-color: {c['log_fg']};
            }}
            QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
                background-color: {c['control_hover']};
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                background-color: {c['bg']};
                border: 1px solid {c['border']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 22px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {c['panel2']};
                color: {c['log_fg']};
                border: 1px solid {c['border']};
                selection-background-color: {c['accent']};
                selection-color: {c['log_fg']};
            }}
            QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
                background-color: {c['control_hover']};
                border: none;
                width: 16px;
            }}
            QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
                background-color: {c['accent']};
            }}
            QCheckBox {{
                color: {c['log_fg']};
                spacing: 10px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid {c['border']};
                background-color: {c['panel2']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {c['accent']};
                border: 1px solid {c['accent_hov']};
            }}
            QPushButton {{
                background-color: {c['panel2']};
                color: {c['log_fg']};
                border: 1px solid {c['border']};
                border-radius: 6px;
                padding: 6px 14px;
                min-height: 26px;
            }}
            QPushButton:hover {{
                background-color: {c['control_hover']};
            }}
            QPushButton:pressed {{
                background-color: {c['control_pressed']};
            }}
            QPushButton:disabled {{
                color: {c['text_mute']};
                background-color: {c['panel2']};
            }}
            QPushButton#primary {{
                background-color: {c['accent']};
                border: 1px solid {c['accent_hov']};
                color: {c['log_fg']};
                font-weight: 600;
                padding: 7px 18px;
            }}
            QPushButton#primary:hover {{
                background-color: {c['accent_hov']};
            }}
            QPushButton#primary:disabled {{
                background-color: {c['panel2']};
                border: 1px solid {c['border']};
                color: {c['text_mute']};
            }}
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background-color: {c['panel2']};
                color: {c['log_fg']};
                text-align: center;
                font-family: "{FONT_FAMILY}";
                font-size: 11px;
                font-weight: 600;
                height: 16px;
            }}
            QProgressBar::chunk {{
                background-color: {c['accent']};
                border-radius: 3px;
            }}
            QTableWidget {{
                background-color: {c['log_bg']};
                color: {c['log_fg']};
                border: 1px solid {c['border']};
                border-radius: 10px;
                gridline-color: {c['border']};
                selection-background-color: {c['active_row']};
                selection-color: {c['log_fg']};
                outline: none;
            }}
            QHeaderView::section {{
                background-color: {c['panel']};
                color: {c['fg_dim']};
                border: none;
                border-bottom: 1px solid {c['border']};
                border-right: 1px solid {c['border']};
                padding: 6px 8px;
                font-weight: 600;
                font-size: 10px;
            }}
            QTextEdit#LogView {{
                background-color: {c['log_bg']};
                color: {c['log_fg']};
                border: 1px solid {c['border']};
                border-radius: 10px;
                padding: 8px;
                selection-background-color: {c['accent']};
                selection-color: {c['log_fg']};
                font-family: "{FONT_MONO}";
                font-size: 12px;
            }}
            QSplitter#MainSplit::handle {{
                background: transparent;
                border: none;
            }}
            QSplitter::handle {{
                background-color: {c['border']};
            }}
            QScrollBar:vertical {{
                background: {c['log_bg']};
                width: 8px;
                margin: 0;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {c['scrollbar']};
                min-height: 28px;
                border-radius: 4px;
                margin: 1px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {c['scrollbar_hover']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
                height: 0;
            }}
            QScrollBar:horizontal {{
                background: {c['log_bg']};
                height: 10px;
                margin: 0;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background: {c['scrollbar']};
                min-width: 28px;
                border-radius: 4px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {c['scrollbar_hover']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: transparent;
                width: 0;
            }}
            QToolTip {{
                background-color: {c['log_bg']};
                color: {c['log_fg']};
                border: 1px solid {c['border']};
                padding: 8px 12px;
            }}
            QMessageBox {{
                background-color: {c['panel']};
                color: {c['log_fg']};
            }}
            QSizeGrip {{
                background: transparent;
                width: 16px;
                height: 16px;
            }}
            QMenu {{
                background-color: {c['panel']};
                color: {c['log_fg']};
                border: 1px solid {c['border']};
                padding: 4px;
            }}
            QMenu::item {{
                background-color: transparent;
                color: {c['log_fg']};
                padding: 6px 18px;
                border-radius: 4px;
            }}
            QMenu::item:selected,
            QMenu::item:hover {{
                background-color: {c['accent']};
                color: #ffffff;
            }}
            QMenu::item:disabled {{
                background-color: transparent;
                color: {c['text_mute']};
            }}
            QMenu::separator {{
                height: 1px;
                background: {c['border']};
                margin: 4px 8px;
            }}
            """
        )

    def _browse_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select song or root folder", self.root_edit.text())
        if path:
            self.root_edit.setText(path)

    def _open_path_in_os(self, path: Path) -> None:
        if not path.is_dir():
            QMessageBox.warning(self, "Missing folder", f"Folder does not exist:\n{path}")
            return
        try:
            os.startfile(str(path))  # noqa: S606 — intentional Explorer open
        except OSError as exc:
            QMessageBox.warning(self, "Could not open", str(exc))

    def _open_root_folder(self) -> None:
        self._open_path_in_os(Path(self.root_edit.text().strip()))

    def _table_context_menu(self, pos) -> None:
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        selected_rows = {idx.row() for idx in self.table.selectionModel().selectedRows()}
        multi = index.row() in selected_rows and len(selected_rows) >= 2
        if not multi:
            self.table.selectRow(index.row())
        count = len(self._selected_folders()) if multi else 1
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        realign_label = f"Re-align {count} selected" if count > 1 else "Re-align"
        recheck_label = f"Re-check {count} selected" if count > 1 else "Re-check"
        realign_act = QAction(realign_label, self)
        realign_act.setToolTip(
            "Re-run gap-aware align on the selected rows.\n"
            "Other results rows stay in the list."
            if count > 1
            else "Re-run gap-aware align on this folder only.\nOther results rows stay in the list."
        )
        realign_act.triggered.connect(lambda: self._realign_selected("align"))
        recheck_act = QAction(recheck_label, self)
        recheck_act.setToolTip(
            "Re-score the selected rows.\nOther results rows stay in the list."
            if count > 1
            else "Re-score alignment for this folder only.\nOther results rows stay in the list."
        )
        recheck_act.triggered.connect(lambda: self._realign_selected("check"))
        busy = self.worker is not None and self.worker.isRunning()
        realign_act.setEnabled(not busy)
        recheck_act.setEnabled(not busy)
        song = self._folder_for_row(index.row())
        play_act = QAction("♫ Play", self)
        play_act.triggered.connect(lambda _checked=False, path=song: self._play_selected(path))
        edit_act = QAction("Edit sections", self)
        edit_act.triggered.connect(lambda _checked=False, path=song: self._edit_selected(path))
        open_act = QAction("Open output folder", self)
        open_act.triggered.connect(lambda _checked=False, path=song: self._open_selected_output(path))
        menu.addAction(realign_act)
        menu.addAction(recheck_act)
        menu.addSeparator()
        menu.addAction(play_act)
        menu.addAction(edit_act)
        menu.addAction(open_act)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _open_selected_output(self, song: Path | None = None) -> None:
        if not isinstance(song, Path):
            song = self._selected_song_path()
        if song is None:
            QMessageBox.information(
                self,
                "Select a track",
                "Right-click a results row, then choose Open output folder.",
            )
            return
        self._open_path_in_os(song)

    def _browse_reaper(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reaper.exe",
            self.reaper_edit.text() or r"C:\Program Files",
            "Executable (*.exe);;All files (*)",
        )
        if path:
            self.reaper_edit.setText(path)

    def _on_engine_changed(self, _index: int) -> None:
        is_reaper = self.engine.currentData() == "reaper"
        self.reaper_edit.setEnabled(is_reaper)

    def _load_settings(self) -> None:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return
        if not isinstance(data, dict):
            return
        root = data.get("root")
        if isinstance(root, str) and root.strip():
            self.root_edit.setText(root)
        engine = data.get("engine")
        if isinstance(engine, str):
            idx = self.engine.findData(engine)
            if idx >= 0:
                self.engine.setCurrentIndex(idx)
        declick = data.get("declick")
        if isinstance(declick, str):
            idx = self.declick.findData(declick)
            if idx >= 0:
                self.declick.setCurrentIndex(idx)
        reaper = data.get("reaper_exe")
        if isinstance(reaper, str):
            self.reaper_edit.setText(reaper)
        only = data.get("only")
        if isinstance(only, str):
            self.only_edit.setText(only)
        self._apply_saved_number(self.limit_spin, data.get("limit"))
        self._apply_saved_number(self.max_pad, data.get("max_pad_sec"))
        self._apply_saved_number(self.corr_min, data.get("corr_min"))
        self._apply_saved_number(self.drift_ms, data.get("drift_ms"))
        self._apply_saved_number(self.window_corr, data.get("window_corr_min"))
        self._apply_saved_number(self.weak_frac, data.get("weak_window_frac"))
        if isinstance(data.get("dry_run"), bool):
            self.dry_run.setChecked(data["dry_run"])
        if isinstance(data.get("move_on_pass"), bool):
            self.move_on_pass.setChecked(data["move_on_pass"])
        if isinstance(data.get("gaps_cut"), bool):
            self.gaps_cut.setChecked(data["gaps_cut"])

    def _load_comments(self) -> None:
        try:
            data = json.loads(COMMENTS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return
        if not isinstance(data, dict):
            return
        self._comments_by_stem = {
            str(key): str(value)
            for key, value in data.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }

    def _save_comments(self) -> None:
        try:
            COMMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = COMMENTS_PATH.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._comments_by_stem, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(COMMENTS_PATH)
        except OSError:
            return

    def _stem_key(self, folder_name: str, folder_path: str = "") -> str:
        name = Path(folder_path).name if folder_path else folder_name
        return _strip_verdict_tag(name or folder_name).casefold()

    def _remember_comment(self, stem: str, comment: str) -> None:
        text = comment.strip()
        if not stem:
            return
        if text:
            self._comments_by_stem[stem] = comment
        else:
            self._comments_by_stem.pop(stem, None)

    def _load_results(self) -> None:
        rows = self._read_results_file()
        from_csv = False
        if rows is None:
            rows = self._read_results_csv()
            from_csv = bool(rows)
        if not rows:
            return
        self.table.setSortingEnabled(False)
        try:
            for row in rows:
                if isinstance(row, dict) and (row.get("folder") or row.get("path")):
                    self._append_result_row(row)
        finally:
            self.table.setSortingEnabled(True)
        self._resort_table()
        self._refresh_accuracy()
        if from_csv:
            self._save_results()
        count = self.table.rowCount()
        if count:
            noun = "result" if count == 1 else "results"
            self.status.setText(f"Restored {count} {noun} from the last run")
        self._save_comments()

    def _read_results_file(self) -> list[dict] | None:
        if not RESULTS_PATH.is_file():
            return None
        try:
            data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return None
        if not isinstance(data, list):
            return None
        return [row for row in data if isinstance(row, dict)]

    def _read_results_csv(self) -> list[dict]:
        if not OUT_CSV.is_file():
            return []
        try:
            with OUT_CSV.open(newline="", encoding="utf-8") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        except (OSError, csv.Error, UnicodeError):
            return []

    def _save_results(self) -> None:
        rows: list[dict] = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, COL_FOLDER)
            if item is None:
                continue
            raw = item.data(ROW_ROLE)
            if not isinstance(raw, str) or not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        try:
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = RESULTS_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            tmp.replace(RESULTS_PATH)
        except OSError:
            return

    def _apply_saved_number(self, widget: QSpinBox | QDoubleSpinBox, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        widget.setValue(value)

    def _save_settings(self) -> None:
        data = {
            "root": self.root_edit.text(),
            "engine": self.engine.currentData(),
            "declick": self.declick.currentData(),
            "reaper_exe": self.reaper_edit.text(),
            "only": self.only_edit.text(),
            "limit": self.limit_spin.value(),
            "dry_run": self.dry_run.isChecked(),
            "move_on_pass": self.move_on_pass.isChecked(),
            "gaps_cut": self.gaps_cut.isChecked(),
            "max_pad_sec": self.max_pad.value(),
            "corr_min": self.corr_min.value(),
            "drift_ms": self.drift_ms.value(),
            "window_corr_min": self.window_corr.value(),
            "weak_window_frac": self.weak_frac.value(),
        }
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = SETTINGS_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(SETTINGS_PATH)
        except OSError:
            return

    def _watch_settings(self) -> None:
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(400)
        self._settings_timer.timeout.connect(self._save_settings)
        save = self._schedule_settings_save
        self.root_edit.textChanged.connect(save)
        self.reaper_edit.textChanged.connect(save)
        self.only_edit.textChanged.connect(save)
        self.engine.currentIndexChanged.connect(save)
        self.declick.currentIndexChanged.connect(save)
        self.limit_spin.valueChanged.connect(save)
        self.max_pad.valueChanged.connect(save)
        self.corr_min.valueChanged.connect(save)
        self.drift_ms.valueChanged.connect(save)
        self.window_corr.valueChanged.connect(save)
        self.weak_frac.valueChanged.connect(save)
        self.dry_run.toggled.connect(save)
        self.move_on_pass.toggled.connect(save)
        self.gaps_cut.toggled.connect(save)

    def _schedule_settings_save(self, *_args: object) -> None:
        self._settings_timer.start()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        self._save_comments()
        # A new run clears the table before the first song finishes. Closing
        # in that window must not wipe the previous results file.
        if self.table.rowCount() or not (self.worker and self.worker.isRunning()):
            self._save_results()
        super().closeEvent(event)

    def _opts(self) -> dict:
        limit = self.limit_spin.value()
        return {
            "root": self.root_edit.text().strip(),
            "engine": self.engine.currentData(),
            "declick": self.declick.currentData(),
            "reaper_exe": self.reaper_edit.text().strip() or None,
            "only": self.only_edit.text().strip() or None,
            "limit": limit if limit > 0 else None,
            "dry_run": self.dry_run.isChecked(),
            "move_on_pass": self.move_on_pass.isChecked(),
            "gaps_cut": self.gaps_cut.isChecked(),
            "max_pad_sec": float(self.max_pad.value()),
            "corr_min": float(self.corr_min.value()),
            "drift_ms": float(self.drift_ms.value()),
            "window_corr_min": float(self.window_corr.value()),
            "weak_window_frac": float(self.weak_frac.value()),
            "csv": str(OUT_CSV),
        }

    def _start(self, mode: str = "align") -> None:
        opts = self._opts()
        opts["mode"] = mode
        root = Path(opts["root"])
        if not root.is_dir():
            QMessageBox.warning(self, "Missing folder", f"Folder does not exist:\n{root}")
            return

        if mode == "check":
            folders = list_check_folders(root, opts.get("only"), opts.get("limit"))
        else:
            folders = list_song_folders(root, opts.get("only"), opts.get("limit"))
        picked = self._selected_folders()
        if len(picked) >= 2:
            folders = picked
            opts["folders"] = [str(path) for path in folders]
            opts["only"] = None
            opts["limit"] = None
            opts["selection_run"] = True
        if not folders:
            QMessageBox.warning(self, "Nothing to process", f"No song folders under:\n{root}")
            return

        self._inplace_update = False
        self._inplace_stem = None
        # Commit a comment still being typed, then keep every existing row.
        # Songs in this run replace their own row. The others stay, comments included.
        focus = QApplication.focusWidget()
        if focus is not None:
            focus.clearFocus()
        self._save_comments()
        self._refresh_accuracy()
        self._launch_worker(opts, folders, mode)

    def _realign_selected(self, mode: str = "align") -> None:
        """Re-process one selected results row without clearing the list."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "Wait for the current run to finish (or Stop) before re-aligning.",
            )
            return
        if len(self._selected_folders()) >= 2:
            self._start(mode)
            return
        song = self._selected_song_path()
        if song is None:
            QMessageBox.information(
                self,
                "Select a track",
                "Right-click a results row, then choose Re-align or Re-check.",
            )
            return
        live = resolve_tagged_folder(song) or song
        if not live.is_dir():
            QMessageBox.warning(self, "Missing folder", f"Folder does not exist:\n{live}")
            return

        opts = self._opts()
        opts["mode"] = mode
        opts["folders"] = [str(live)]
        opts["root"] = str(live.parent if live.parent.is_dir() else live)
        opts["only"] = None
        opts["limit"] = None
        opts["inplace_update"] = True

        self._inplace_update = True
        self._inplace_stem = _strip_verdict_tag(live.name).casefold()
        label = "re-check" if mode == "check" else "re-align"
        self._append_log(f"Queued 1 folder for {label}: {live.name}")
        self._launch_worker(opts, [live], mode)

    def _launch_worker(self, opts: dict, folders: list[Path], mode: str) -> None:
        self.progress.setValue(0)
        self._run_started = time.monotonic()
        self._song_started = None
        self._song_open = False
        self._batch_completed = 0
        self._batch_total = len(folders)
        self._song_durations = []
        self._progress_timer.start()
        self._refresh_batch_progress()
        if opts.get("selection_run"):
            verb = "check" if mode == "check" else "align"
            self._append_log(f"Queued {len(folders)} selected folder(s) for {verb}.")
        elif not opts.get("inplace_update"):
            label = "check" if mode == "check" else "align"
            self._append_log(f"Queued {len(folders)} folder(s) for {label}.")
        self.start_btn.setEnabled(False)
        self.check_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        if opts.get("selection_run"):
            verb = "Checking" if mode == "check" else "Aligning"
            self.status.setText(f"{verb} {len(folders)} selected…")
        elif opts.get("inplace_update"):
            name = folders[0].name if folders else ""
            self.status.setText(
                f"Re-checking {name}…" if mode == "check" else f"Re-aligning {name}…"
            )
        else:
            self.status.setText("Checking…" if mode == "check" else "Running…")

        self.worker = AlignWorker(opts)
        self.worker.log.connect(self._append_log)
        self.worker.folder_start.connect(self._on_folder_start)
        self.worker.step.connect(self._on_worker_step)
        self.worker.folder_done.connect(self._on_folder_done)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self._open_progress_modal(mode, len(folders))
        self._batch_busy = True
        self.worker.start()
        self._sync_progress_button()

    def _open_progress_modal(self, mode: str, folder_total: int) -> None:
        from progress_modal import ProgressModal

        self._close_progress_modal()
        tagging = self.move_on_pass.isChecked() and not self.dry_run.isChecked()
        modal = ProgressModal(
            self,
            mode=mode,
            folder_total=folder_total,
            tagging=tagging,
            gaps_cut=self.gaps_cut.isChecked(),
        )
        modal.stop_requested.connect(self._stop)
        modal.open_over(self)
        self._progress_modal = modal

    def _close_progress_modal(self) -> None:
        modal = self._progress_modal
        self._progress_modal = None
        if modal is not None:
            modal.finish()

    def _on_folder_start(self, name: str, index: int, total: int) -> None:
        modal = self._progress_modal
        if modal is not None:
            modal.set_folder(name, index, total)
        self._song_started = time.monotonic()
        self._song_open = True
        self._batch_total = total
        self._refresh_batch_progress()
        self.status.setText(f"Working… {name}")

    def _on_worker_step(self, step_id: str) -> None:
        modal = self._progress_modal
        if modal is not None:
            modal.set_step(step_id)
        self._refresh_batch_progress()

    def _refresh_batch_progress(self) -> None:
        total = self._batch_total
        done = self._batch_completed
        within = 0.0
        if self._song_open:
            modal = self._progress_modal
            if modal is not None:
                within = modal.fraction()
        if total <= 0:
            self.progress.setValue(0)
            self.progress_pct.setText("0%")
            self.progress_eta.setText("ETA —")
            return
        frac = (done + (within if self._song_open else 0.0)) / total
        frac = max(0.0, min(1.0, frac))
        pct = int(round(100.0 * frac))
        self.progress.setValue(pct)
        self.progress_pct.setText(f"{pct}%")
        self.progress_eta.setText(format_eta(self._batch_eta_sec(frac)))

    def _batch_eta_sec(self, frac: float) -> float | None:
        if self._run_started is None or self._batch_total <= 0:
            return None
        if frac >= 0.999:
            return 0.0
        done = self._batch_completed
        song_elapsed = 0.0
        if self._song_open and self._song_started is not None:
            song_elapsed = time.monotonic() - self._song_started
        if done <= 0:
            if frac < 0.002:
                return None
            elapsed = time.monotonic() - self._run_started
            return elapsed * (1.0 - frac) / frac
        typical = sum(self._song_durations) / len(self._song_durations)
        remaining = typical * (self._batch_total - done) - song_elapsed
        if remaining >= 0:
            return remaining
        within = 0.0
        if self._song_open:
            modal = self._progress_modal
            if modal is not None:
                within = modal.fraction()
        if song_elapsed > 0 and 0.05 <= within < 0.999:
            tail = song_elapsed * (1.0 - within) / within
            return tail + typical * max(0, self._batch_total - done - 1)
        return 0.0

    def _stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self._append_log("Stop requested — finishing current folder…")
            self.stop_btn.setEnabled(False)
            modal = self._progress_modal
            if modal is not None and modal.isVisible():
                modal.mark_stopping()

    def _sync_progress_button(self) -> None:
        self.progress_btn.setVisible(self._batch_busy or self._editor_busy)

    def _show_progress(self) -> None:
        modal = self._progress_modal
        if modal is None:
            return
        modal.reveal(self)

    def _editor_apply_begin(self, name: str) -> bool:
        if self.worker is not None and self.worker.isRunning():
            return False
        self._editor_busy = True
        self._open_progress_modal("edit", 1)
        modal = self._progress_modal
        if modal is not None:
            modal.set_folder(name, 1, 1)
            modal.set_step("write")
        self._sync_progress_button()
        return True

    def _editor_apply_step(self, step_id: str) -> None:
        modal = self._progress_modal
        if modal is not None:
            modal.set_step(step_id)

    def _editor_apply_scored(self, row: dict) -> None:
        found = self._find_row_for_result(row)
        if found is not None:
            for key, col in (("aca_pad_sec", COL_ACA), ("inst_pad_sec", COL_INST)):
                item = self.table.item(found, col)
                if item is not None and row.get(key) is None:
                    row[key] = item.data(SORT_ROLE)
        self.table.setSortingEnabled(False)
        try:
            self._update_or_append_result_row(row)
        finally:
            self.table.setSortingEnabled(True)
        self._resort_table()
        self._refresh_accuracy()
        self._save_results()
        self._append_log(f"── Edited {row.get('folder') or ''}")
        for line in format_song_log(row):
            self._append_log(line)
        self._append_log("")

    def _editor_apply_end(self) -> None:
        self._editor_busy = False
        self._close_progress_modal()
        self._sync_progress_button()

    def _append_log(self, text: str) -> None:
        normal = QTextCharFormat()
        normal.setForeground(QColor(COLORS["log_fg"]))
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.setCharFormat(normal)
        if self.log_view.document().characterCount() > 1:
            cursor.insertBlock()
            cursor.setCharFormat(normal)
        token = "PASS" if re.search(r"\bPASS\b", text) else None
        if token is None and re.search(r"\bFAIL\b", text):
            token = "FAIL"
        if token is None:
            cursor.insertText(text, normal)
        else:
            at = text.find(token)
            colored = QTextCharFormat()
            colored.setForeground(QColor(LOG_OK if token == "PASS" else LOG_ERR))
            cursor.insertText(text[:at], normal)
            cursor.insertText(text[at : at + len(token)], colored)
            cursor.insertText(text[at + len(token) :], normal)
        cursor.setCharFormat(normal)
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _sync_verdicts_from_folders(self) -> None:
        """Match each row to its folder name so a player P/F rename updates the verdict."""
        count = self.table.rowCount()
        if count == 0:
            return
        indexes: dict[str, dict[str, list[Path]]] = {}
        changed = False
        for row in range(count):
            folder_item = self.table.item(row, COL_FOLDER)
            if folder_item is None:
                continue
            raw = folder_item.data(PATH_ROLE)
            if not raw:
                continue
            stored = Path(str(raw))
            parent_key = os.path.normcase(str(stored.parent))
            if parent_key not in indexes:
                indexes[parent_key] = _folders_by_stem(stored.parent) if stored.parent.is_dir() else {}
            current = resolve_tagged_folder(stored, indexes[parent_key])
            if current is None:
                continue
            seen = str(folder_item.data(SEEN_NAME_ROLE) or folder_item.text())
            if current.name == seen and os.path.normcase(str(current)) == os.path.normcase(str(stored)):
                continue
            folder_item.setText(current.name)
            folder_item.setData(PATH_ROLE, str(current))
            folder_item.setData(SEEN_NAME_ROLE, current.name)
            folder_item.setData(SORT_ROLE, current.name.casefold())
            raw_row = folder_item.data(ROW_ROLE)
            if isinstance(raw_row, str) and raw_row:
                try:
                    stored_row = json.loads(raw_row)
                except json.JSONDecodeError:
                    stored_row = None
                if isinstance(stored_row, dict):
                    stored_row["path"] = str(current)
                    stored_row["folder"] = current.name
                    folder_item.setData(ROW_ROLE, json.dumps(stored_row, ensure_ascii=False))
            changed = True
        self._refresh_accuracy()
        if changed:
            self._save_results()
        if changed and self.status.text().startswith("Finished"):
            passed = sum(1 for row in range(count) if self._row_passed(row))
            self.status.setText(f"Finished — {count} processed, {passed} pass")
        if changed:
            self._resort_table()

    def _resort_table(self) -> None:
        if not self.table.isSortingEnabled():
            return
        header = self.table.horizontalHeader()
        self.table.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())

    def _on_folder_done(self, row: dict, index: int, total: int) -> None:
        if self._song_open and self._song_started is not None:
            self._song_durations.append(time.monotonic() - self._song_started)
        self._song_open = False
        self._song_started = None
        self._batch_completed = index
        self._batch_total = total
        modal = self._progress_modal
        if modal is not None:
            modal.note_song_done()
        self._refresh_batch_progress()
        self.table.setSortingEnabled(False)
        try:
            self._update_or_append_result_row(row)
        finally:
            self.table.setSortingEnabled(True)
        self._resort_table()
        self._refresh_accuracy()
        self._save_results()

    def _find_row_for_result(self, row: dict) -> int | None:
        """Locate an existing results row by stem (survives _[pass]/_[fail] renames)."""
        folder_path = str(row.get("path") or "")
        folder_name = str(row.get("folder") or "")
        stem = self._inplace_stem
        if not stem:
            name = Path(folder_path).name if folder_path else folder_name
            stem = _strip_verdict_tag(name).casefold()
        if not stem:
            return None
        target_path = os.path.normcase(folder_path) if folder_path else ""
        for r in range(self.table.rowCount()):
            item = self.table.item(r, COL_FOLDER)
            if item is None:
                continue
            raw = item.data(PATH_ROLE)
            if raw and target_path and os.path.normcase(str(raw)) == target_path:
                return r
            if raw and _strip_verdict_tag(Path(str(raw)).name).casefold() == stem:
                return r
            if _strip_verdict_tag(item.text()).casefold() == stem:
                return r
        return None

    def _update_or_append_result_row(self, row: dict) -> None:
        existing = self._find_row_for_result(row)
        if existing is None:
            self._append_result_row(row)
            return
        self._fill_result_row(existing, row)

    def _on_table_double_clicked(self, index) -> None:
        if index.column() == COL_COMMENT:
            return
        self._play_selected()

    def _on_comment_cell_clicked(self, row: int, column: int) -> None:
        if column != COL_COMMENT or self._filling_rows:
            return
        mods = QApplication.keyboardModifiers()
        if mods & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
            return
        item = self.table.item(row, column)
        if item is not None:
            self.table.editItem(item)

    def _on_comment_edited(self, item: QTableWidgetItem) -> None:
        if self._filling_rows or item.column() != COL_COMMENT:
            return
        folder_item = self.table.item(item.row(), COL_FOLDER)
        if folder_item is None:
            return
        comment = item.text()
        path = str(folder_item.data(PATH_ROLE) or "")
        stem = self._stem_key(folder_item.text(), path)
        self._remember_comment(stem, comment)
        raw = folder_item.data(ROW_ROLE)
        stored: dict = {}
        if isinstance(raw, str) and raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                stored = parsed
        stored["comment"] = comment
        self._filling_rows = True
        try:
            item.setData(SORT_ROLE, comment.casefold())
            folder_item.setData(ROW_ROLE, json.dumps(stored, ensure_ascii=False))
        finally:
            self._filling_rows = False
        self._save_comments()
        self._save_results()

    def _append_result_row(self, row: dict) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self._fill_result_row(r, row)

    def _fill_result_row(self, r: int, row: dict) -> None:
        self._filling_rows = True
        try:
            self._fill_result_row_inner(r, row)
        finally:
            self._filling_rows = False

    def _fill_result_row_inner(self, r: int, row: dict) -> None:
        aca_verdict = str(row.get("aca_verdict") or "—")
        inst_verdict = str(row.get("inst_verdict") or "—")
        folder_path = str(row.get("path") or "")
        folder_name = str(row.get("folder") or "")
        live = Path(folder_path) if folder_path else None
        if live is not None and live.is_dir():
            folder_name = live.name
        previous = self.table.item(r, COL_COMMENT)
        previous_comment = previous.text() if previous is not None else ""
        stem = self._stem_key(folder_name, folder_path)
        if isinstance(row.get("comment"), str) and str(row.get("comment")).strip():
            comment = str(row["comment"])
        elif previous_comment.strip():
            comment = previous_comment
        else:
            comment = self._comments_by_stem.get(stem, "")
        if comment.strip():
            self._remember_comment(stem, comment)

        def _num(key: str, fmt: str) -> tuple[str, float]:
            raw = row.get(key)
            if raw is None or raw == "":
                return "—", 0.0
            value = float(raw)
            return fmt.format(value), value

        corr_txt, corr_sort = _num("corr", "{:.3f}")
        drift_txt, drift_sort = _num("drift_ms", "{:.1f}")
        aca_txt, aca_sort = _num("aca_pad_sec", "{:+.3f}")
        inst_txt, inst_sort = _num("inst_pad_sec", "{:+.3f}")
        aca_notes, inst_notes = review_fail_parts(str(row.get("notes") or ""))
        values = {
            COL_FOLDER: folder_name,
            COL_ACA_VERDICT: aca_verdict,
            COL_INST_VERDICT: inst_verdict,
            COL_CORR: corr_txt,
            COL_DRIFT: drift_txt,
            COL_ACA: aca_txt,
            COL_INST: inst_txt,
            COL_COMMENT: comment,
            COL_NOTES_ACA: aca_notes,
            COL_NOTES_INST: inst_notes,
        }
        sort_keys: dict[int, str | float] = {
            COL_FOLDER: folder_name.casefold(),
            COL_ACA_VERDICT: aca_verdict.casefold(),
            COL_INST_VERDICT: inst_verdict.casefold(),
            COL_CORR: corr_sort,
            COL_DRIFT: drift_sort,
            COL_ACA: aca_sort,
            COL_INST: inst_sort,
            COL_COMMENT: comment.casefold(),
            COL_NOTES_ACA: aca_notes.casefold(),
            COL_NOTES_INST: inst_notes.casefold(),
        }
        tips = {
            COL_ACA_VERDICT: self._stem_tip("Acapella vs Demucs vocal", row, "aca"),
            COL_INST_VERDICT: self._stem_tip("Instrumental vs Demucs instrumental", row, "inst"),
        }
        for c, val in values.items():
            item = SortItem(val)
            item.setData(SORT_ROLE, sort_keys[c])
            if c in (COL_DRIFT, COL_ACA, COL_INST):
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
            if c == COL_FOLDER and folder_path:
                item.setData(PATH_ROLE, folder_path)
                item.setData(SEEN_NAME_ROLE, folder_name)
            if c in (COL_ACA_VERDICT, COL_INST_VERDICT):
                color = VERDICT_COLORS.get(str(val).lower())
                if color is not None:
                    item.setForeground(color)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                item.setToolTip(tips[c])
            flags = item.flags()
            if c == COL_COMMENT:
                flags |= Qt.ItemFlag.ItemIsEditable
                item.setToolTip("Click and type what you hear on this track.")
            else:
                flags &= ~Qt.ItemFlag.ItemIsEditable
            if c == COL_NOTES_ACA and aca_notes:
                item.setToolTip(aca_notes)
            if c == COL_NOTES_INST and inst_notes:
                item.setToolTip(inst_notes)
            item.setFlags(flags)
            self.table.setItem(r, c, item)
        folder_item = self.table.item(r, COL_FOLDER)
        if folder_item is not None:
            stored = _json_ready(row)
            stored["path"] = folder_path
            stored["folder"] = folder_name
            stored["comment"] = comment
            folder_item.setData(ROW_ROLE, json.dumps(stored, ensure_ascii=False))

    @staticmethod
    def _stem_tip(title: str, row: dict, prefix: str) -> str:
        corr = row.get(f"{prefix}_corr")
        drift = row.get(f"{prefix}_check_drift_ms")
        lag = row.get(f"{prefix}_lag_sec")
        if corr in (None, "") or drift in (None, ""):
            return title
        try:
            text = f"{title}\ncorr={float(corr):.3f}  drift={float(drift):.1f} ms"
            if lag not in (None, ""):
                text += f"  lag={float(lag):+.3f} s"
            return text
        except (TypeError, ValueError):
            return title

    def _folder_tag(self, row: int) -> str | None:
        item = self.table.item(row, COL_FOLDER)
        if item is None:
            return None
        return _verdict_from_folder_name(item.text())

    def _row_passed(self, row: int) -> bool:
        tag = self._folder_tag(row)
        if tag in ("pass", "fail"):
            return tag == "pass"
        aca = self.table.item(row, COL_ACA_VERDICT)
        inst = self.table.item(row, COL_INST_VERDICT)
        return (
            aca is not None
            and inst is not None
            and aca.text().strip().lower() == "pass"
            and inst.text().strip().lower() == "pass"
        )

    def _row_failed(self, row: int) -> bool:
        tag = self._folder_tag(row)
        if tag in ("pass", "fail"):
            return tag == "fail"
        aca = self.table.item(row, COL_ACA_VERDICT)
        inst = self.table.item(row, COL_INST_VERDICT)
        return any(
            item is not None and item.text().strip().lower() == "fail"
            for item in (aca, inst)
        )

    def _refresh_accuracy(self) -> None:
        total = self.table.rowCount()
        failed = sum(1 for row in range(total) if self._row_failed(row))
        self.accuracy.setText(accuracy_bar_text(failed, total))

    def _folder_for_row(self, row: int) -> Path | None:
        item = self.table.item(row, COL_FOLDER)
        if item is None:
            return None
        raw = item.data(PATH_ROLE)
        if raw:
            stored = Path(str(raw))
            live = resolve_tagged_folder(stored) or stored
            if live.is_dir():
                return live
            if stored.is_dir():
                return stored
        name = item.text().strip()
        root = Path(self.root_edit.text().strip())
        if root.is_dir() and name:
            cand = root / name
            if cand.is_dir():
                return cand
        return None

    def _selected_folders(self) -> list[Path]:
        model = self.table.selectionModel()
        if model is None:
            return []
        found: list[Path] = []
        seen: set[str] = set()
        for index in sorted(model.selectedRows(), key=lambda item: item.row()):
            folder = self._folder_for_row(index.row())
            if folder is None:
                continue
            key = os.path.normcase(str(folder))
            if key in seen:
                continue
            seen.add(key)
            found.append(folder)
        return found

    def _selected_song_path(self) -> Path | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        if item is None:
            return None
        raw = item.data(PATH_ROLE)
        if raw:
            p = Path(str(raw))
            if p.is_dir():
                return p
        # Fallback: resolve by name under current root
        name = item.text().strip()
        root = Path(self.root_edit.text().strip())
        if root.is_dir() and name:
            cand = root / name
            if cand.is_dir():
                return cand
            if root.name == name:
                return root
        return None

    def _edit_selected(self, song: Path | None = None) -> None:
        if not isinstance(song, Path):
            song = self._selected_song_path()
        if song is None:
            root = Path(self.root_edit.text().strip())
            from check_alignment import scan_folder

            if root.is_dir():
                aca, _inst, orig, _ = scan_folder(root)
                if aca and orig:
                    song = root
        if song is None:
            QMessageBox.information(
                self,
                "Select a track",
                "Select a row, then click Edit.\n"
                "The window shows the Demucs vocal and lets you drag acapella sections.",
            )
            return
        from section_editor import open_section_editor

        try:
            open_section_editor(
                song,
                declick=self.declick.currentData() or "rx",
                on_apply_begin=self._editor_apply_begin,
                on_apply_step=self._editor_apply_step,
                on_apply_scored=self._editor_apply_scored,
                on_apply_end=self._editor_apply_end,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open editor", f"{type(exc).__name__}: {exc}")

    def _play_selected(self, song: Path | None = None) -> None:
        if not isinstance(song, Path):
            song = self._selected_song_path()
        if song is None:
            # Allow Play on the source folder itself when it already has stems
            root = Path(self.root_edit.text().strip())
            from check_alignment import scan_folder

            if root.is_dir():
                aca, inst, orig, _ = scan_folder(root)
                if aca and inst and orig:
                    song = root
        if song is None:
            QMessageBox.information(
                self,
                "Select a track",
                "Select a row in the results table, then click ♫ Play.\n"
                "Or set Source to a single song folder that already has stems.",
            )
            return

        from check_alignment import scan_folder

        aca, inst, orig, notes = scan_folder(song)
        if not aca or not inst:
            QMessageBox.warning(
                self,
                "Missing stems",
                f"Need acapella + instrumental in:\n{song}\n{notes}",
            )
            return

        if not PLAYER_LAUNCHER.is_file():
            QMessageBox.critical(self, "Player missing", f"Launcher not found:\n{PLAYER_LAUNCHER}")
            return
        if not (STEM_ORG_DEFAULT / "stem_organizer" / "player" / "stem_player_window.py").is_file():
            QMessageBox.critical(
                self,
                "STEM organizer missing",
                f"Stem player not found at:\n{STEM_ORG_DEFAULT}\n"
                "Install/clone STEM-organizer or update the path in launch_stem_player.py.",
            )
            return

        # Replace previous player process so one window stays in focus
        if self._player_proc is not None and self._player_proc.poll() is None:
            try:
                self._player_proc.terminate()
            except OSError:
                pass

        cmd = [
            sys.executable,
            str(PLAYER_LAUNCHER),
            "--song",
            str(song),
            "--library",
            str(song.parent),
            "--stem-org",
            str(STEM_ORG_DEFAULT),
        ]
        try:
            self._player_proc = subprocess.Popen(
                cmd,
                cwd=str(STEM_ORG_DEFAULT),
            )
        except OSError as exc:
            QMessageBox.critical(self, "Could not start player", str(exc))
            return

        self._append_log(f"Audio Aligner - Player → {song.name}")
        self.status.setText(f"Playing: {song.name}")

    def _on_finished(self, rows: list) -> None:
        self.start_btn.setEnabled(True)
        self.check_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        self._batch_busy = False
        self._close_progress_modal()
        self._sync_progress_button()
        self._progress_timer.stop()
        self._song_open = False
        self.progress.setValue(100)
        self.progress_pct.setText("100%")
        self.progress_eta.setText(format_eta(0))
        passed = sum(1 for r in rows if r.get("verdict") == "pass")
        if self._inplace_update and rows:
            name = rows[0].get("folder") or ""
            verdict = rows[0].get("verdict") or ""
            self.status.setText(f"Updated — {name} → {verdict}")
        else:
            self.status.setText(f"Finished — {len(rows)} processed, {passed} pass")
        self._inplace_update = False
        self._inplace_stem = None
        self._refresh_accuracy()

    def _on_failed(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.check_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        self._batch_busy = False
        self._close_progress_modal()
        self._sync_progress_button()
        self._progress_timer.stop()
        self._song_open = False
        self.progress_eta.setText("ETA —")
        self._inplace_update = False
        self._inplace_stem = None
        self.status.setText("Failed")
        self._append_log(message)
        QMessageBox.critical(self, "Align failed", message.splitlines()[0])


def _apply_dark_palette(app: QApplication) -> None:
    """Match STEM-organizer dark palette so stock widgets aren't light."""
    pal = QPalette()
    window = QColor(COLORS["bg"])
    panel = QColor(COLORS["panel"])
    text = QColor(COLORS["log_fg"])
    muted = QColor(COLORS["text_mute"])
    accent = QColor(COLORS["accent"])
    base = QColor(COLORS["log_bg"])
    tip_bg = QColor(COLORS["log_bg"])
    tip_fg = QColor(COLORS["log_fg"])

    pal.setColor(QPalette.ColorRole.Window, window)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, base)
    pal.setColor(QPalette.ColorRole.AlternateBase, panel)
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.Button, panel)
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.BrightText, QColor(COLORS["danger"]))
    pal.setColor(QPalette.ColorRole.Highlight, accent)
    pal.setColor(QPalette.ColorRole.HighlightedText, tip_fg)
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        pal.setColor(group, QPalette.ColorRole.ToolTipBase, tip_bg)
        pal.setColor(group, QPalette.ColorRole.ToolTipText, tip_fg)
    pal.setColor(QPalette.ColorRole.PlaceholderText, muted)
    pal.setColor(QPalette.ColorRole.Link, accent)
    app.setPalette(pal)


def _app_icon() -> QIcon:
    # Prefer PNG — more reliable alpha on Windows titlebars than some .ico paths
    for path in (PNG_ICON_PATH, ICON_PATH):
        if path.is_file():
            return QIcon(str(path))
    return QIcon()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Aligner")
    app.setStyle("Fusion")
    icon = _app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    _apply_dark_palette(app)
    body = QFont(FONT_FAMILY)
    body.setPixelSize(12)
    app.setFont(body)
    win = MainWindow()
    if not icon.isNull():
        win.setWindowIcon(icon)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
