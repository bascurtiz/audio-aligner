#!/usr/bin/env python3
"""Launch Audio Aligner - Player (STEM-organizer engine) for one song folder.

PyQt6 Audio Aligner cannot host PySide6 in-process — this script is the bridge.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

STEM_ORG = Path(r"D:\github\STEM-organizer-BasCurtiz\stem-organizer")
ALIGNER_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Open Audio Aligner - Player on a song folder")
    ap.add_argument("--song", type=Path, required=True, help="Song folder with stems")
    ap.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Parent library for prev/next (default: song parent; scanned quietly)",
    )
    ap.add_argument(
        "--stem-org",
        type=Path,
        default=STEM_ORG,
        help="STEM-organizer repo root",
    )
    args = ap.parse_args(argv)

    song = args.song.resolve()
    if not song.is_dir():
        print(f"ERROR: not a folder: {song}", file=sys.stderr)
        return 2

    stem_org = args.stem_org.resolve()
    if not (stem_org / "stem_organizer" / "player" / "stem_player_window.py").is_file():
        print(f"ERROR: STEM-organizer not found at {stem_org}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(stem_org))

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from stem_organizer import theme
    from stem_organizer.player.stem_player_window import (
        list_player_song_folders,
        open_stem_player,
    )

    library = (args.library or song.parent).resolve()
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Aligner - Player")
    theme.apply_theme(app)

    icon_path = ALIGNER_DIR / "logo.ico"
    if not icon_path.is_file():
        icon_path = ALIGNER_DIR / "icon.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Do NOT pass library_root here — that triggers "Scanning …" on the big
    # parent tree before the selected song loads. Open the song immediately.
    win = open_stem_player(None, library_root=None)
    win.setWindowTitle("Audio Aligner - Player")
    if icon_path.is_file():
        win.setWindowIcon(QIcon(str(icon_path)))
    if hasattr(win, "title_bar") and hasattr(win.title_bar, "_title_lbl"):
        win.title_bar._title_lbl.setText("Audio Aligner - Player")
        if icon_path.is_file():
            from PySide6.QtCore import Qt as _Qt

            px = theme.TITLE_ICON_SIZE
            win.title_bar._icon_lbl.setPixmap(
                QIcon(str(icon_path)).pixmap(px, px)
            )
            win.title_bar._icon_lbl.setAlignment(_Qt.AlignCenter)

    theme.polish_fluent_controls(win)

    win._library_root = library
    win._song_folders = [song]
    win._folder_index = 0
    win._open_folder(song, library_index=0)

    def _quiet_fill_library() -> None:
        """Fill prev/next list without the Scanning… title flash."""
        try:
            folders = list_player_song_folders(library)
            if not folders:
                folders = [song]
            win._library_root = library
            win._song_folders = folders
            idx = win._index_in_library(song)
            win._folder_index = idx if idx >= 0 else 0
        except Exception:
            win._library_root = library
            win._song_folders = [song]
            win._folder_index = 0

    QTimer.singleShot(50, _quiet_fill_library)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
