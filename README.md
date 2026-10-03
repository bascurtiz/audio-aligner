# Audio Aligner

[![Audio Aligner (teaser)](https://i.ytimg.com/vi/6UzBik4-adg/maxresdefault.jpg)](https://www.youtube.com/watch?v=6UzBik4-adg)

Aligns an acapella and an instrumental to the original mix, then checks the result.

The window is `align_gui.py`. REAPER élastique is the stretch engine. Rubber Band is the fallback when REAPER is not available. Each song folder is renamed in place with `_[pass]` or `_[fail]`.

## Requirements

- Windows 10 or 11
- Python 3.10 or newer, on PATH
- [REAPER](https://www.reaper.fm/) for the élastique engine
- Rubber Band for Windows is included under `tools/rubberband/` (GPL; see `COPYING.txt` in that folder)
- Optional: iZotope RX 11 De-click. If it is not installed, the acapella is left as the stretch wrote it
- Optional: `sounddevice`, so the section editor can play audio

## Setup

```powershell
pip install -r requirements.txt
pip install sounddevice
python align_gui.py
```

The first alignment run downloads the Demucs model. That needs a network connection once.

### REAPER, once

Close every REAPER window. Open REAPER, open File → Render, set the output to WAV, and save those render settings as the default. Quit REAPER before a batch. Do not use REAPER for other work while a batch is running.

## Song folder

Each song is one subfolder of the library root:

- an acapella (the name contains Acapella, Vocal, or similar)
- an instrumental (Instrumental, Karaoke, or similar)
- the original mix (the name contains `(original song)`)
- `_backup_before_align`, with the acapella and instrumental from before alignment

The original stays at the folder root. The stems in `_backup_before_align` are what a new alignment reads.

## What a run does

1. Splits the original with Demucs into a vocal and an instrumental.
2. Puts rests back into the acapella when they were cut out, then time-aligns both stems to those splits. Pitch is kept.
3. Matches loudness to the Demucs vocal and the Demucs instrumental.
4. Runs RX 11 De-click on the acapella when that plugin is installed and De-click is set to auto-detect. The instrumental is never de-clicked.
5. Scores both stems and renames the folder `_[pass]` or `_[fail]`.

In the results table, Notes Aca and Notes Inst list the time ranges where that stem is still off.

Edit opens the section editor. Acapella shows the Demucs vocal and the acapella, split into sections you can drag. Instrumental does the same with the Demucs instrumental and the instrumental. Apply re-aligns only the stem you are looking at.

## Command line

```powershell
python warp_align_reaper.py --root "D:\path\to\songs" --only "Song" --no-move
python warp_align_fail_all.py --root "D:\path\to\songs" --only "Song" --no-move
python check_alignment.py --root "D:\path\to\songs" --dry-run --limit 5
```

Pass `--root`. The scripts’ built-in default path is only an example. `--declick` is `rx` or `off`. Rubber Band cannot host RX 11, so that engine leaves the acapella untouched. `--no-move` writes the stems and leaves the folder where it is.
