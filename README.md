# Audio Aligner

[![Audio Aligner (teaser)](https://i.ytimg.com/vi/6UzBik4-adg/maxresdefault.jpg)](https://www.youtube.com/watch?v=6UzBik4-adg)

Aligns an acapella and an instrumental to the original mix, then checks the result.

The window is `align_gui.py`. REAPER élastique is the stretch engine. Rubber Band is the fallback when REAPER is not available. Each song folder is renamed in place with `_[pass]` or `_[fail]`.

## Requirements

- Windows 10 or 11
- Python 3.10 or newer, on PATH
- [REAPER](https://www.reaper.fm/) for the élastique engine
- Rubber Band for Windows is included under `tools/rubberband/` (GPL; see `COPYING.txt` in that folder)
- Mel-Band RoFormer under `tools/melband_roformer/` (checkpoint downloads on first split if missing)
- Optional: iZotope RX 11 De-click. If it is not installed, the acapella is left as the stretch wrote it
- Optional: `sounddevice`, so the section editor can play audio
- A CUDA GPU is used when PyTorch sees one; otherwise the split runs on CPU

## Setup

```powershell
pip install -r requirements.txt
pip install sounddevice
python align_gui.py
```

The first alignment run may download the Mel-Band RoFormer checkpoint into `tools/melband_roformer/weights/`. That needs a network connection once. Stem caches are stored per song under `_demucs_cache/melband_roformer/` (legacy folder name).

### REAPER, once

Close every REAPER window. Open REAPER, open File → Render, set the output to WAV, and save those render settings as the default. Quit REAPER before a batch. A warm REAPER worker stays up between songs in one GUI session; still do not use REAPER for other work while a batch is running.

## Song folder

Each song is one subfolder of the library root:

- an acapella (the name contains Acapella, Vocal, or similar)
- an instrumental (Instrumental, Karaoke, or similar)
- the original mix (the name contains `(original song)`)
- `_backup_before_align`, with the acapella and instrumental from before alignment

The original stays at the folder root. The stems in `_backup_before_align` are what a new alignment reads.

## What a run does

1. **SPLIT** — Separates the original with Mel-Band RoFormer into a vocal and an instrumental reference.
2. **SILENCE** — When “Silences cut between vocal phrases” is on, puts rests back into the acapella from that Mel-Band vocal.
3. **ALIGN** — Time-aligns both stems to the Mel-Band references (élastique in REAPER, or Rubber Band). Pitch is kept.
4. **LOUDNESS** — Matches loudness to the Mel-Band vocal and instrumental, then runs RX 11 De-click on the acapella when De-click is set to auto-detect and the plugin is installed. The instrumental is never de-clicked.
5. **SCORE** — Scores the mix and each stem against Mel-Band, then renames the folder `_[pass]` or `_[fail]` when tagging is on.

In the results table, Notes Aca and Notes Inst list the time ranges where that stem is still off.

**Edit** opens the section editor. Acapella shows the Mel-Band vocal and the acapella, split into sections you can drag. Instrumental does the same with the Mel-Band instrumental. Apply re-aligns only the stem you are looking at.

## Command line

```powershell
python warp_align_reaper.py --root "D:\path\to\songs" --only "Song" --no-move
python warp_align_fail_all.py --root "D:\path\to\songs" --only "Song" --no-move
python check_alignment.py --root "D:\path\to\songs" --dry-run --limit 5
```

Pass `--root`. The scripts’ built-in default path is only an example.

Useful flags on the warp scripts:

- `--declick rx` or `--declick off` (default `rx`; Rubber Band cannot host RX, so that engine leaves the acapella untouched)
- `--gaps-cut` / `--no-gaps-cut` — restore cut vocal rests (default on)
- `--no-move` — write stems and leave the folder name unchanged
