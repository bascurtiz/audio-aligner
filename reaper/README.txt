REAPER + ELASTIQUE WARP
=======================

Uses REAPER's bundled elastique 3.3.3 Pro for pitch-preserving stem alignment.

Requires: REAPER (https://www.reaper.fm/)


SETUP (one-time — important)
----------------------------
1. CLOSE all REAPER windows before running the Python script.
2. Open REAPER once manually.
3. File > Render (Ctrl+Alt+R):
   - Output format: WAV (Wave)
   - Sample rate: use project rate
   - Bounds: Custom (script overrides times)
   - Click "Save as default" / render once so defaults exist
4. Optional: Project Settings > pitch/time stretch mode
   = elastique 3.3.3 Pro
5. Quit REAPER completely.


RUN
---
  python warp_align_reaper.py --root "D:\path\to\folders" --only "Song" --no-move
  python warp_align_reaper.py --root "D:\path\to\folders" --limit 5

If you see errors about missing rendered.wav:
  - Make sure REAPER is fully quit
  - Do step 3 again (WAV render defaults)
  - Retry


NOTES
-----
- Slower than Rubber Band (REAPER starts per stem).
- Do not use REAPER for other work during a batch.
- Faster free path: warp_align_fail_all.py (Rubber Band)
