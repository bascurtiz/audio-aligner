#!/usr/bin/env python3
"""Mel-Band RoFormer stem cache + loudness matching for aligned stems.

Splits the original mix with Kimberley Jensen's vocal model, caches vocals and
instrumental (mix minus vocals), and scales aligned stems to that loudness.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

CACHE_DIRNAME = "_demucs_cache"
MODEL = "melband_roformer"
REFERENCE_LABEL = "Mel-Band RoFormer"


def _cache_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{MODEL}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _stem_dir(orig_path: Path, *, folder: Path | None = None) -> Path:
    base = folder if folder is not None else orig_path.parent
    return base / CACHE_DIRNAME / MODEL / _cache_key(orig_path)


def vocals_cache_path(orig_path: Path, *, folder: Path | None = None) -> Path:
    return _stem_dir(orig_path, folder=folder) / "vocals.flac"


def instrumental_cache_path(orig_path: Path, *, folder: Path | None = None) -> Path:
    return _stem_dir(orig_path, folder=folder) / "instrumental.flac"


def demucs_cache_ready(orig_path: Path, *, folder: Path | None = None) -> bool:
    vox_path = vocals_cache_path(orig_path, folder=folder)
    inst_path = instrumental_cache_path(orig_path, folder=folder)
    return (
        vox_path.is_file()
        and vox_path.stat().st_size > 1000
        and inst_path.is_file()
        and inst_path.stat().st_size > 1000
    )


def ensure_demucs_stems(
    orig_path: Path,
    *,
    folder: Path | None = None,
    model_name: str = MODEL,
    device: str | None = None,
) -> tuple[Path, Path]:
    """Return (vocals_flac, instrumental_flac), running Mel-Band RoFormer if needed.

    Prefer ``ensure_demucs_stems_isolated`` from the Align GUI / worker threads:
    importing torch in that process often hits WinError 1114 on c10.dll.
    """
    del model_name
    vox_path = vocals_cache_path(orig_path, folder=folder)
    inst_path = instrumental_cache_path(orig_path, folder=folder)
    if demucs_cache_ready(orig_path, folder=folder):
        return vox_path, inst_path
    return _ensure_roformer_stems(orig_path, vox_path, inst_path, device=device)


def _ensure_roformer_stems(
    orig_path: Path,
    vox_path: Path,
    inst_path: Path,
    *,
    device: str | None,
) -> tuple[Path, Path]:
    """Kimberley Jensen Mel-Band RoFormer. Instrumental is mix minus vocals."""
    root = Path(__file__).resolve().parent / "tools" / "melband_roformer"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from separate import separate_to_files

    separate_to_files(orig_path, vox_path, inst_path, device=device)
    if not (vox_path.is_file() and inst_path.is_file() and vox_path.stat().st_size > 1000):
        raise RuntimeError(f"Mel-Band RoFormer wrote no stems under {vox_path.parent}")
    return vox_path, inst_path


def ensure_demucs_stems_isolated(
    orig_path: Path,
    *,
    folder: Path | None = None,
    model_name: str = MODEL,
    device: str | None = None,
    timeout_sec: float = 3600.0,
) -> tuple[Path, Path]:
    """Ensure the Mel-Band cache exists without importing torch in this process.

    Cache hit → return paths. Cache miss → spawn a fresh Python process that
    runs the vocal model (avoids WinError 1114 on torch c10.dll in the GUI).
    """
    del model_name
    vox_path = vocals_cache_path(orig_path, folder=folder)
    inst_path = instrumental_cache_path(orig_path, folder=folder)
    if demucs_cache_ready(orig_path, folder=folder):
        return vox_path, inst_path

    script = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script),
        "--ensure",
        "--orig",
        str(orig_path),
    ]
    if folder is not None:
        cmd.extend(["--folder", str(folder)])
    if device:
        cmd.extend(["--device", device])
    timeout_sec = max(timeout_sec, 6 * 3600.0)

    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=str(script.parent),
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"stem split subprocess failed ({proc.returncode}): {err[-1200:]}")

    if not demucs_cache_ready(orig_path, folder=folder):
        raise RuntimeError(
            f"Mel-Band split finished but cache still missing under:\n{vox_path.parent}"
        )
    return vox_path, inst_path


def ensure_demucs_vocals(
    orig_path: Path,
    *,
    folder: Path | None = None,
    model_name: str = MODEL,
    device: str | None = None,
) -> Path:
    """Back-compat: vocals only (also caches instrumental). Safe in GUI process."""
    vox, _inst = ensure_demucs_stems_isolated(
        orig_path, folder=folder, model_name=model_name, device=device
    )
    return vox


def _load_mono_file(path: Path, sr: int) -> np.ndarray:
    """Decode mono at ``sr`` without dragging librosa's audioread path first."""
    import librosa

    try:
        y, file_sr = sf.read(str(path), always_2d=False, dtype="float32")
        if getattr(y, "ndim", 1) == 2:
            y = y.mean(axis=1)
        y = np.asarray(y, dtype=np.float32)
        if int(file_sr) != int(sr):
            y = librosa.resample(y, orig_sr=int(file_sr), target_sr=int(sr)).astype(
                np.float32
            )
        return y
    except Exception:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y.astype(np.float32)


def load_vocals_mono(orig_path: Path, sr: int, *, folder: Path | None = None) -> np.ndarray:
    vpath = ensure_demucs_vocals(orig_path, folder=folder)
    return _load_mono_file(vpath, sr)


def load_instrumental_mono(orig_path: Path, sr: int, *, folder: Path | None = None) -> np.ndarray:
    _vox, ipath = ensure_demucs_stems_isolated(orig_path, folder=folder)
    return _load_mono_file(ipath, sr)


def _phrase_level(y: np.ndarray, *, floor: float = 1e-8) -> float:
    """Height of the loud phrases, as the editor draws them.

    The 95th percentile of frame peaks. Averaging every non-silent frame sits
    well below the sung notes on a Demucs vocal, because quiet bleed and tails
    pull that mean down and the processed stem is then turned down to match it.
    """
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    frame = 2048
    if len(y) < frame:
        return max(float(np.max(np.abs(y))) if y.size else 0.0, floor)
    hop = frame // 2
    n = 1 + (len(y) - frame) // hop
    # One strided window pass instead of a Python frame loop.
    from numpy.lib.stride_tricks import as_strided

    frames = as_strided(
        y,
        shape=(n, frame),
        strides=(y.strides[0] * hop, y.strides[0]),
        writeable=False,
    )
    peaks = np.max(np.abs(frames), axis=1)
    return max(float(np.percentile(peaks, 95)), floor)


def gain_to_match_loudness(
    stem: np.ndarray,
    ref: np.ndarray,
    *,
    max_gain: float = 8.0,
    min_gain: float = 1.0 / 8.0,
) -> float:
    """Linear gain so the stem's loud phrases match the reference amplitude."""
    n = min(len(stem), len(ref))
    if n < 1024:
        return 1.0
    g = _phrase_level(ref[:n]) / _phrase_level(stem[:n])
    return float(np.clip(g, min_gain, max_gain))


def _write_audio(path: Path, y: np.ndarray, sr: int) -> None:
    """Replace ``path`` with ``y``. Retries on Windows file locks."""
    suffix = path.suffix.lower()
    fmt = "FLAC" if suffix == ".flac" else "WAV" if suffix == ".wav" else None
    tmp = path.with_name(f"{path.stem}._gain_tmp{path.suffix}")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    # Fast FLAC encode: default compression is slow for multi-minute stems.
    if fmt == "FLAC":
        try:
            sf.write(str(tmp), y, sr, format="FLAC", compression_level=0)
        except TypeError:
            sf.write(str(tmp), y, sr, format="FLAC")
    elif fmt:
        sf.write(str(tmp), y, sr, format=fmt)
    else:
        sf.write(str(tmp), y, sr)

    import time

    last_exc: OSError | None = None
    for attempt in range(8):
        try:
            os.replace(str(tmp), str(path))
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.15 * (attempt + 1))
    try:
        if path.exists():
            path.unlink()
        os.replace(str(tmp), str(path))
    except OSError as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise OSError(f"gain write failed for {path.name}: {last_exc or exc}") from exc


def apply_gain_inplace(path: Path, gain: float) -> None:
    if abs(gain - 1.0) < 1e-3:
        return
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    y = y * np.float32(gain)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y *= np.float32(0.99 / peak)
    _write_audio(path, y, sr)


def match_aligned_stems_to_demucs(
    orig_path: Path,
    aca_path: Path | None,
    inst_path: Path | None,
    *,
    folder: Path | None = None,
    aca_only: bool = False,
    inst_only: bool = False,
) -> dict:
    """Scale aligned aca/inst so loud-phrase amplitude matches the Mel-Band stems."""
    import librosa

    vox_ref_p, inst_ref_p = ensure_demucs_stems(orig_path, folder=folder)

    aca = inst = None
    sr_a = sr_i = 44100
    if not inst_only:
        if aca_path is None:
            raise FileNotFoundError("Acapella path is required for loudness match.")
        aca, sr_a = sf.read(str(aca_path), always_2d=True, dtype="float32")
    vox_ref, sr_v = sf.read(str(vox_ref_p), always_2d=True, dtype="float32")
    inst_ref = None
    sr_n = 44100
    if not aca_only:
        if inst_path is None:
            raise FileNotFoundError("Instrumental path is required for loudness match.")
        inst, sr_i = sf.read(str(inst_path), always_2d=True, dtype="float32")
        inst_ref, sr_n = sf.read(str(inst_ref_p), always_2d=True, dtype="float32")

    def _to_sr(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        if sr_from == sr_to:
            return y
        return np.stack(
            [
                librosa.resample(y[:, c], orig_sr=sr_from, target_sr=sr_to)
                for c in range(y.shape[1])
            ],
            axis=1,
        ).astype(np.float32)

    def _apply_loaded(path: Path, audio: np.ndarray, sr: int, gain: float) -> None:
        if abs(gain - 1.0) < 1e-3:
            return
        out = audio * np.float32(gain)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out *= np.float32(0.99 / peak)
        _write_audio(path, out, sr)

    g_aca = 1.0
    if not inst_only:
        vox_ref = _to_sr(vox_ref, sr_v, sr_a)
        g_aca = gain_to_match_loudness(aca, vox_ref)
        _apply_loaded(aca_path, aca, sr_a, g_aca)
    g_inst = 1.0
    if not aca_only:
        inst_ref = _to_sr(inst_ref, sr_n, sr_i)
        g_inst = gain_to_match_loudness(inst, inst_ref)
        _apply_loaded(inst_path, inst, sr_i, g_inst)

    return {
        "aca_gain_db": float(20.0 * np.log10(max(g_aca, 1e-9))),
        "inst_gain_db": float(20.0 * np.log10(max(g_inst, 1e-9))),
        "aca_gain": g_aca,
        "inst_gain": g_inst,
        "vocals_ref": str(vox_ref_p),
        "instrumental_ref": str(inst_ref_p),
    }


def match_aligned_stems_to_demucs_isolated(
    orig_path: Path,
    aca_path: Path | None,
    inst_path: Path | None,
    *,
    folder: Path | None = None,
    timeout_sec: float = 1800.0,
    aca_only: bool = False,
    inst_only: bool = False,
) -> dict:
    """Run loudness match in a fresh Python process.

    Avoids WinError 1114 (torch c10.dll init failure) after REAPER / GUI has
    loaded conflicting native libraries into this process.
    """
    # Fast path: cache already present → no torch needed in this process
    if demucs_cache_ready(orig_path, folder=folder):
        return match_aligned_stems_to_demucs(
            orig_path,
            aca_path,
            inst_path,
            folder=folder,
            aca_only=aca_only,
            inst_only=inst_only,
        )

    script = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script),
        "--match",
        "--orig",
        str(orig_path),
    ]
    if inst_only:
        if inst_path is None:
            raise FileNotFoundError("Instrumental path is required for loudness match.")
        cmd.append("--inst-only")
        cmd.extend(["--inst", str(inst_path)])
    else:
        if aca_path is None:
            raise FileNotFoundError("Acapella path is required for loudness match.")
        cmd.extend(["--aca", str(aca_path)])
        if aca_only:
            cmd.append("--aca-only")
        elif inst_path is not None:
            cmd.extend(["--inst", str(inst_path)])
    if folder is not None:
        cmd.extend(["--folder", str(folder)])

    env = os.environ.copy()
    # Prefer a clean CUDA/torch load in the child
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=str(script.parent),
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"loudness subprocess failed ({proc.returncode}): {err[-800:]}")

    # Last JSON line is the result
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("loudness subprocess produced no output")
    return json.loads(lines[-1])


def _cli_main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Mel-Band RoFormer cache + loudness match CLI")
    ap.add_argument("--match", action="store_true", help="Match aca/inst to the Mel-Band refs")
    ap.add_argument("--ensure", action="store_true", help="Build Mel-Band vocal/instrumental cache")
    ap.add_argument("--orig", type=Path, required=True)
    ap.add_argument("--aca", type=Path, default=None)
    ap.add_argument("--inst", type=Path, default=None)
    ap.add_argument("--folder", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--aca-only", action="store_true")
    ap.add_argument("--inst-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.match and not args.ensure:
        ap.error("--match or --ensure is required")
    if args.ensure:
        vox, inst = ensure_demucs_stems(
            args.orig, folder=args.folder, device=args.device
        )
        print(json.dumps({"vocals": str(vox), "instrumental": str(inst)}), flush=True)
        return 0
    if args.inst_only:
        if args.inst is None:
            ap.error("--inst is required with --inst-only")
    elif args.aca is None:
        ap.error("--aca is required with --match")
    elif not args.aca_only and args.inst is None:
        ap.error("--inst is required unless --aca-only")
    result = match_aligned_stems_to_demucs(
        args.orig,
        args.aca,
        args.inst,
        folder=args.folder,
        aca_only=args.aca_only,
        inst_only=args.inst_only,
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
