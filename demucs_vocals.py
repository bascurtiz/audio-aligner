#!/usr/bin/env python3
"""Demucs (htdemucs) stem cache + loudness matching for aligned stems.

Splits the original mix into vocals + instrumental (no_vocals), caches them,
and scales our aligned acapella/instrumental so their overall RMS matches the
Demucs stems from the original (mix-balanced levels).
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
MODEL = "htdemucs"


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


def _load_stereo_np(path: Path, target_sr: int) -> np.ndarray:
    """Return float32 array shaped (channels, samples) at target_sr."""
    import librosa

    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    y = y.T
    if sr != target_sr:
        y = np.stack(
            [librosa.resample(y[c], orig_sr=sr, target_sr=target_sr) for c in range(y.shape[0])],
            axis=0,
        ).astype(np.float32)
    if y.shape[0] == 1:
        y = np.concatenate([y, y], axis=0)
    elif y.shape[0] > 2:
        y = y[:2]
    return y


def _write_flac(path: Path, y_nc: np.ndarray, sr: int) -> None:
    """y_nc: (N, C)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.flac")
    if tmp.exists():
        tmp.unlink()
    sf.write(str(tmp), y_nc.astype(np.float32), sr, format="FLAC")
    tmp.replace(path)


def ensure_demucs_stems(
    orig_path: Path,
    *,
    folder: Path | None = None,
    model_name: str = MODEL,
    device: str | None = None,
) -> tuple[Path, Path]:
    """Return (vocals_flac, instrumental_flac), running htdemucs if needed.

    Prefer ``ensure_demucs_stems_isolated`` from the Align GUI / worker threads:
    importing torch in that process often hits WinError 1114 on c10.dll.
    """
    vox_path = vocals_cache_path(orig_path, folder=folder)
    inst_path = instrumental_cache_path(orig_path, folder=folder)
    if demucs_cache_ready(orig_path, folder=folder):
        return vox_path, inst_path

    # Lazy-import torch only when we actually need to run Demucs (cache miss).
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    vox_path.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(model_name)
    model.to(device)
    model.eval()

    wav = torch.from_numpy(_load_stereo_np(orig_path, model.samplerate))
    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None].to(device),
            device=device,
            split=True,
            overlap=0.25,
            progress=True,
        )[0]  # (sources, channels, samples)

    names = list(model.sources)
    if "vocals" not in names:
        raise RuntimeError(f"model sources lack vocals: {names}")
    vox_idx = names.index("vocals")
    vox = np.ascontiguousarray(sources[vox_idx].cpu().numpy().T)
    # instrumental = sum of non-vocal stems (same as demucs --two-stems no_vocals)
    inst = None
    for i, name in enumerate(names):
        if i == vox_idx:
            continue
        stem = sources[i].cpu().numpy()
        inst = stem if inst is None else inst + stem
    if inst is None:
        raise RuntimeError("no non-vocal stems to build instrumental")
    inst = np.ascontiguousarray(inst.T)

    _write_flac(vox_path, vox, model.samplerate)
    _write_flac(inst_path, inst, model.samplerate)
    return vox_path, inst_path


def ensure_demucs_stems_isolated(
    orig_path: Path,
    *,
    folder: Path | None = None,
    model_name: str = MODEL,
    device: str | None = None,
    timeout_sec: float = 3600.0,
) -> tuple[Path, Path]:
    """Ensure Demucs cache exists without importing torch in this process.

    Cache hit → return paths. Cache miss → spawn a fresh Python process that
    runs htdemucs (avoids WinError 1114 on torch c10.dll in the GUI).
    """
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
    if model_name != MODEL:
        cmd.extend(["--model", model_name])
    if device:
        cmd.extend(["--device", device])

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
        raise RuntimeError(f"demucs ensure subprocess failed ({proc.returncode}): {err[-800:]}")

    if not demucs_cache_ready(orig_path, folder=folder):
        raise RuntimeError(
            f"demucs ensure finished but cache still missing under:\n{vox_path.parent}"
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


def load_vocals_mono(orig_path: Path, sr: int, *, folder: Path | None = None) -> np.ndarray:
    import librosa

    vpath = ensure_demucs_vocals(orig_path, folder=folder)
    y, _ = librosa.load(str(vpath), sr=sr, mono=True)
    return y.astype(np.float32)


def load_instrumental_mono(orig_path: Path, sr: int, *, folder: Path | None = None) -> np.ndarray:
    import librosa

    _vox, ipath = ensure_demucs_stems_isolated(orig_path, folder=folder)
    y, _ = librosa.load(str(ipath), sr=sr, mono=True)
    return y.astype(np.float32)


def _phrase_level(y: np.ndarray, *, floor: float = 1e-8) -> float:
    """Height of the loud phrases, as the editor draws them.

    The 95th percentile of frame peaks. Averaging every non-silent frame sits
    well below the sung notes on a Demucs vocal, because quiet bleed and tails
    pull that mean down and the processed stem is then turned down to match it.
    """
    if y.ndim == 2:
        y = y.mean(axis=1)
    frame = 2048
    if len(y) < frame:
        return max(float(np.max(np.abs(y))) if y.size else 0.0, floor)
    hop = frame // 2
    n = 1 + (len(y) - frame) // hop
    peaks = np.empty(n, dtype=np.float64)
    for i in range(n):
        sl = y[i * hop : i * hop + frame]
        peaks[i] = float(np.max(np.abs(sl)))
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


def apply_gain_inplace(path: Path, gain: float) -> None:
    if abs(gain - 1.0) < 1e-3:
        return
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    y = y * np.float32(gain)
    # Soft ceiling to avoid hard clips from large boosts
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y *= np.float32(0.99 / peak)

    suffix = path.suffix.lower()
    fmt = "FLAC" if suffix == ".flac" else "WAV" if suffix == ".wav" else None
    tmp = path.with_name(f"{path.stem}._gain_tmp{path.suffix}")
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass
    if fmt:
        sf.write(str(tmp), y, sr, format=fmt)
    else:
        sf.write(str(tmp), y, sr)

    # Windows often locks a file that was just written by REAPER/AV — retry replace
    import time

    last_exc: OSError | None = None
    for attempt in range(8):
        try:
            os.replace(str(tmp), str(path))
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.15 * (attempt + 1))
    # Fallback: rewrite in place via temp in same folder then replace
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



def match_aligned_stems_to_demucs(
    orig_path: Path,
    aca_path: Path | None,
    inst_path: Path | None,
    *,
    folder: Path | None = None,
    aca_only: bool = False,
    inst_only: bool = False,
) -> dict:
    """Scale aligned aca/inst so loud-phrase amplitude matches the Demucs stems."""
    import librosa

    vox_ref_p, inst_ref_p = ensure_demucs_stems(orig_path, folder=folder)

    aca = inst = inst_ref = None
    sr_a = sr_i = sr_n = 44100
    if not inst_only:
        if aca_path is None:
            raise FileNotFoundError("Acapella path is required for loudness match.")
        aca, sr_a = sf.read(str(aca_path), always_2d=True, dtype="float32")
    vox_ref, sr_v = sf.read(str(vox_ref_p), always_2d=True, dtype="float32")
    if not aca_only:
        if inst_path is None:
            raise FileNotFoundError("Instrumental path is required for loudness match.")
        inst, sr_i = sf.read(str(inst_path), always_2d=True, dtype="float32")
        inst_ref, sr_n = sf.read(str(inst_ref_p), always_2d=True, dtype="float32")

    # Resample refs to stem rates if needed
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

    g_aca = 1.0
    if not inst_only:
        vox_ref = _to_sr(vox_ref, sr_v, sr_a)
        g_aca = gain_to_match_loudness(aca, vox_ref)
        apply_gain_inplace(aca_path, g_aca)
    g_inst = 1.0
    if not aca_only:
        inst_ref = _to_sr(inst_ref, sr_n, sr_i)
        g_inst = gain_to_match_loudness(inst, inst_ref)
        apply_gain_inplace(inst_path, g_inst)

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

    ap = argparse.ArgumentParser(description="Demucs cache + loudness match CLI")
    ap.add_argument("--match", action="store_true", help="Match aca/inst to Demucs refs")
    ap.add_argument("--ensure", action="store_true", help="Build Demucs vocal/instrumental cache")
    ap.add_argument("--orig", type=Path, required=True)
    ap.add_argument("--aca", type=Path, default=None)
    ap.add_argument("--inst", type=Path, default=None)
    ap.add_argument("--folder", type=Path, default=None)
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--aca-only", action="store_true")
    ap.add_argument("--inst-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.match and not args.ensure:
        ap.error("--match or --ensure is required")
    if args.ensure:
        vox, inst = ensure_demucs_stems(
            args.orig, folder=args.folder, model_name=args.model, device=args.device
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
