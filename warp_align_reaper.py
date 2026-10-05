#!/usr/bin/env python3
"""Warp-align stems using REAPER's élastique 3.3.3 Pro (pitch preserved).

Pipeline:
  original -> Beat This! -> master beats, downbeats, tempo curve
  instrumental and acapella, silence-aware
  time alignment on that master phase
  locally varying élastique stretch

Requires REAPER installed (uses bundled élastique). First-time tip:
  In REAPER, set Project Settings → default pitch/time mode to
  "elastique 3.3.3 Pro" if takes ignore I_PITCHMODE on your build.
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from declick import resolve_declick
from check_alignment import (
    DEFAULT_CORR_MIN,
    DEFAULT_DRIFT_MS,
    DEFAULT_WEAK_WINDOW_FRAC,
    DEFAULT_WINDOW_CORR_MIN,
    analyze_alignment,
    apply_full_score,
    apply_stem_check,
    plan_residual_repair,
    scan_folder,
    score_rendered_pair,
    stem_verdicts,
)
from warp_align_fail_all import (
    BACKUP_NAME,
    DEFAULT_MAX_PAD_SEC,
    FAIL_ALL,
    SR_ANALYSIS,
    WarpResult,
    _pad_text,
    bandpass,
    chroma_xcorr_pad,
    find_backup_stems,
    fit_len,
    fit_len_2d,
    lag_curve,
    load_mono,
    onset_lag_curve,
    pad_or_trim_front,
    peak_norm,
    absorb_lag_offset,
    follow_lag_src_dst,
    smooth_lags,
    stretch_src_dst,
    tag_folder_verdict,
    write_csv,
)

OUT_DIR = Path(__file__).resolve().parent
REAPER_DIR = OUT_DIR / "reaper"
LUA_WORKER = REAPER_DIR / "elastique_worker.lua"
DEFAULT_REAPER = Path(r"C:\Program Files\REAPER (x64)\reaper.exe")


def find_reaper(explicit: Path | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("REAPER_EXE")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            DEFAULT_REAPER,
            Path(r"C:\Program Files\REAPER\reaper.exe"),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "REAPER (x64)" / "reaper.exe",
        ]
    )
    for c in candidates:
        if c and c.is_file():
            return c
    raise FileNotFoundError(
        "REAPER not found. Install REAPER or pass --reaper-exe / set REAPER_EXE."
    )


# RX 11 De-click, preset "Fix Discontinuous Waveform":
# multi-band (random clicks), sensitivity 5, click widening 4 ms.
# The plugin ignores host parameter writes, so this is its saved state.
RX_DECLICK_FIX_DISCONTINUOUS = (
    "WwEAAAEAAABzU0MAAwAAAE8BAADRCAAAeJy1lU1TgzAQhu/8ikzOHvy8eBNbHQ86HdF6XmmsqyHpJKkj0+G/"
    "K9japiRtoHAC3k3effKxyyIihNBrKQz7NiQxYBgll2RRymXkKZ9VAh1galAKUDk9WgXHwOfW8FIcgYJMb6n7"
    "rGqGxTqweq2eRTWDDpJRJ7Clz5CzjAlzMLKlV+Ys5Zh+OoNh7ntz2JmYIld8KhWa92zHcCv3850wzqzbuU/d"
    "YwrP3A2mGMGxuW6cGy4hjOe4Nc8j0ziZAw9liqXkQUhvwDVrjZUwodHgF5q8+926aI31ghMmUEy7ZzpvyLQs"
    "VPIA2a5qsGgSo0r2EJz/em2IVTU8MmZK/9Zw52V35qFxyDWtjkxvuXz13/wOW1Lfh7VcSMOz+ptF4nwGOrgn"
    "9Vr//d4eX5MMuj2R58v1Q76HD6lcy/BArxFPNl1QHOQSFdEPeUa7jgAAAAAAAAAA"
)


def markers_from_lag(
    times: np.ndarray,
    lags: np.ndarray,
    *,
    target_sec: float,
    in_sec: float,
    follow: bool = False,
    scores: np.ndarray | None = None,
    despike: bool = True,
    grid_sec: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Build stretch markers: dst on the output timeline, src in the padded input.

    The default is two endpoints, one rate. follow=True samples the lag curve
    so a bend in the middle is not left behind. grid_sec is the 8-bar ruler.
    """
    if follow:
        src, dst = follow_lag_src_dst(
            times,
            lags,
            target_sec=target_sec,
            in_sec=in_sec,
            scores=scores,
            despike=despike,
            grid_sec=grid_sec,
        )
    else:
        src, dst = stretch_src_dst(times, lags, target_sec=target_sec, in_sec=in_sec)
    return [{"src": float(s), "dst": float(d)} for s, d in zip(src, dst)]


def playback_end_sec(
    markers: list[dict[str, float]],
    *,
    in_sec: float,
    target_sec: float,
) -> float:
    """Item length so the source tail plays out past the last marker.

    The last rate holds. The item is at least the original's length, and
    longer when the source still has audio after that length.
    """
    if len(markers) >= 2:
        prev, last = markers[-2], markers[-1]
        dt = float(last["dst"]) - float(prev["dst"])
        ds = float(last["src"]) - float(prev["src"])
        rate = ds / dt if dt > 1e-6 else 1.0
        src0 = float(last["src"])
        dst0 = float(last["dst"])
    else:
        rate = 1.0
        src0 = float(markers[0]["src"]) if markers else 0.0
        dst0 = float(markers[0]["dst"]) if markers else 0.0
    if rate <= 1e-4:
        return target_sec
    remaining = in_sec - src0
    if remaining <= 1e-3:
        return max(target_sec, dst0)
    return max(target_sec, dst0 + remaining / rate)


def write_padded_wav(src: Path, dest_wav: Path, *, pad_sec: float, target_sr: int) -> float:
    """Pad/trim and resample to target_sr WAV. Returns duration seconds of result."""
    y, file_sr = sf.read(str(src), always_2d=True, dtype="float32")
    if file_sr != target_sr:
        import librosa

        y = np.stack(
            [librosa.resample(y[:, c], orig_sr=file_sr, target_sr=target_sr) for c in range(y.shape[1])],
            axis=1,
        ).astype(np.float32)
    n = int(round(pad_sec * target_sr))
    if n > 0:
        y = np.concatenate([np.zeros((n, y.shape[1]), dtype=np.float32), y], axis=0)
    elif n < 0:
        y = y[min(-n, len(y)) :]
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest_wav), y, target_sr, subtype="FLOAT")
    return len(y) / target_sr


# Keep the worker off-screen so a fullscreen game keeps focus.
SW_HIDE = 0
GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080

CREATE_UNICODE_ENVIRONMENT = 0x00000400
STARTF_USESHOWWINDOW = 0x00000001
DESKTOP_ALL_ACCESS = 0x01FF
STILL_ACTIVE = 259

if sys.platform == "win32":
    try:
        _user32 = ctypes.WinDLL("user32", use_last_error=True)
        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except OSError:
        _user32 = None
        _kernel32 = None
else:
    _user32 = None
    _kernel32 = None

if _user32 is not None:
    _LONG_PTR = ctypes.c_ssize_t
    _get_exstyle = getattr(_user32, "GetWindowLongPtrW", _user32.GetWindowLongW)
    _set_exstyle = getattr(_user32, "SetWindowLongPtrW", _user32.SetWindowLongW)
    _get_exstyle.argtypes = [wintypes.HWND, ctypes.c_int]
    _get_exstyle.restype = _LONG_PTR
    _set_exstyle.argtypes = [wintypes.HWND, ctypes.c_int, _LONG_PTR]
    _set_exstyle.restype = _LONG_PTR
    _user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.ShowWindow.restype = wintypes.BOOL
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _user32.IsWindow.argtypes = [wintypes.HWND]
    _user32.IsWindow.restype = wintypes.BOOL
    _WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
    _user32.EnumWindows.restype = wintypes.BOOL
    _user32.CreateDesktopW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    _user32.CreateDesktopW.restype = wintypes.HANDLE
    _user32.CloseDesktop.argtypes = [wintypes.HANDLE]
    _user32.CloseDesktop.restype = wintypes.BOOL
    _kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _kernel32.CreateProcessW.restype = wintypes.BOOL
    _kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateProcess.restype = wintypes.BOOL
    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL


def _hide_pid_windows(pid: int) -> None:
    """Hide top-level windows owned by pid and stop them taking foreground."""
    if _user32 is None or pid <= 0:
        return

    @ _WNDENUMPROC
    def _callback(hwnd, _lparam):
        try:
            if not _user32.IsWindow(hwnd):
                return True
            owner = wintypes.DWORD()
            _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if int(owner.value) != pid:
                return True
            style = int(_get_exstyle(hwnd, GWL_EXSTYLE))
            style |= WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            _set_exstyle(hwnd, GWL_EXSTYLE, style)
            _user32.ShowWindow(hwnd, SW_HIDE)
        except Exception:
            pass
        return True

    try:
        _user32.EnumWindows(_callback, 0)
    except Exception:
        return


def _keep_reaper_hidden(proc: subprocess.Popen, stop: threading.Event) -> None:
    while not stop.is_set():
        _hide_pid_windows(proc.pid)
        if proc.poll() is not None:
            return
        stop.wait(0.05)
    _hide_pid_windows(proc.pid)


def _hidden_startupinfo() -> subprocess.STARTUPINFO | None:
    if sys.platform != "win32":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = SW_HIDE
    return info


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _WinProcess:
    """Handle from CreateProcess, with the same poll/terminate surface as Popen."""

    def __init__(self, handle: int, pid: int) -> None:
        self._handle = handle
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None or _kernel32 is None:
            return self.returncode
        code = wintypes.DWORD()
        if not _kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
            return None
        if int(code.value) == STILL_ACTIVE:
            return None
        self.returncode = int(code.value)
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        if _kernel32 is None:
            return self.poll()
        ms = 0xFFFFFFFF if timeout is None else max(0, int(timeout * 1000))
        _kernel32.WaitForSingleObject(self._handle, ms)
        return self.poll()

    def terminate(self) -> None:
        if _kernel32 is not None and self._handle:
            _kernel32.TerminateProcess(self._handle, 1)

    def kill(self) -> None:
        self.terminate()

    def close(self) -> None:
        if _kernel32 is not None and self._handle:
            _kernel32.CloseHandle(self._handle)
            self._handle = None


def _environment_block(env: dict[str, str]) -> ctypes.Array:
    raw = "\0".join(f"{key}={value}" for key, value in env.items()) + "\0\0"
    return ctypes.create_unicode_buffer(raw)


def _open_hidden_desktop() -> tuple[int, str] | None:
    """A desktop on this window station that is never switched to."""
    if _user32 is None:
        return None
    name = f"AlignCheckerReaper{os.getpid()}_{time.monotonic_ns()}"
    handle = _user32.CreateDesktopW(name, None, None, 0, DESKTOP_ALL_ACCESS, None)
    if not handle:
        return None
    return int(handle), name


def _start_reaper_process(
    argv: list[str],
    env: dict[str, str],
    desktop: str | None,
) -> _WinProcess | subprocess.Popen:
    """Start REAPER. Prefer a desktop the user never sees.

    subprocess.STARTUPINFO cannot pass lpDesktop, so a direct CreateProcess
    is required. Without that, REAPER opens on the visible desktop and the
    hide loop only catches it after it has already flashed.
    """
    if _kernel32 is None or _user32 is None or not desktop:
        return subprocess.Popen(argv, env=env, startupinfo=_hidden_startupinfo())

    startup = _STARTUPINFOW()
    startup.cb = ctypes.sizeof(_STARTUPINFOW)
    startup.lpDesktop = desktop
    startup.dwFlags = STARTF_USESHOWWINDOW
    startup.wShowWindow = SW_HIDE
    process_info = _PROCESS_INFORMATION()
    command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
    environment = _environment_block(env)
    ok = _kernel32.CreateProcessW(
        argv[0],
        command,
        None,
        None,
        False,
        CREATE_UNICODE_ENVIRONMENT,
        ctypes.cast(environment, ctypes.c_void_p),
        None,
        ctypes.byref(startup),
        ctypes.byref(process_info),
    )
    if not ok:
        raise OSError(f"Could not start REAPER ({ctypes.get_last_error()})")
    _kernel32.CloseHandle(process_info.hThread)
    return _WinProcess(int(process_info.hProcess), int(process_info.dwProcessId))


class _WarmReaper:
    """One hidden REAPER instance that stays up between songs.

    Cold ``-newinst`` pays process + plugin + project startup on every warp.
    The warm worker loads once, then polls an inbox for job batches.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._proc: _WinProcess | subprocess.Popen | None = None
        self._inbox: Path | None = None
        self._desktop_handle: int | None = None
        self._hide_stop: threading.Event | None = None
        self._hide_thread: threading.Thread | None = None
        self._reaper_exe: Path | None = None

    def shutdown(self, *, grace_sec: float = 8.0) -> None:
        with self._lock:
            self._shutdown_unlocked(grace_sec=grace_sec)

    def _shutdown_unlocked(self, *, grace_sec: float = 8.0) -> None:
        proc = self._proc
        inbox = self._inbox
        hide_stop = self._hide_stop
        hide_thread = self._hide_thread
        desktop_handle = self._desktop_handle
        self._proc = None
        self._inbox = None
        self._hide_stop = None
        self._hide_thread = None
        self._desktop_handle = None
        self._reaper_exe = None
        if inbox is not None:
            try:
                (inbox / "quit").write_text("1\n", encoding="utf-8")
            except OSError:
                pass
        if hide_stop is not None:
            hide_stop.set()
        if hide_thread is not None:
            hide_thread.join(timeout=min(1.0, max(0.05, grace_sec)))
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=max(0.05, float(grace_sec)))
            except Exception:
                pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=min(1.0, max(0.2, float(grace_sec))))
                except Exception:
                    proc.kill()
        close = getattr(proc, "close", None) if proc is not None else None
        if close is not None:
            close()
        if desktop_handle and _user32 is not None:
            _user32.CloseDesktop(desktop_handle)
        if inbox is not None and inbox.is_dir():
            shutil.rmtree(inbox, ignore_errors=True)

    def _ensure_unlocked(self, reaper_exe: Path) -> Path:
        """Start or reuse the warm worker. Caller must hold ``_lock``."""
        reaper_exe = Path(reaper_exe)
        if (
            self._proc is not None
            and self._inbox is not None
            and self._reaper_exe == reaper_exe
            and self._proc.poll() is None
            and (self._inbox / "ready").is_file()
        ):
            return self._inbox
        self._shutdown_unlocked()
        if not LUA_WORKER.is_file():
            raise FileNotFoundError(f"Missing Lua worker: {LUA_WORKER}")
        inbox = Path(tempfile.mkdtemp(prefix="align_reaper_warm_"))
        for name in ("job.json", "job.json.done", "job.json.error", "job.busy", "quit", "ready"):
            path = inbox / name
            if path.exists():
                path.unlink()
        env = os.environ.copy()
        env["ALIGN_CHECKER_REAPER_INBOX"] = str(inbox)
        env.pop("ALIGN_CHECKER_REAPER_JOB", None)
        cmd = [str(reaper_exe), "-nosplash", "-newinst", str(LUA_WORKER)]
        hidden = _open_hidden_desktop()
        desktop_handle = hidden[0] if hidden else None
        desktop_name = hidden[1] if hidden else None
        proc = _start_reaper_process(cmd, env, desktop_name)
        hide_stop = threading.Event()
        hide_thread = threading.Thread(
            target=_keep_reaper_hidden,
            args=(proc, hide_stop),
            daemon=True,
        )
        hide_thread.start()
        self._proc = proc
        self._inbox = inbox
        self._desktop_handle = desktop_handle
        self._hide_stop = hide_stop
        self._hide_thread = hide_thread
        self._reaper_exe = reaper_exe
        ready = inbox / "ready"
        deadline = time.time() + 90.0
        while time.time() < deadline:
            if proc.poll() is not None:
                code = proc.returncode
                self._shutdown_unlocked()
                raise RuntimeError(f"Warm REAPER exited before ready (code {code})")
            if ready.is_file():
                return inbox
            time.sleep(0.05)
        self._shutdown_unlocked()
        raise TimeoutError("Warm REAPER did not become ready")

    def run_jobs(
        self,
        reaper_exe: Path,
        jobs: list[dict],
        *,
        timeout_sec: float,
        expected_outputs: list[Path] | None,
    ) -> None:
        with self._job_lock:
            with self._lock:
                inbox = self._ensure_unlocked(reaper_exe)
                job_path = inbox / "job.json"
                done_path = Path(str(job_path) + ".done")
                err_path = Path(str(job_path) + ".error")
                busy_path = inbox / "job.busy"
                for path in (done_path, err_path, busy_path, job_path):
                    if path.exists():
                        path.unlink()
                partial = inbox / "job.json.partial"
                partial.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")
                partial.replace(job_path)
                proc = self._proc
            deadline = time.time() + timeout_sec * max(1, len(jobs))
            while time.time() < deadline:
                if err_path.is_file():
                    msg = err_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(f"REAPER worker failed: {msg.strip()}")
                if done_path.is_file():
                    time.sleep(0.35)
                    break
                if proc is not None and proc.poll() is not None:
                    time.sleep(0.35)
                    if err_path.is_file():
                        msg = err_path.read_text(encoding="utf-8", errors="replace")
                        raise RuntimeError(f"REAPER worker failed: {msg.strip()}")
                    if done_path.is_file():
                        break
                    raise RuntimeError(
                        f"Warm REAPER exited early (code {proc.returncode}) without .done"
                    )
                time.sleep(0.2)
            else:
                raise TimeoutError(f"REAPER job timed out after {timeout_sec}s")
            if expected_outputs:
                missing = [p for p in expected_outputs if not p.is_file()]
                if missing:
                    parent = missing[0].parent
                    listing = ", ".join(x.name for x in parent.glob("*")) or "(empty)"
                    raise FileNotFoundError(
                        f"REAPER finished but output missing: {missing[0].name}. "
                        f"Temp dir contains: {listing}"
                    )


_WARM_REAPER = _WarmReaper()
atexit.register(lambda: _WARM_REAPER.shutdown())


def shutdown_warm_reaper(*, grace_sec: float = 8.0) -> None:
    """Stop the shared REAPER worker. Safe to call more than once.

    ``grace_sec`` is how long to wait for a polite quit before terminate.
    App close should pass a short value so the window can disappear promptly.
    """
    _WARM_REAPER.shutdown(grace_sec=grace_sec)


def run_reaper_jobs(
    reaper_exe: Path,
    jobs: list[dict],
    *,
    timeout_sec: float = 600.0,
    expected_outputs: list[Path] | None = None,
) -> None:
    if not LUA_WORKER.is_file():
        raise FileNotFoundError(f"Missing Lua worker: {LUA_WORKER}")
    if not jobs:
        return
    # ALIGN_CHECKER_REAPER_COLD=1 keeps the old one-shot process for debugging.
    if os.environ.get("ALIGN_CHECKER_REAPER_COLD", "").strip() not in ("", "0", "false", "False"):
        _run_reaper_jobs_cold(
            reaper_exe, jobs, timeout_sec=timeout_sec, expected_outputs=expected_outputs
        )
        return
    try:
        _WARM_REAPER.run_jobs(
            reaper_exe,
            jobs,
            timeout_sec=timeout_sec,
            expected_outputs=expected_outputs,
        )
        return
    except RuntimeError as exc:
        text = str(exc)
        if text.startswith("REAPER worker failed:"):
            raise
        if "exited" not in text.lower() and "ready" not in text.lower():
            raise
    except TimeoutError as exc:
        if "ready" not in str(exc).lower():
            raise
    shutdown_warm_reaper()
    _run_reaper_jobs_cold(
        reaper_exe, jobs, timeout_sec=timeout_sec, expected_outputs=expected_outputs
    )


def _run_reaper_jobs_cold(
    reaper_exe: Path,
    jobs: list[dict],
    *,
    timeout_sec: float = 600.0,
    expected_outputs: list[Path] | None = None,
) -> None:
    if not LUA_WORKER.is_file():
        raise FileNotFoundError(f"Missing Lua worker: {LUA_WORKER}")

    with tempfile.TemporaryDirectory(prefix="reaper_elastique_") as tmp:
        tmp_path = Path(tmp)
        job_path = tmp_path / "reaper_job.json"
        done_path = Path(str(job_path) + ".done")
        err_path = Path(str(job_path) + ".error")
        for p in (done_path, err_path):
            if p.exists():
                p.unlink()

        job_path.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env["ALIGN_CHECKER_REAPER_JOB"] = str(job_path)
        env.pop("ALIGN_CHECKER_REAPER_INBOX", None)

        # -newinst so an already-open REAPER is not brought forward.
        # The worker runs on a desktop that is never shown.
        cmd = [str(reaper_exe), "-nosplash", "-newinst", str(LUA_WORKER)]
        hidden = _open_hidden_desktop()
        desktop_handle = hidden[0] if hidden else None
        desktop_name = hidden[1] if hidden else None
        proc = None
        hide_stop = threading.Event()
        hide_thread = None
        deadline = time.time() + timeout_sec * max(1, len(jobs))
        try:
            proc = _start_reaper_process(cmd, env, desktop_name)
            hide_thread = threading.Thread(
                target=_keep_reaper_hidden,
                args=(proc, hide_stop),
                daemon=True,
            )
            hide_thread.start()
            while time.time() < deadline:
                if err_path.is_file():
                    msg = err_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(f"REAPER worker failed: {msg.strip()}")
                if done_path.is_file():
                    time.sleep(0.5)  # flush
                    break
                if proc.poll() is not None:
                    time.sleep(0.5)
                    if err_path.is_file():
                        msg = err_path.read_text(encoding="utf-8", errors="replace")
                        raise RuntimeError(f"REAPER worker failed: {msg.strip()}")
                    if done_path.is_file():
                        break
                    raise RuntimeError(
                        f"REAPER exited early (code {proc.returncode}) without .done — "
                        "open REAPER once, set File>Render defaults to WAV, try again."
                    )
                time.sleep(0.4)
            else:
                raise TimeoutError(f"REAPER job timed out after {timeout_sec}s")

            if expected_outputs:
                missing = [p for p in expected_outputs if not p.is_file()]
                if missing:
                    # show what did appear next to first missing
                    parent = missing[0].parent
                    listing = ", ".join(x.name for x in parent.glob("*")) or "(empty)"
                    raise FileNotFoundError(
                        f"REAPER finished but output missing: {missing[0].name}. "
                        f"Temp dir contains: {listing}"
                    )
        finally:
            hide_stop.set()
            if hide_thread is not None:
                hide_thread.join(timeout=1.0)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
            close = getattr(proc, "close", None)
            if close is not None:
                close()
            if desktop_handle and _user32 is not None:
                _user32.CloseDesktop(desktop_handle)


@dataclass
class _StagedStem:
    """Padded wav plus the REAPER job. The temp dir stays until commit."""

    tmp: tempfile.TemporaryDirectory
    rendered: Path
    dest: Path
    target_sr: int
    target_frames: int
    job: dict


def marker_bar_grid(orig: Path, target_sec: float) -> tuple[np.ndarray | None, str]:
    """8-bar downbeats of the original, used only as stretch-marker times.

    The lag at each mark is still the measured lag. A tracker failure keeps
    the 30 s marks.
    """
    from warp_align_fail_all import bar_grid_for

    return bar_grid_for(orig, target_sec)


def stage_stem_reaper(
    src: Path,
    dest: Path,
    *,
    pad_sec: float,
    times: np.ndarray,
    lags: np.ndarray,
    target_sr: int,
    target_frames: int,
    declick: bool = False,
    follow_lag: bool = False,
    scores: np.ndarray | None = None,
    despike: bool = True,
    stretch_mode: str | None = None,
    grid_sec: np.ndarray | None = None,
    src_dst: tuple[np.ndarray, np.ndarray] | None = None,
) -> _StagedStem:
    """Write the padded wav and the stretch job. Does not start REAPER."""
    target_sec = target_frames / target_sr
    tmp = tempfile.TemporaryDirectory(prefix="elastique_stem_")
    tmp_path = Path(tmp.name)
    padded = tmp_path / "padded.wav"
    rendered = tmp_path / "rendered.wav"
    in_sec = write_padded_wav(src, padded, pad_sec=pad_sec, target_sr=target_sr)
    if src_dst is not None:
        from align_model.time_map import source_on_padded_wav

        # Model src is an original-stem time. The file above has pad_sec of
        # pre-roll before that time 0, so the marker reads src + pad_sec.
        src_s = source_on_padded_wav(src_dst[0], pad_sec)
        dst_s = src_dst[1]
        markers = [{"src": float(s), "dst": float(d)} for s, d in zip(src_s, dst_s)]
    else:
        markers = markers_from_lag(
            times,
            lags,
            target_sec=target_sec,
            in_sec=in_sec,
            follow=follow_lag,
            scores=scores,
            despike=despike,
            grid_sec=grid_sec,
        )
    # The last marker is not the end. Hold its rate until the source tail
    # has played, and keep the original's length when the source is shorter.
    render_sec = playback_end_sec(markers, in_sec=in_sec, target_sec=target_sec)
    render_frames = max(target_frames, int(round(render_sec * target_sr)))
    render_sec = render_frames / target_sr
    job = {
        "input": str(padded).replace("\\", "/"),
        "output": str(rendered).replace("\\", "/"),
        "target_length_sec": render_sec,
        "sample_rate": target_sr,
        "markers": markers,
    }
    if declick:
        job["declick_chunk"] = RX_DECLICK_FIX_DISCONTINUOUS
    if stretch_mode == "transient":
        job["stretch_mode"] = "transient"
    return _StagedStem(tmp, rendered, dest, target_sr, render_frames, job)


def apply_rx_declick(path: Path, reaper_exe: Path) -> None:
    """Run RX 11 De-click on a finished file. No stretch markers."""
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        return
    tmp = tempfile.TemporaryDirectory(prefix="rx_declick_")
    tmp_path = Path(tmp.name)
    src_wav = tmp_path / "in.wav"
    rendered = tmp_path / "rendered.wav"
    in_sec = write_padded_wav(path, src_wav, pad_sec=0.0, target_sr=info.samplerate)
    frames = max(1, int(round(in_sec * info.samplerate)))
    job = {
        "input": str(src_wav).replace("\\", "/"),
        "output": str(rendered).replace("\\", "/"),
        "target_length_sec": frames / info.samplerate,
        "sample_rate": info.samplerate,
        "markers": [],
        "declick_chunk": RX_DECLICK_FIX_DISCONTINUOUS,
        "declick_only": True,
    }
    staged = _StagedStem(tmp, rendered, path, info.samplerate, frames, job)
    warp_stems_reaper(reaper_exe, [staged])


def _apply_aca_declick(path: Path, method: str, reaper_exe: Path | None) -> None:
    """De-click the acapella after loudness, so the gain does not lift the click."""
    if method == "off":
        return
    if method == "rx":
        if reaper_exe is None:
            raise FileNotFoundError("RX 11 De-click needs REAPER.")
        apply_rx_declick(path, reaper_exe)


def _commit_staged_stem(staged: _StagedStem) -> None:
    rendered = staged.rendered
    if not rendered.is_file():
        tmp_path = Path(staged.tmp.name)
        alt = list(tmp_path.glob("rendered*")) + list(tmp_path.glob("*.wav"))
        alt = [p for p in alt if p.name != "padded.wav"]
        if not alt:
            raise FileNotFoundError(f"REAPER did not produce {rendered}")
        rendered = alt[0]
    y, file_sr = sf.read(str(rendered), always_2d=True, dtype="float32")
    if file_sr != staged.target_sr:
        import librosa

        y = np.stack(
            [
                librosa.resample(y[:, c], orig_sr=file_sr, target_sr=staged.target_sr)
                for c in range(y.shape[1])
            ],
            axis=1,
        ).astype(np.float32)
    y = fit_len_2d(y, staged.target_frames, staged.target_sr)
    dest = staged.dest
    if dest.suffix.lower() == ".flac":
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        sf.write(str(tmp), y, staged.target_sr, format="FLAC")
        tmp.replace(dest)
    else:
        sf.write(str(dest), y, staged.target_sr, subtype="FLOAT")


def warp_stems_reaper(reaper_exe: Path, staged: list[_StagedStem]) -> None:
    """Render every staged stem in one REAPER process, then write the files."""
    if not staged:
        return
    try:
        run_reaper_jobs(
            reaper_exe,
            [item.job for item in staged],
            timeout_sec=480.0,
            expected_outputs=[item.rendered for item in staged],
        )
        for item in staged:
            _commit_staged_stem(item)
    finally:
        for item in staged:
            item.tmp.cleanup()


def warp_stem_reaper(
    src: Path,
    dest: Path,
    *,
    pad_sec: float,
    times: np.ndarray,
    lags: np.ndarray,
    target_sr: int,
    target_frames: int,
    reaper_exe: Path,
    declick: bool = False,
    follow_lag: bool = False,
    scores: np.ndarray | None = None,
    grid_sec: np.ndarray | None = None,
    stretch_mode: str | None = None,
    src_dst: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    staged = stage_stem_reaper(
        src,
        dest,
        pad_sec=pad_sec,
        times=times,
        lags=lags,
        target_sr=target_sr,
        target_frames=target_frames,
        declick=declick,
        follow_lag=follow_lag,
        scores=scores,
        grid_sec=grid_sec,
        stretch_mode=stretch_mode,
        src_dst=src_dst,
    )
    warp_stems_reaper(reaper_exe, [staged])


def reprocess_edited_acapella(
    folder: Path,
    aca_dest: Path,
    *,
    reaper_exe: Path,
    declick: str = "rx",
    on_step=None,
    tag_folder: bool = False,
) -> dict:
    """Élastique, loudness-match, and re-score a manually placed acapella.

    The edited file is the vocal placement. Gap placement is not run again.
    Residual élastique uses the Demucs vocal. The instrumental is left as it is.
    The folder is renamed when ``tag_folder`` is set and the score is pass or fail.
    The returned dict is the new row for the list.
    """
    _aca_bak, _inst_src, orig = find_backup_stems(folder)
    if not aca_dest.is_file():
        raise FileNotFoundError(f"Missing edited acapella: {aca_dest}")
    if not orig:
        raise FileNotFoundError("Cannot re-process without an original.")

    info = sf.info(str(orig))

    from demucs_vocals import load_vocals_mono, match_aligned_stems_to_demucs_isolated

    aca_m = peak_norm(load_mono(aca_dest, SR_ANALYSIS))
    aca_v = peak_norm(bandpass(aca_m, SR_ANALYSIS))
    elastique_ref = peak_norm(
        bandpass(load_vocals_mono(orig, SR_ANALYSIS, folder=folder), SR_ANALYSIS)
    )
    n = min(len(elastique_ref), len(aca_v))
    from align_model.params import resolve_profile
    from align_model.pipeline import map_stem

    profile = resolve_profile(elastique_ref, SR_ANALYSIS)
    aca_map = map_stem(
        elastique_ref[:n],
        aca_v[:n],
        SR_ANALYSIS,
        kind="vocal",
        profile=profile,
        target_sec=n / SR_ANALYSIS,
        in_sec=n / SR_ANALYSIS,
    )
    t_ae, lag_ae, sc_ae = aca_map.times, aca_map.lags, aca_map.scores
    aca_bias = float(aca_map.pad_sec)
    aca_src_dst = (aca_map.src, aca_map.dst)

    declick_plan = resolve_declick(declick, host_rx=True)
    notes = [
        "section_edit=1",
        "aca_elastique_ref=demucs_vocals",
        "inst_unchanged=1",
        declick_plan.note,
    ]
    if len(lag_ae):
        notes.append(f"aca_drift_range_ms={(float(lag_ae.max() - lag_ae.min()) * 1000):.1f}")

    if abs(aca_bias) >= 1e-4:
        notes.append(f"aca_lag_offset={aca_bias:+.3f}s")

    _emit_step(on_step, "align")
    bar_grid, bar_note = marker_bar_grid(orig, info.frames / info.samplerate)
    notes.append(bar_note)
    warp_stem_reaper(
        aca_dest,
        aca_dest,
        pad_sec=aca_bias,
        times=t_ae,
        lags=lag_ae,
        target_sr=info.samplerate,
        target_frames=info.frames,
        reaper_exe=reaper_exe,
        declick=False,
        scores=sc_ae,
        grid_sec=bar_grid,
        src_dst=aca_src_dst,
    )

    try:
        repair_note = _repair_rendered_stems(
            [("aca", aca_dest, load_vocals_mono(orig, SR_ANALYSIS, folder=folder))],
            reaper_exe=reaper_exe,
            target_sr=info.samplerate,
            target_frames=info.frames,
            drift_ms=DEFAULT_DRIFT_MS,
            on_step=on_step,
            grid_sec=bar_grid,
            declick_aca=False,
        )
        if repair_note:
            notes.append(repair_note)
    except Exception as exc:  # noqa: BLE001 — the edited warp still stands
        notes.append(f"repair_skip:{type(exc).__name__}:{exc}")

    time.sleep(0.35)
    _emit_step(on_step, "loudness")
    try:
        loud = match_aligned_stems_to_demucs_isolated(
            orig, aca_dest, None, folder=folder, aca_only=True
        )
        notes.append(f"loudness_match aca={loud['aca_gain_db']:+.1f}dB")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"loudness_match_skip:{type(exc).__name__}:{exc}")

    # After the gain. A click removed first would be raised again with the vocal.
    _apply_aca_declick(aca_dest, declick_plan.method, reaper_exe)
    return _scored_edit_row(folder, notes, on_step, tag_folder=tag_folder)


def reprocess_edited_instrumental(
    folder: Path,
    inst_dest: Path,
    *,
    reaper_exe: Path,
    on_step=None,
    tag_folder: bool = False,
) -> dict:
    """Élastique, loudness-match, and re-score a manually placed instrumental.

    The edited file is the placement. Residual élastique uses the Demucs
    instrumental and the transient stretch. The acapella is left as it is.
    The instrumental is not de-clicked. The folder is renamed when
    ``tag_folder`` is set and the score is pass or fail.
    """
    _aca_bak, _inst_src, orig = find_backup_stems(folder)
    if not inst_dest.is_file():
        raise FileNotFoundError(f"Missing edited instrumental: {inst_dest}")
    if not orig:
        raise FileNotFoundError("Cannot re-process without an original.")

    info = sf.info(str(orig))

    from demucs_vocals import load_instrumental_mono, match_aligned_stems_to_demucs_isolated

    inst_m = peak_norm(load_mono(inst_dest, SR_ANALYSIS))
    inst_ref = peak_norm(load_instrumental_mono(orig, SR_ANALYSIS, folder=folder))
    n = min(len(inst_ref), len(inst_m))
    from align_model.params import resolve_profile
    from align_model.pipeline import map_stem

    profile = resolve_profile(inst_ref, SR_ANALYSIS)
    inst_map = map_stem(
        inst_ref[:n],
        inst_m[:n],
        SR_ANALYSIS,
        kind="instrumental",
        profile=profile,
        target_sec=n / SR_ANALYSIS,
        in_sec=n / SR_ANALYSIS,
    )
    t_i, lag_i, sc_i = inst_map.times, inst_map.lags, inst_map.scores
    inst_bias = float(inst_map.pad_sec)
    inst_src_dst = (inst_map.src, inst_map.dst)

    notes = [
        "section_edit=inst",
        "inst_elastique_ref=demucs_instrumental",
        "inst_stretch=transient",
        "inst_declick=off",
        "aca_unchanged=1",
    ]
    if len(lag_i):
        notes.append(f"inst_drift_range_ms={(float(lag_i.max() - lag_i.min()) * 1000):.1f}")
    if abs(inst_bias) >= 1e-4:
        notes.append(f"inst_lag_offset={inst_bias:+.3f}s")

    _emit_step(on_step, "align")
    bar_grid, bar_note = marker_bar_grid(orig, info.frames / info.samplerate)
    notes.append(bar_note)
    warp_stem_reaper(
        inst_dest,
        inst_dest,
        pad_sec=inst_bias,
        times=t_i,
        lags=lag_i,
        target_sr=info.samplerate,
        target_frames=info.frames,
        reaper_exe=reaper_exe,
        declick=False,
        scores=sc_i,
        grid_sec=bar_grid,
        stretch_mode="transient",
        src_dst=inst_src_dst,
    )

    try:
        repair_note = _repair_rendered_stems(
            [
                (
                    "inst",
                    inst_dest,
                    load_instrumental_mono(orig, SR_ANALYSIS, folder=folder),
                )
            ],
            reaper_exe=reaper_exe,
            target_sr=info.samplerate,
            target_frames=info.frames,
            drift_ms=DEFAULT_DRIFT_MS,
            on_step=on_step,
            grid_sec=bar_grid,
            declick_aca=False,
        )
        if repair_note:
            notes.append(repair_note)
    except Exception as exc:  # noqa: BLE001 — the edited warp still stands
        notes.append(f"repair_skip:{type(exc).__name__}:{exc}")

    time.sleep(0.35)
    _emit_step(on_step, "loudness")
    try:
        loud = match_aligned_stems_to_demucs_isolated(
            orig, None, inst_dest, folder=folder, inst_only=True
        )
        notes.append(f"loudness_match inst={loud['inst_gain_db']:+.1f}dB")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"loudness_match_skip:{type(exc).__name__}:{exc}")

    return _scored_edit_row(folder, notes, on_step, tag_folder=tag_folder)


def _scored_edit_row(folder: Path, notes: list[str], on_step, *, tag_folder: bool = False) -> dict:
    """Re-score both stems after a section edit.

    When ``tag_folder`` is set, a pass or fail renames the folder to match.
    """
    aca, inst, orig2, scan_notes = scan_folder(folder)
    if not aca or not inst or not orig2:
        notes.append(f"post_scan:{scan_notes}")
        return {
            "folder": folder.name,
            "path": str(folder),
            "verdict": "error",
            "aca_verdict": "error",
            "inst_verdict": "error",
            "notes": "; ".join(notes),
        }
    _emit_step(on_step, "score")
    scored = score_rendered_pair(
        aca,
        inst,
        orig2,
        folder=folder,
        corr_min=DEFAULT_CORR_MIN,
        drift_ms=DEFAULT_DRIFT_MS,
        window_corr_min=DEFAULT_WINDOW_CORR_MIN,
        weak_window_frac=DEFAULT_WEAK_WINDOW_FRAC,
    )
    mix = scored["mix"]
    stem = scored["stem"]
    verdict = mix["verdict"]
    corr = mix["corr"]
    drift = mix["drift_ms"]
    check_notes = mix["notes"]
    aca_verdict = stem["aca_verdict"]
    inst_verdict = stem["inst_verdict"]
    combined = stem["combined"]
    aca_lag = stem["aca_lag"]
    inst_lag = stem["inst_lag"]
    stem_notes = stem["notes"]
    aca_points = stem["aca_checkpoints"]
    inst_points = stem["inst_checkpoints"]
    notes.append(stem_notes)
    notes.append(f"check={verdict} corr={corr:.3f} drift={drift:.1f}ms; {check_notes}")
    live = folder
    moved_to = ""
    if tag_folder and combined in ("pass", "fail"):
        _emit_step(on_step, "tag")
        try:
            dest = tag_folder_verdict(folder, combined)
        except FileExistsError:
            notes.append("dest_exists")
        except OSError as exc:
            notes.append(f"rename_failed:{exc}")
        else:
            if dest is not None:
                live = dest
                moved_to = str(dest)
    row = {
        "folder": live.name,
        "path": str(live),
        "verdict": combined,
        "aca_verdict": aca_verdict,
        "inst_verdict": inst_verdict,
        "corr": corr,
        "drift_ms": drift,
        "aca_lag_sec": aca_lag,
        "inst_lag_sec": inst_lag,
        "aca_checkpoints": aca_points,
        "inst_checkpoints": inst_points,
        "notes": "; ".join(notes),
    }
    if moved_to:
        row["moved_to"] = moved_to
    return row


def _emit_step(on_step, step_id: str) -> None:
    if on_step is None:
        return
    try:
        on_step(step_id)
    except Exception:  # noqa: BLE001 — UI callback must not abort processing
        pass


def _repair_rendered_stems(
    jobs: list[tuple[str, Path, np.ndarray]],
    *,
    reaper_exe: Path,
    target_sr: int,
    target_frames: int,
    drift_ms: float,
    on_step=None,
    grid_sec: np.ndarray | None = None,
    declick_aca: bool = False,
) -> str:
    """Warp once more where the onset checkpoints are still outside the limit.

    Each job is a label, the file just rendered, and the same reference the
    pass/fail check uses. A stem already inside the limit is not rendered again.
    """
    notes: list[str] = []
    pending: list[tuple[Path, float, np.ndarray, np.ndarray, str | None, bool]] = []
    for label, dest, reference in jobs:
        if reference is None or len(reference) == 0 or not dest.is_file():
            notes.append(f"{label}_repair=skip")
            continue
        try:
            query = load_mono(dest, SR_ANALYSIS)
        except Exception as exc:  # noqa: BLE001 — a locked render must not abort the song
            notes.append(f"{label}_repair_skip:{type(exc).__name__}")
            continue
        n = min(len(reference), len(query))
        status, times, lags, detail = plan_residual_repair(
            np.asarray(reference[:n], dtype=np.float32),
            query[:n],
            SR_ANALYSIS,
            drift_ms,
        )
        if status != "repair" or times is None or lags is None:
            extra = f" {detail}" if detail else ""
            notes.append(f"{label}_repair={status}{extra}")
            continue
        lags, pad = absorb_lag_offset(times, lags)
        notes.append(f"{label}_repair=1 pad={pad * 1000:+.0f}ms {detail}")
        stretch = "transient" if label == "inst" else None
        declick = declick_aca and label != "inst"
        pending.append((dest, float(pad), times, lags, stretch, declick))
    if not pending:
        return "; ".join(notes)

    _emit_step(on_step, "repair")
    staged: list[_StagedStem] = []
    try:
        for dest, pad, times, lags, stretch, declick in pending:
            staged.append(
                stage_stem_reaper(
                    dest,
                    dest,
                    pad_sec=pad,
                    times=times,
                    lags=lags,
                    target_sr=target_sr,
                    target_frames=target_frames,
                    declick=declick,
                    follow_lag=True,
                    despike=False,
                    stretch_mode=stretch,
                    grid_sec=grid_sec,
                )
            )
        warp_stems_reaper(reaper_exe, staged)
    finally:
        for item in staged:
            item.tmp.cleanup()
    return "; ".join(notes)


def _process_folder_model(
    result: WarpResult,
    *,
    folder: Path,
    reaper_exe: Path,
    max_pad_sec: float,
    corr_min: float,
    drift_ms: float,
    window_corr_min: float,
    weak_window_frac: float,
    move_on_pass: bool,
    dry_run: bool,
    use_demucs_vocals: bool,
    gaps_cut: bool,
    declick: str,
    on_step,
    aca_src: Path,
    inst_src: Path,
    orig: Path,
) -> None:
    """Evidence, robust time map, gap-aware vocal render, then the usual checks."""
    from align_model.beats import load_beats
    from align_model.gaps import write_gap_file
    from align_model.params import resolve_profile
    from align_model.pipeline import combine_reports, map_stem, map_vocal_gaps
    from align_model.quality import notes_fragment, stamp_result

    orig_m = peak_norm(load_mono(orig, SR_ANALYSIS))
    aca_m = peak_norm(load_mono(aca_src, SR_ANALYSIS))
    inst_m = peak_norm(load_mono(inst_src, SR_ANALYSIS))
    orig_vox = peak_norm(bandpass(orig_m, SR_ANALYSIS))
    aca_vox = peak_norm(bandpass(aca_m, SR_ANALYSIS))
    profile = resolve_profile(
        orig_m,
        SR_ANALYSIS,
        max_pad_sec=max_pad_sec,
    )
    max_pad_sec = profile.max_pad_sec
    # Chroma is a diagnostic only. The model sees the unshifted stem and
    # its own offset is the only pad the renderer adds.
    aca_chroma, aca_score = chroma_xcorr_pad(
        orig_vox, aca_vox, sr=SR_ANALYSIS, max_pad_sec=max_pad_sec
    )
    if use_demucs_vocals:
        from demucs_vocals import load_instrumental_mono, load_vocals_mono

        _emit_step(on_step, "demucs")
        inst_ref = peak_norm(load_instrumental_mono(orig, SR_ANALYSIS, folder=folder))
        vocal_ref = peak_norm(bandpass(load_vocals_mono(orig, SR_ANALYSIS, folder=folder), SR_ANALYSIS))
        inst_ref_note = "demucs_instrumental"
        vocal_ref_note = "demucs_vocals"
    else:
        inst_ref = orig_m
        vocal_ref = orig_vox
        inst_ref_note = "mix"
        vocal_ref_note = "mix"
    inst_chroma, inst_score = chroma_xcorr_pad(
        inst_ref, inst_m, sr=SR_ANALYSIS, max_pad_sec=max_pad_sec
    )
    beats, downbeats = load_beats(orig)
    inst_map = map_stem(
        inst_ref,
        inst_m,
        SR_ANALYSIS,
        kind="instrumental",
        profile=profile,
        beats=beats,
        downbeats=downbeats,
        target_sec=len(inst_ref) / SR_ANALYSIS,
        in_sec=len(inst_m) / SR_ANALYSIS,
    )
    inst_pad = float(inst_map.pad_sec)
    if gaps_cut:
        _emit_step(on_step, "silence")
        aca_map = map_vocal_gaps(
            aca_m,
            vocal_ref,
            SR_ANALYSIS,
            profile=profile,
            target_sec=len(vocal_ref) / SR_ANALYSIS,
            beats=beats,
            downbeats=downbeats,
        )
        result.aca_pad_sec = float(aca_map.pad_sec)
    else:
        aca_map = map_stem(
            orig_vox,
            aca_vox,
            SR_ANALYSIS,
            kind="vocal",
            profile=profile,
            beats=beats,
            downbeats=downbeats,
            target_sec=len(orig_vox) / SR_ANALYSIS,
            in_sec=len(aca_vox) / SR_ANALYSIS,
        )
        result.aca_pad_sec = float(aca_map.pad_sec)
    result.inst_pad_sec = inst_pad
    if len(aca_map.lags):
        result.aca_drift_range_ms = float((float(np.max(aca_map.lags)) - float(np.min(aca_map.lags))) * 1000)
    if len(inst_map.lags):
        result.inst_drift_range_ms = float((float(np.max(inst_map.lags)) - float(np.min(inst_map.lags))) * 1000)
    report = combine_reports(aca_map.report, inst_map.report, profile=profile.name)
    if len(beats):
        report["beats"] = [round(float(v), 3) for v in beats if v >= 0]
        report["downbeats"] = [round(float(v), 3) for v in downbeats if v >= 0]
    result.alignment_report = report
    gap_tag = "+aca_gaps" if gaps_cut else ""
    result.notes = (
        f"engine=reaper-model{gap_tag}; "
        f"aca_score={aca_score:.3f}; inst_score={inst_score:.3f}; "
        f"chroma_pad=({aca_chroma:+.3f},{inst_chroma:+.3f}); "
        f"{notes_fragment(report)}"
    )
    result.notes += "; aca_gaps=1" if gaps_cut else "; aca_gaps=0"
    result.notes += f"; inst_elastique_ref={inst_ref_note}; aca_ref={vocal_ref_note}"
    inserts = report.get("gap_inserts") or []
    if inserts:
        shown = ",".join(f"{float(span[1]) - float(span[0]):.3f}" for span in inserts[:6])
        result.notes += f"; aca_gap_inserts={shown}"
    declick_plan = resolve_declick(declick, host_rx=True)
    result.notes += f"; {declick_plan.note}; inst_declick=off"
    if dry_run:
        result.verdict = "dry_run"
        return

    info = sf.info(str(orig))
    aca_dest = folder / aca_src.name
    inst_dest = folder / inst_src.name
    _emit_step(on_step, "align")
    bar_grid, bar_note = marker_bar_grid(orig, info.frames / info.samplerate)
    result.notes += f"; {bar_note}; inst_stretch=transient"
    staged: list[_StagedStem] = []
    try:
        if gaps_cut and aca_map.spans:
            gap_notes = write_gap_file(
                aca_src,
                aca_dest,
                aca_map.spans,
                target_sr=info.samplerate,
                target_frames=info.frames,
                target_channels=info.channels,
                engine="reaper",
                reaper_exe=reaper_exe,
            )
            if gap_notes:
                result.notes += "; " + "; ".join(gap_notes)
        else:
            staged.append(
                stage_stem_reaper(
                    aca_src,
                    aca_dest,
                    pad_sec=float(result.aca_pad_sec or 0.0),
                    times=aca_map.times,
                    lags=aca_map.lags,
                    target_sr=info.samplerate,
                    target_frames=info.frames,
                    scores=aca_map.scores,
                    grid_sec=bar_grid,
                    src_dst=(aca_map.src, aca_map.dst),
                )
            )
        staged.append(
            stage_stem_reaper(
                inst_src,
                inst_dest,
                pad_sec=inst_pad,
                times=inst_map.times,
                lags=inst_map.lags,
                target_sr=info.samplerate,
                target_frames=info.frames,
                scores=inst_map.scores,
                stretch_mode="transient",
                grid_sec=bar_grid,
                src_dst=(inst_map.src, inst_map.dst),
            )
        )
        warp_stems_reaper(reaper_exe, staged)
    finally:
        for item in staged:
            item.tmp.cleanup()

    try:
        from demucs_vocals import load_instrumental_mono, load_vocals_mono

        # Gap-aware aca already sits on Mel phrase attacks. A global residual
        # warp keyed off second-half spikes re-delays a good first half
        # (Groove Thang ~50 ms late at 0:48). Only repair the instrumental.
        repair_jobs: list[tuple[str, Path, np.ndarray]] = [
            ("inst", inst_dest, load_instrumental_mono(orig, SR_ANALYSIS, folder=folder)),
        ]
        if not gaps_cut:
            repair_jobs.insert(
                0,
                ("aca", aca_dest, load_vocals_mono(orig, SR_ANALYSIS, folder=folder)),
            )
        else:
            result.notes += "; aca_repair=skip_gaps"
        repair_note = _repair_rendered_stems(
            repair_jobs,
            reaper_exe=reaper_exe,
            target_sr=info.samplerate,
            target_frames=info.frames,
            drift_ms=drift_ms,
            on_step=on_step,
            grid_sec=bar_grid,
            declick_aca=False,
        )
        if repair_note:
            result.notes += f"; {repair_note}"
    except Exception as exc:  # noqa: BLE001 — the first warp still stands
        result.notes += f"; repair_skip:{type(exc).__name__}:{exc}"

    time.sleep(0.35)
    _emit_step(on_step, "loudness")
    try:
        from demucs_vocals import match_aligned_stems_to_demucs_isolated

        loud = match_aligned_stems_to_demucs_isolated(orig, aca_dest, inst_dest, folder=folder)
        result.notes += (
            f"; loudness_match aca={loud['aca_gain_db']:+.1f}dB inst={loud['inst_gain_db']:+.1f}dB"
        )
    except Exception as exc:  # noqa: BLE001
        result.notes += f"; loudness_match_skip:{type(exc).__name__}:{exc}"

    # De-click under LOUDNESS (after gain) so that step absorbs RX time; score
    # stays on the alignment check only. Order still: gain → de-click → score.
    _apply_aca_declick(aca_dest, declick_plan.method, reaper_exe)
    _emit_step(on_step, "score")
    aca, inst, orig2, scan_notes = scan_folder(folder)
    if not aca or not inst or not orig2:
        result.verdict = "error"
        result.notes += f"; post_scan:{scan_notes}"
        return
    from align_model.quality import stamp_result

    mix = apply_full_score(
        result,
        aca,
        inst,
        orig2,
        folder=folder,
        corr_min=corr_min,
        drift_ms=drift_ms,
        window_corr_min=window_corr_min,
        weak_window_frac=weak_window_frac,
    )
    mix_verdict, corr, lag, drift = (
        mix["verdict"],
        mix["corr"],
        mix["lag_sec"],
        mix["drift_ms"],
    )
    if isinstance(result.alignment_report, dict):
        result.alignment_report["validation"] = "post_render"
        result.alignment_report["post_render"] = {
            "verdict": mix_verdict,
            "corr": corr,
            "lag_sec": lag,
            "drift_ms": drift,
        }
    stamp_result(result, mix_verdict=mix_verdict, mix_corr=corr, mix_lag=lag, mix_drift=drift)
    if move_on_pass and result.verdict in ("pass", "fail"):
        _emit_step(on_step, "tag")
        try:
            dest = tag_folder_verdict(folder, result.verdict)
        except FileExistsError:
            result.notes += "; dest_exists"
        except OSError as exc:
            result.notes += f"; rename_failed:{exc}"
        else:
            if dest is not None:
                result.moved_to = str(dest)


def process_folder(
    folder: Path,
    *,
    reaper_exe: Path,
    max_pad_sec: float,
    corr_min: float,
    drift_ms: float,
    window_corr_min: float,
    weak_window_frac: float,
    move_on_pass: bool,
    dry_run: bool,
    use_demucs_vocals: bool = True,
    gaps_cut: bool = True,
    declick: str = "rx",
    on_step=None,
    legacy_lag: bool = False,
) -> WarpResult:
    result = WarpResult(folder=folder.name)
    aca_src, inst_src, orig = find_backup_stems(folder)
    if not aca_src or not inst_src or not orig:
        missing = []
        if not aca_src:
            missing.append("aca_backup")
        if not inst_src:
            missing.append("inst_backup")
        if not orig:
            missing.append("original")
        result.verdict = "skip"
        result.notes = f"missing:{','.join(missing)}"
        return result

    try:
        if not legacy_lag:
            _process_folder_model(
                result,
                folder=folder,
                reaper_exe=reaper_exe,
                max_pad_sec=max_pad_sec,
                corr_min=corr_min,
                drift_ms=drift_ms,
                window_corr_min=window_corr_min,
                weak_window_frac=weak_window_frac,
                move_on_pass=move_on_pass,
                dry_run=dry_run,
                use_demucs_vocals=use_demucs_vocals,
                gaps_cut=gaps_cut,
                declick=declick,
                on_step=on_step,
                aca_src=aca_src,
                inst_src=inst_src,
                orig=orig,
            )
            return result

        orig_m = peak_norm(load_mono(orig, SR_ANALYSIS))
        aca_m = peak_norm(load_mono(aca_src, SR_ANALYSIS))
        inst_m = peak_norm(load_mono(inst_src, SR_ANALYSIS))
        orig_vox = peak_norm(bandpass(orig_m, SR_ANALYSIS))
        aca_vox = peak_norm(bandpass(aca_m, SR_ANALYSIS))

        aca_pad, aca_score = chroma_xcorr_pad(
            orig_vox, aca_vox, sr=SR_ANALYSIS, max_pad_sec=max_pad_sec
        )
        if use_demucs_vocals:
            from demucs_vocals import load_instrumental_mono

            _emit_step(on_step, "demucs")
            inst_ref = peak_norm(load_instrumental_mono(orig, SR_ANALYSIS, folder=folder))
            inst_ref_note = "demucs_instrumental"
        else:
            inst_ref = orig_m
            inst_ref_note = "mix"
        inst_pad, inst_score = chroma_xcorr_pad(
            inst_ref, inst_m, sr=SR_ANALYSIS, max_pad_sec=max_pad_sec
        )
        result.aca_pad_sec = aca_pad
        result.inst_pad_sec = inst_pad

        aca_p = fit_len(pad_or_trim_front(aca_m, aca_pad, SR_ANALYSIS), len(orig_m))
        inst_p = fit_len(pad_or_trim_front(inst_m, inst_pad, SR_ANALYSIS), len(inst_ref))
        aca_pv = peak_norm(bandpass(peak_norm(aca_p), SR_ANALYSIS))

        t_a, lag_a, sc_a = lag_curve(orig_vox, aca_pv, sr=SR_ANALYSIS)
        t_i, lag_i, sc_i = lag_curve(inst_ref, peak_norm(inst_p), sr=SR_ANALYSIS)
        t_a, lag_a = smooth_lags(t_a, lag_a, sc_a)
        t_i, lag_i = smooth_lags(t_i, lag_i, sc_i)
        lag_i, inst_bias = absorb_lag_offset(t_i, lag_i, sc_i)
        inst_pad += inst_bias
        inst_for_onset = fit_len(pad_or_trim_front(inst_m, inst_pad, SR_ANALYSIS), len(inst_ref))
        t_o, lag_o, sc_o = onset_lag_curve(inst_ref, peak_norm(inst_for_onset), sr=SR_ANALYSIS)
        if len(lag_o) >= 2:
            lag_o, onset_bias = absorb_lag_offset(t_o, lag_o, sc_o)
            inst_pad += onset_bias
            t_i, lag_i, sc_i = t_o, lag_o, sc_o
            if abs(onset_bias) >= 1e-4:
                inst_bias += onset_bias
        result.inst_pad_sec = inst_pad

        # Beat grid is off. A lag window that disagreed with nearby beats was
        # replaced, so a real offset against the Demucs split was dropped.
        # Both stems keep the lag measured against those splits.

        if len(lag_a):
            result.aca_drift_range_ms = float((lag_a.max() - lag_a.min()) * 1000)
        if len(lag_i):
            result.inst_drift_range_ms = float((lag_i.max() - lag_i.min()) * 1000)

        gap_tag = "+aca_gaps" if gaps_cut else ""
        result.notes = (
            f"engine=reaper-elastique{gap_tag}; "
            f"aca_score={aca_score:.3f}; inst_score={inst_score:.3f}"
        )
        result.notes += "; aca_gaps=1" if gaps_cut else "; aca_gaps=0"
        if abs(inst_bias) >= 1e-4:
            result.notes += f"; inst_lag_offset={inst_bias:+.3f}s"
        result.notes += "; beat_layer=off"
        declick_plan = resolve_declick(declick, host_rx=True)
        result.notes += f"; {declick_plan.note}"

        if dry_run:
            result.verdict = "dry_run"
            return result

        info = sf.info(str(orig))
        aca_dest = folder / aca_src.name
        if gaps_cut:
            from aca_gap_align import write_gap_aligned_acapella

            _emit_step(on_step, "silence")
            gap_info = write_gap_aligned_acapella(
                aca_src,
                orig,
                aca_dest,
                folder=folder,
                use_demucs_vocals=use_demucs_vocals,
            )
            plan = gap_info.get("plan", {})
            result.aca_pad_sec = float(plan.get("front_pad_sec", aca_pad))
            gap_notes = gap_info.get("notes") or []
            ref_note = next((n for n in gap_notes if n.startswith("ref_src=")), "")
            result.notes += (
                f"; aca_front={plan.get('front_pad_sec', 0):.3f}s"
                f"; aca_segs={plan.get('n_segments', 0)}"
            )
            if ref_note:
                result.notes += f"; {ref_note}"
            gaps = plan.get("gaps") or []
            if gaps:
                inserts = ",".join(f"{g['insert_sec']:.3f}" for g in gaps[:6])
                result.notes += f"; aca_gap_inserts={inserts}"
            aca_gap_m = peak_norm(load_mono(aca_dest, SR_ANALYSIS))
            aca_gap_v = peak_norm(bandpass(aca_gap_m, SR_ANALYSIS))
            if use_demucs_vocals:
                from demucs_vocals import load_vocals_mono

                elastique_ref = peak_norm(
                    bandpass(load_vocals_mono(orig, SR_ANALYSIS, folder=folder), SR_ANALYSIS)
                )
            else:
                elastique_ref = orig_vox
            n = min(len(elastique_ref), len(aca_gap_v))
            t_ae, lag_ae, sc_ae = lag_curve(
                elastique_ref[:n], aca_gap_v[:n], sr=SR_ANALYSIS
            )
            t_ae, lag_ae = smooth_lags(t_ae, lag_ae, sc_ae)
            vocal_base_pad = 0.0
        else:
            shutil.copy2(aca_src, aca_dest)
            t_ae, lag_ae, sc_ae = t_a, lag_a, sc_a
            vocal_base_pad = aca_pad

        # Vocal lag against the Demucs vocal, with no beat-grid rewrite.
        _emit_step(on_step, "align")
        lag_ae, extra = absorb_lag_offset(t_ae, lag_ae, sc_ae)
        aca_warp_pad = vocal_base_pad + extra
        elastique_ref_note = "demucs_vocals" if use_demucs_vocals else "mix"
        if not gaps_cut:
            result.aca_pad_sec = aca_warp_pad
        if len(lag_ae):
            result.aca_drift_range_ms = float((lag_ae.max() - lag_ae.min()) * 1000)
        if abs(aca_warp_pad - vocal_base_pad) >= 1e-4:
            result.notes += f"; aca_lag_offset={aca_warp_pad - vocal_base_pad:+.3f}s"
        # Both stems share the original's 8-bar downbeats as marker times.
        # The lag at each mark is still the measured lag.
        bar_grid, bar_note = marker_bar_grid(orig, info.frames / info.samplerate)
        result.notes += (
            f"; aca_elastique=1; aca_elastique_ref={elastique_ref_note}"
            f"; {bar_note}"
        )

        # Instrumental de-click is off. It treats drum attacks as clicks.
        result.notes += f"; inst_elastique_ref={inst_ref_note}"
        result.notes += "; inst_elastique=3.3.3; inst_stretch=transient"
        result.notes += "; inst_declick=off"
        inst_dest = folder / inst_src.name
        # One REAPER process renders both stems. Starting it twice was most
        # of the wait; élastique and RX stay on the CPU either way.
        staged: list[_StagedStem] = []
        try:
            staged.append(
                stage_stem_reaper(
                    aca_dest,
                    aca_dest,
                    pad_sec=aca_warp_pad,
                    times=t_ae,
                    lags=lag_ae,
                    target_sr=info.samplerate,
                    target_frames=info.frames,
                    declick=False,
                    follow_lag=True,
                    scores=sc_ae,
                    grid_sec=bar_grid,
                )
            )
            staged.append(
                stage_stem_reaper(
                    inst_src,
                    inst_dest,
                    pad_sec=inst_pad,
                    times=t_i,
                    lags=lag_i,
                    target_sr=info.samplerate,
                    target_frames=info.frames,
                    declick=False,
                    follow_lag=True,
                    scores=sc_i,
                    stretch_mode="transient",
                    grid_sec=bar_grid,
                )
            )
            warp_stems_reaper(reaper_exe, staged)
        finally:
            for item in staged:
                item.tmp.cleanup()

        try:
            from demucs_vocals import load_instrumental_mono, load_vocals_mono

            repair_jobs: list[tuple[str, Path, np.ndarray]] = [
                (
                    "inst",
                    inst_dest,
                    load_instrumental_mono(orig, SR_ANALYSIS, folder=folder),
                ),
            ]
            if not gaps_cut:
                repair_jobs.insert(
                    0,
                    ("aca", aca_dest, load_vocals_mono(orig, SR_ANALYSIS, folder=folder)),
                )
            else:
                result.notes += "; aca_repair=skip_gaps"
            repair_note = _repair_rendered_stems(
                repair_jobs,
                reaper_exe=reaper_exe,
                target_sr=info.samplerate,
                target_frames=info.frames,
                drift_ms=drift_ms,
                on_step=on_step,
                grid_sec=bar_grid,
                declick_aca=False,
            )
            if repair_note:
                result.notes += f"; {repair_note}"
        except Exception as exc:  # noqa: BLE001 — the first warp still stands
            result.notes += f"; repair_skip:{type(exc).__name__}:{exc}"

        # Match loud-phrase amplitude to the Demucs vocal and instrumental
        # Brief pause so Windows/AV releases handles on freshly written FLACs
        time.sleep(0.35)
        _emit_step(on_step, "loudness")
        try:
            from demucs_vocals import match_aligned_stems_to_demucs_isolated

            loud = match_aligned_stems_to_demucs_isolated(
                orig, aca_dest, inst_dest, folder=folder
            )
            result.notes += (
                f"; loudness_match aca={loud['aca_gain_db']:+.1f}dB"
                f" inst={loud['inst_gain_db']:+.1f}dB"
            )
        except Exception as exc:  # noqa: BLE001
            result.notes += f"; loudness_match_skip:{type(exc).__name__}:{exc}"

        # De-click under LOUDNESS (after gain) so that step absorbs RX time; score
        # stays on the alignment check only. Order still: gain → de-click → score.
        _apply_aca_declick(aca_dest, declick_plan.method, reaper_exe)
        _emit_step(on_step, "score")
        aca, inst, orig2, scan_notes = scan_folder(folder)
        if not aca or not inst or not orig2:
            result.verdict = "error"
            result.notes += f"; post_scan:{scan_notes}"
            return result

        apply_full_score(
            result,
            aca,
            inst,
            orig2,
            folder=folder,
            corr_min=corr_min,
            drift_ms=drift_ms,
            window_corr_min=window_corr_min,
            weak_window_frac=weak_window_frac,
        )
        verdict = result.verdict

        if move_on_pass and verdict in ("pass", "fail"):
            _emit_step(on_step, "tag")
            try:
                dest = tag_folder_verdict(folder, verdict)
            except FileExistsError:
                result.notes += "; dest_exists"
            except OSError as exc:
                result.notes += f"; rename_failed:{exc}"
            else:
                if dest is not None:
                    result.moved_to = str(dest)

    except Exception as exc:  # noqa: BLE001
        result.verdict = "error"
        result.notes += f"; {type(exc).__name__}:{exc}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=FAIL_ALL)
    ap.add_argument("--reaper-exe", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-move", action="store_true")
    ap.add_argument("--max-pad-sec", type=float, default=DEFAULT_MAX_PAD_SEC)
    ap.add_argument("--corr-min", type=float, default=DEFAULT_CORR_MIN)
    ap.add_argument("--drift-ms", type=float, default=DEFAULT_DRIFT_MS)
    ap.add_argument("--window-corr-min", type=float, default=DEFAULT_WINDOW_CORR_MIN)
    ap.add_argument("--weak-window-frac", type=float, default=DEFAULT_WEAK_WINDOW_FRAC)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument(
        "--use-demucs-vocals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Place the acapella on the Mel-Band vocal and warp the instrumental to the "
        "Mel-Band instrumental. Pass --no-use-demucs-vocals to use the full mix instead.",
    )
    ap.add_argument(
        "--declick",
        choices=("rx", "off"),
        default="rx",
        help="Acapella de-click. rx uses RX 11 De-click when it is installed, "
        "otherwise the acapella is left untouched. off always leaves it untouched. "
        "The instrumental is not de-clicked.",
    )
    ap.add_argument(
        "--gaps-cut",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="The acapella has rests removed between phrases. "
        "--no-gaps-cut keeps those rests and locks the vocal to the instrumental.",
    )
    ap.add_argument("--csv", type=Path, default=OUT_DIR / "fail_all_reaper_elastique.csv")
    ap.add_argument(
        "--legacy-lag",
        action="store_true",
        help="Use the previous chroma lag path instead of the alignment model.",
    )
    args = ap.parse_args()

    reaper_exe = find_reaper(args.reaper_exe)
    declick_plan = resolve_declick(args.declick, host_rx=True)
    print("Reference: Mel-Band RoFormer")
    print(f"REAPER: {reaper_exe}")
    print(f"De-click: {declick_plan.summary}")
    print(f"Lua:    {LUA_WORKER}")

    root = args.root
    if not root.is_dir():
        print(f"ERROR: {root}", file=sys.stderr)
        return 2

    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if args.only:
        needles = [n.lower() for n in args.only]
        folders = [f for f in folders if any(n in f.name.lower() for n in needles)]
    if args.limit is not None:
        folders = folders[: args.limit]

    print(f"Folders: {len(folders)}  dry_run={args.dry_run}")
    print("NOTE: REAPER path is sequential (one instance); expect ~1–3 min per stem.")

    rows = []
    for i, folder in enumerate(folders, 1):
        r = process_folder(
            folder,
            reaper_exe=reaper_exe,
            max_pad_sec=args.max_pad_sec,
            corr_min=args.corr_min,
            drift_ms=args.drift_ms,
            window_corr_min=args.window_corr_min,
            weak_window_frac=args.weak_window_frac,
            move_on_pass=not args.no_move and not args.dry_run,
            dry_run=args.dry_run,
            use_demucs_vocals=args.use_demucs_vocals,
            gaps_cut=args.gaps_cut,
            declick=args.declick,
            legacy_lag=args.legacy_lag,
        )
        rows.append(asdict(r))
        moved = f" -> {Path(r.moved_to).name}" if r.moved_to else ""
        print(
            f"[{r.verdict.upper()}] {r.folder} corr={r.corr:.3f} drift={r.drift_ms:.1f}ms "
            f"pads=({_pad_text(r.aca_pad_sec)},{_pad_text(r.inst_pad_sec)}) ({i}/{len(folders)}){moved}",
            flush=True,
        )

    write_csv(args.csv, rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("---")
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
