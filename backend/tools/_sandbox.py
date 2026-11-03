"""
tools/_sandbox.py — per-process resource limits for shell_exec's child
process, split by platform since there is no single stdlib primitive
that covers both.
"""

from __future__ import annotations

import sys
from typing import Callable

from backend.config import SHELL_CPU_SECONDS, SHELL_MAX_PROCS, SHELL_MEM_MB, SHELL_SANDBOX_ENABLED
from backend.logging_config import get_logger

logger = get_logger("sandbox")

IS_WINDOWS = sys.platform == "win32"

_warned_once = False


def _warn_once(msg: str) -> None:
    global _warned_once
    logger.warning(msg)
    _warned_once = True


# ── POSIX: resource.setrlimit via preexec_fn ────────────────────────────

def posix_preexec_fn() -> Callable[[], None] | None:
    """
    Returns a preexec_fn for subprocess/create_subprocess_* that applies
    CPU-time, address-space, and process-count limits in the forked
    child before exec. Returns None on Windows (caller must use
    apply_windows_job_limits instead) or if the `resource` module can't
    be imported for some other reason.
    """
    if IS_WINDOWS or not SHELL_SANDBOX_ENABLED:
        return None

    try:
        import resource  # POSIX-only; imported lazily so this module
                          # loads fine on Windows.
    except ImportError:
        _warn_once("resource module unavailable — shell_exec will run unsandboxed on this platform.")
        return None

    mem_bytes = SHELL_MEM_MB * 1024 * 1024

    def _limit() -> None:
        # Each setrlimit call is independent and best-effort: a kernel
        # that refuses one limit (e.g. RLIMIT_AS under some sandboxes)
        # shouldn't prevent the others from applying. Exceptions here
        # run in the forked child, so they're deliberately swallowed —
        # raising would just kill the child before exec with no useful
        # message back to the parent.
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (SHELL_CPU_SECONDS, SHELL_CPU_SECONDS))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (SHELL_MAX_PROCS, SHELL_MAX_PROCS))
        except Exception:
            pass

    return _limit


# ── Windows: Job Object via pywin32 ─────────────────────────────────────

def apply_windows_job_limits(pid: int) -> bool:
    """
    Creates a Job Object with CPU-time/memory/process-count limits and
    assigns the already-spawned process `pid` to it. Returns True on
    success, False if pywin32 isn't installed or any step fails (in
    which case the process keeps running unsandboxed — fail open).

    The job handle is intentionally not closed/tracked here: Windows
    tears the job down once the last handle to it closes, and keeping
    the limits enforced for the lifetime of the process (including any
    children it spawns, which inherit job membership) is the point —
    there's nothing more for this module to do with it afterward.
    """
    if not IS_WINDOWS or not SHELL_SANDBOX_ENABLED:
        return False

    try:
        import win32api
        import win32con
        import win32job
    except ImportError:
        _warn_once("pywin32 not installed — shell_exec will run unsandboxed on Windows. `pip install pywin32` to enable Job Object limits.")
        return False

    try:
        job = win32job.CreateJobObject(None, "")

        info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        info["BasicLimitInformation"]["LimitFlags"] = (
            win32job.JOB_OBJECT_LIMIT_PROCESS_TIME
            | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        # PerProcessUserTimeLimit is in 100-nanosecond units.
        info["BasicLimitInformation"]["PerProcessUserTimeLimit"] = SHELL_CPU_SECONDS * 10_000_000
        info["BasicLimitInformation"]["ActiveProcessLimit"] = SHELL_MAX_PROCS
        info["ProcessMemoryLimit"] = SHELL_MEM_MB * 1024 * 1024
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)

        handle = win32api.OpenProcess(
            win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE,
            False,
            pid,
        )
        win32job.AssignProcessToJobObject(job, handle)
        return True
    except Exception as exc:
        logger.warning("failed to apply Windows Job Object limits to pid %s: %s", pid, exc)
        return False