"""
tools/shell.py — shell_exec: run a subprocess, capture stdout + stderr.

Security hardening
──────────────────────────────────────────────────────────────────────
Previously this always ran via asyncio.create_subprocess_shell(cmd),
i.e. full shell interpretation of whatever string the model supplied —
pipes, redirects, command substitution, globbing, the works. That's
still available (use_shell=True, now the explicit default to preserve
existing behavior), but:

  - When backend.config.FS_WORKSPACE_ROOT is set, `cwd` is now
    containment-checked the same way tools/fs.py checks its paths —
    a command can't be launched with a working directory outside the
    configured workspace.
  - A new `use_shell` flag lets a caller opt into argv-exec mode
    (asyncio.create_subprocess_exec, cmd split with shlex) for simple
    commands that don't need shell features — no shell metacharacter
    interpretation at all in that mode. This is NOT the default
    because changing default behavior would silently break commands
    relying on pipes/redirects that already exist in the wild; it's
    an available safer path for new/gated callers to prefer.
  - The child process now gets CPU/memory/process-count limits — see
    backend/tools/_sandbox.py for why that's split POSIX/Windows and
    why it fails open with "sandboxed": false rather than silently.
    This is a ceiling on an approved-but-runaway command, not a
    replacement for the approval gate itself.
"""
from __future__ import annotations
import asyncio
import os
import shlex
from pathlib import Path

from backend.config import FS_WORKSPACE_ROOT
from backend.tools import _sandbox


class CwdNotAllowed(Exception):
    pass


def _check_cwd(cwd: str | None) -> str:
    resolved = str(Path(cwd).expanduser().resolve()) if cwd else os.getcwd()
    if FS_WORKSPACE_ROOT:
        root = Path(FS_WORKSPACE_ROOT).expanduser().resolve()
        try:
            Path(resolved).relative_to(root)
        except ValueError:
            raise CwdNotAllowed(
                f"cwd '{resolved}' is outside the configured workspace root '{root}'."
            )
    return resolved


async def shell_exec(
    cmd: str,
    cwd: str | None = None,
    timeout: int = 30,
    use_shell: bool = True,
) -> dict:
    """
    Run `cmd` as a subprocess.

    use_shell=True (default, unchanged behavior): full shell
    interpretation via create_subprocess_shell — pipes, redirects,
    globbing all work, but so does command injection if `cmd` is built
    from untrusted input.
    use_shell=False: `cmd` is split with shlex and executed directly
    via create_subprocess_exec — no shell metacharacter interpretation.
    Prefer this when the command is a single program + plain args.

    Returns:
        {
            "stdout": str,
            "stderr": str,
            "returncode": int,
            "error": str | None,  # set only on Python-level exception
            "sandboxed": bool     # whether resource limits actually applied
        }
    """
    try:
        resolved_cwd = _check_cwd(cwd)
    except CwdNotAllowed as exc:
        return {"stdout": "", "stderr": "", "returncode": -1, "error": str(exc), "sandboxed": False}

    # POSIX: the limiting preexec_fn runs in the forked child before
    # exec, so it's passed to create_subprocess_* directly. Windows has
    # no equivalent spawn-time hook — the job assignment happens right
    # after spawn instead (see _sandbox.apply_windows_job_limits).
    preexec_fn = _sandbox.posix_preexec_fn()
    sandboxed = preexec_fn is not None

    try:
        if use_shell:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
                preexec_fn=preexec_fn,
            )
        else:
            try:
                argv = shlex.split(cmd)
            except ValueError as exc:
                return {"stdout": "", "stderr": "", "returncode": -1, "error": f"Could not parse command: {exc}", "sandboxed": False}
            if not argv:
                return {"stdout": "", "stderr": "", "returncode": -1, "error": "Empty command.", "sandboxed": False}
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
                preexec_fn=preexec_fn,
            )

        if _sandbox.IS_WINDOWS:
            sandboxed = _sandbox.apply_windows_job_limits(proc.pid)

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": f"Command timed out after {timeout}s",
                "sandboxed": sandboxed,
            }

        return {
            "stdout":     stdout.decode(errors="replace"),
            "stderr":     stderr.decode(errors="replace"),
            "returncode": proc.returncode,
            "error":      None,
            "sandboxed":  sandboxed,
        }
    except Exception as exc:
        return {
            "stdout":     "",
            "stderr":     "",
            "returncode": -1,
            "error":      str(exc),
            "sandboxed":  sandboxed,
        }