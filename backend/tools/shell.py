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
"""
from __future__ import annotations
import asyncio
import os
import shlex
from pathlib import Path

from backend.config import FS_WORKSPACE_ROOT


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
            "error": str | None   # set only on Python-level exception
        }
    """
    try:
        resolved_cwd = _check_cwd(cwd)
    except CwdNotAllowed as exc:
        return {"stdout": "", "stderr": "", "returncode": -1, "error": str(exc)}

    try:
        if use_shell:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
            )
        else:
            try:
                argv = shlex.split(cmd)
            except ValueError as exc:
                return {"stdout": "", "stderr": "", "returncode": -1, "error": f"Could not parse command: {exc}"}
            if not argv:
                return {"stdout": "", "stderr": "", "returncode": -1, "error": "Empty command."}
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
            )
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
            }

        return {
            "stdout":     stdout.decode(errors="replace"),
            "stderr":     stderr.decode(errors="replace"),
            "returncode": proc.returncode,
            "error":      None,
        }
    except Exception as exc:
        return {
            "stdout":     "",
            "stderr":     "",
            "returncode": -1,
            "error":      str(exc),
        }
