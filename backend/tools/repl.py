"""
tools/repl.py — repl_run: persistent Python REPL per chat session.

Uses a dict of `code.InteractiveInterpreter` instances keyed by session_id.
Variables persist across calls within the same session.
Stdout/stderr are captured via io.StringIO redirection.

Bounded cache (P0 hardening)
──────────────────────────────────────────────────────────────────────
Previously _kernels grew forever — one entry per session_id for the
life of the process, removed only by an explicit clear_repl() call on
session delete. A long-running server that ever created many sessions
(including ones the user never explicitly deleted) leaked memory
without bound. Now it's an OrderedDict used as an LRU cache, capped at
REPL_MAX_SESSIONS entries, with entries also expiring after
REPL_IDLE_TTL_SECONDS of inactivity.

The TTL is checked lazily — on each repl_run call, before touching the
dict — rather than via a separate background sweep task. No new
asyncio task/thread to manage, no risk of it firing mid-eviction while
a call is using the kernel; the tradeoff is that a session idle past
its TTL isn't actually freed until *some* repl_run call happens
anywhere (any session), not exactly at the TTL boundary. Acceptable
here: the goal is bounding worst-case memory, not precise timing.
"""
from __future__ import annotations
import asyncio
import code
import io
import sys
import time
import traceback
from collections import OrderedDict
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from backend.config import REPL_IDLE_TTL_SECONDS, REPL_MAX_SESSIONS
from backend.logging_config import get_logger

logger = get_logger("repl")

# In-process kernels: session_id → {"interp", "locals", "last_used"}.
# OrderedDict so move_to_end() on access gives true LRU ordering.
_kernels: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def _evict_expired_and_over_capacity() -> None:
    """
    Lazy maintenance pass, run at the top of every repl_run call.
    First drops anything idle past REPL_IDLE_TTL_SECONDS, then — if
    still over REPL_MAX_SESSIONS — evicts the least-recently-used
    entries (the front of the OrderedDict) until back at the cap.
    """
    now = time.monotonic()
    expired = [sid for sid, k in _kernels.items() if now - k["last_used"] > REPL_IDLE_TTL_SECONDS]
    for sid in expired:
        _kernels.pop(sid, None)
        logger.info("repl session %s evicted (idle past %ds TTL)", sid, REPL_IDLE_TTL_SECONDS)

    while len(_kernels) > REPL_MAX_SESSIONS:
        sid, _ = _kernels.popitem(last=False)  # least-recently-used
        logger.info("repl session %s evicted (LRU cap of %d sessions)", sid, REPL_MAX_SESSIONS)


def _get_kernel(session_id: str) -> tuple[code.InteractiveInterpreter, dict]:
    _evict_expired_and_over_capacity()

    if session_id not in _kernels:
        local_ns: dict = {}
        interp = code.InteractiveInterpreter(locals=local_ns)
        _kernels[session_id] = {"interp": interp, "locals": local_ns, "last_used": time.monotonic()}
    else:
        _kernels[session_id]["last_used"] = time.monotonic()
        _kernels.move_to_end(session_id)  # mark as most-recently-used

    k = _kernels[session_id]
    return k["interp"], k["locals"]


def _run_sync(session_id: str, src: str) -> dict:
    interp, local_ns = _get_kernel(session_id)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        compiled = compile(src, "<repl>", "exec")
    except SyntaxError:
        tb = traceback.format_exc()
        return {"ok": False, "stdout": "", "stderr": tb, "error": "SyntaxError"}

    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        try:
            exec(compiled, local_ns)  # noqa: S102
            error = None
            ok = True
        except Exception:
            error = traceback.format_exc()
            ok = False

    stdout = stdout_buf.getvalue()
    stderr = stderr_buf.getvalue()
    if error:
        stderr = (stderr + "\n" + error).strip()

    return {"ok": ok, "stdout": stdout, "stderr": stderr, "error": error}


async def repl_run(
    code: str,
    session_id: str = "default",
    timeout: int = 60,
) -> dict:
    """
    Execute Python code in a persistent in-process REPL.

    Variables survive between calls within the same session_id.

    Returns:
        {
            "ok": bool,
            "stdout": str,
            "stderr": str,
            "error": str | None
        }
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, _run_sync, session_id, code),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        return {"ok": False, "stdout": "", "stderr": "", "error": f"Execution timed out after {timeout}s"}
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": "", "error": str(exc)}


def clear_repl(session_id: str) -> None:
    """Clear the REPL state for a session (call on session delete)."""
    _kernels.pop(session_id, None)