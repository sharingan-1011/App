"""
checkpointer.py — the approval gate's LangGraph checkpointer.

Deliberately a SEPARATE SQLite file (checkpoints.db) from forge.db, so
LangGraph's own internal schema never mixes with backend/db.py's
sessions/messages tables.

Each /chat call runs under its own fresh thread_id (NOT the session_id —
see agent.py), so a checkpoint only needs to outlive one turn: long
enough to survive the gap between a C137-level gated tool call pausing
the graph and a later POST /sessions/{id}/approve resuming it, even
across separate HTTP connections. Finished threads' checkpoints are
simply left in the file; pruning them is a separate concern, out of
scope here.
"""
from __future__ import annotations

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.config import CHECKPOINTS_DB_PATH

_saver_cm = None
_saver: AsyncSqliteSaver | None = None


async def get_checkpointer() -> AsyncSqliteSaver:
    """
    Lazily opens one long-lived AsyncSqliteSaver for the process. Not
    thread-safe to call concurrently for the very first call (fine here:
    FastAPI's lifespan can warm it once at startup — see main.py), but
    safe to call repeatedly afterward since it just returns the same
    cached instance.
    """
    global _saver_cm, _saver
    if _saver is None:
        _saver_cm = AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB_PATH)
        _saver = await _saver_cm.__aenter__()
    return _saver


async def close_checkpointer() -> None:
    global _saver_cm, _saver
    if _saver_cm is not None:
        await _saver_cm.__aexit__(None, None, None)
        _saver_cm = None
        _saver = None
