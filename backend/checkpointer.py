"""
checkpointer.py — the approval gate's LangGraph checkpointer.

Deliberately a SEPARATE SQLite file (checkpoints.db) from forge.db, so
LangGraph's own internal schema never mixes with backend/db.py's
sessions/messages tables.

Each /chat call runs under its own fresh thread_id (NOT the session_id —
see agent.py), so a checkpoint only needs to outlive one turn: long
enough to survive the gap between a C137-level gated tool call pausing
the graph and a later POST /sessions/{id}/approve resuming it, even
across separate HTTP connections.

Pruning (reliability hardening)
──────────────────────────────────────────────────────────────────────
Previously finished threads' checkpoints were simply left in the file
forever — checkpoints.db grew unboundedly for as long as the server
ran, since every turn mints a brand-new thread_id. prune_old_checkpoints()
below finds thread_ids recorded in forge.db's turn_threads table
(backend/db.py) older than a retention window and deletes them via
AsyncSqliteSaver.adelete_thread() when that method is available on the
installed langgraph-checkpoint-sqlite version; if it isn't, pruning
logs a warning and does nothing rather than reaching into that
library's internal table schema directly.
"""
from __future__ import annotations

import time

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.config import CHECKPOINTS_DB_PATH
from backend.db import delete_thread_record, get_stale_threads
from backend.logging_config import get_logger

logger = get_logger("checkpointer")

_saver_cm = None
_saver: AsyncSqliteSaver | None = None

DEFAULT_RETENTION_SECONDS = 7 * 24 * 3600  # 7 days


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


async def prune_old_checkpoints(retention_seconds: float = DEFAULT_RETENTION_SECONDS) -> int:
    """
    Deletes checkpoints.db rows for any thread_id recorded in forge.db's
    turn_threads table more than `retention_seconds` ago. Returns the
    number of threads pruned.

    Safe to call repeatedly (e.g. from a periodic background task) —
    each call only ever looks at threads currently older than the
    cutoff, and a thread whose adelete_thread() call fails is left in
    turn_threads to retry next time rather than being dropped from the
    registry.
    """
    saver = await get_checkpointer()
    if not hasattr(saver, "adelete_thread"):
        logger.warning(
            "installed langgraph-checkpoint-sqlite has no adelete_thread() — "
            "checkpoint pruning skipped this cycle; checkpoints.db will keep growing."
        )
        return 0

    cutoff = time.time() - retention_seconds
    stale = await get_stale_threads(cutoff)
    pruned = 0
    for row in stale:
        thread_id = row["thread_id"]
        try:
            await saver.adelete_thread(thread_id)
            await delete_thread_record(thread_id)
            pruned += 1
        except Exception:
            logger.exception("failed to prune checkpoint thread %s", thread_id)
    if pruned:
        logger.info("pruned %d checkpoint thread(s) older than %.0fs", pruned, retention_seconds)
    return pruned
