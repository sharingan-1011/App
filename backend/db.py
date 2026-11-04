"""SQLite session + message persistence (aiosqlite)."""
import json
import time
import uuid
import aiosqlite
from backend.config import DB_PATH


def _connect(path: str = DB_PATH):
    """
    aiosqlite.connect() wrapper — callers should `async with` this just
    like aiosqlite.connect() directly. Reliability hardening: every
    connection now gets WAL journal mode (readers don't block writers,
    and vice versa) and a busy_timeout, so a concurrent write from the
    scheduler loop firing while a live /chat turn is also writing
    doesn't immediately surface as "database is locked" — it waits up
    to the timeout instead. Previously every call opened a connection
    with SQLite's defaults (rollback-journal mode, 0 busy_timeout).
    """
    return _ConnectWithPragmas(path)


class _ConnectWithPragmas:
    def __init__(self, path: str):
        self._path = path
        self._conn = None

    async def __aenter__(self) -> aiosqlite.Connection:
        self._conn = await aiosqlite.connect(self._path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        await self._conn.close()


async def init_db():
    async with _connect() as db:
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id        TEXT PRIMARY KEY,
            title     TEXT NOT NULL DEFAULT 'New Chat',
            created   REAL NOT NULL,
            updated   REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            ts         REAL NOT NULL
        );

        -- Subagent deploy config, set once on a session's first message
        -- and never changed afterward. One row per session; absence of a
        -- row means "solo mode" (the default before the picker is used).
        CREATE TABLE IF NOT EXISTS session_config (
            session_id          TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
            subagents_enabled   INTEGER NOT NULL DEFAULT 0,
            active_specialists  TEXT NOT NULL DEFAULT '[]'
        );

        -- The shared blackboard: a session-scoped, append-only findings
        -- log every active agent (C137 + specialists) can post to.
        -- Deliberately separate from `messages` — this is a structured
        -- findings feed, not chat history replayed to the model.
        CREATE TABLE IF NOT EXISTS blackboard_entries (
            id         TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            author     TEXT NOT NULL,
            tag        TEXT NOT NULL,
            text       TEXT NOT NULL,
            ts         REAL NOT NULL
        );

        -- Cross-session persistent memory. Deliberately a flat key/value
        -- store, global (not per-session) — this is C137's own long-term
        -- memory, not chat history. No concrete spec for this was ever
        -- agreed beyond "a memory_read/memory_write tool should exist";
        -- this is a minimal, honest implementation of that, not a richer
        -- design someone signed off on.
        CREATE TABLE IF NOT EXISTS memory_entries (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL,
            updated REAL NOT NULL
        );

        -- Scheduled/background jobs: a stored prompt + a due time. Picked
        -- up by the polling loop in backend/scheduler.py, not by this
        -- file (db.py only owns the storage).
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id         TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            prompt     TEXT NOT NULL,
            run_at     REAL NOT NULL,
            status     TEXT NOT NULL DEFAULT 'pending',
            created    REAL NOT NULL
        );

        -- Registry of LangGraph thread_ids, one per turn (see agent.py —
        -- every stream_agent() call mints a fresh thread_id). Lets
        -- backend/checkpointer.py's pruning job find and delete
        -- checkpoints.db rows for turns old enough to be done with,
        -- without needing to know that saver's internal schema beyond
        -- "it's keyed by thread_id". Deliberately NOT in checkpoints.db
        -- itself — same separation-of-concerns reasoning as the rest of
        -- this app's two-database split.
        CREATE TABLE IF NOT EXISTS turn_threads (
            thread_id  TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            created    REAL NOT NULL
        );

        -- Durable index for the C137-level approval gate: call_id ->
        -- which checkpointed thread to resume. The actual paused graph
        -- state already lives durably in checkpoints.db (AsyncSqliteSaver) —
        -- this table is only the lookup /approve needs to find it, so a
        -- server restart doesn't orphan an approval that's otherwise
        -- perfectly resumable. Deliberately NOT used for
        -- PENDING_SPECIALIST_APPROVALS or cancel_event (backend/agent.py,
        -- backend/tools/subagent.py) — those are tied to a live coroutine
        -- or an open connection in the now-dead process, so there is
        -- nothing a restart could actually recover for them.
        CREATE TABLE IF NOT EXISTS pending_graph_approvals (
            call_id    TEXT PRIMARY KEY,
            thread_id  TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created    REAL NOT NULL
        );
        """)
        await db.commit()


# ── C137-level approval gate index (restart-durable) ────────────────────

async def add_pending_graph_approval(call_id: str, thread_id: str, session_id: str) -> None:
    async with _connect() as db:
        await db.execute(
            "INSERT OR REPLACE INTO pending_graph_approvals (call_id, thread_id, session_id, created) VALUES (?,?,?,?)",
            (call_id, thread_id, session_id, time.time()),
        )
        await db.commit()


async def get_pending_graph_approval(call_id: str) -> dict | None:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM pending_graph_approvals WHERE call_id=?", (call_id,)
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def pop_pending_graph_approval(call_id: str) -> dict | None:
    """Atomically reads then removes one entry — used by POST /approve so
    a call_id can only ever be resolved once."""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM pending_graph_approvals WHERE call_id=?", (call_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            await db.execute("DELETE FROM pending_graph_approvals WHERE call_id=?", (call_id,))
            await db.commit()
    return dict(row) if row else None


async def delete_pending_graph_approval(call_id: str) -> None:
    async with _connect() as db:
        await db.execute("DELETE FROM pending_graph_approvals WHERE call_id=?", (call_id,))
        await db.commit()


async def list_pending_graph_approvals() -> list[dict]:
    """Used once at startup to re-arm approvals left pending by a
    previous process (backend/agent.py's rearm_pending_approvals)."""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM pending_graph_approvals") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def create_session(title: str = "New Chat") -> dict:
    sid = str(uuid.uuid4())
    now = time.time()
    async with _connect() as db:
        await db.execute(
            "INSERT INTO sessions (id,title,created,updated) VALUES (?,?,?,?)",
            (sid, title, now, now),
        )
        await db.commit()
    return {"id": sid, "title": title, "created": now, "updated": now}


async def list_sessions() -> list[dict]:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sessions ORDER BY updated DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_session(sid: str) -> dict | None:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions WHERE id=?", (sid,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def update_session_title(sid: str, title: str):
    async with _connect() as db:
        await db.execute(
            "UPDATE sessions SET title=?, updated=? WHERE id=?",
            (title, time.time(), sid),
        )
        await db.commit()


async def delete_session(sid: str):
    async with _connect() as db:
        await db.execute("DELETE FROM sessions WHERE id=?", (sid,))
        await db.commit()


async def append_message(session_id: str, role: str, content: str) -> str:
    mid = str(uuid.uuid4())
    now = time.time()
    async with _connect() as db:
        await db.execute(
            "INSERT INTO messages (id,session_id,role,content,ts) VALUES (?,?,?,?,?)",
            (mid, session_id, role, content, now),
        )
        await db.execute(
            "UPDATE sessions SET updated=? WHERE id=?", (now, session_id)
        )
        await db.commit()
    return mid


async def get_history(session_id: str) -> list[dict]:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY ts",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_history_for_llm(session_id: str) -> list[dict]:
    """Return history in OpenAI message format."""
    rows = await get_history(session_id)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ── session_config: subagent deploy picker, locked for the session ─────────

async def set_session_config(session_id: str, subagents_enabled: bool, active_specialists: list[str]) -> None:
    async with _connect() as db:
        await db.execute(
            """
            INSERT INTO session_config (session_id, subagents_enabled, active_specialists)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                subagents_enabled = excluded.subagents_enabled,
                active_specialists = excluded.active_specialists
            """,
            (session_id, 1 if subagents_enabled else 0, json.dumps(active_specialists)),
        )
        await db.commit()


async def get_session_config(session_id: str) -> dict:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM session_config WHERE session_id=?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {"subagents_enabled": False, "active_specialists": []}
    return {
        "subagents_enabled": bool(row["subagents_enabled"]),
        "active_specialists": json.loads(row["active_specialists"]),
    }


# ── shared blackboard ────────────────────────────────────────────────────────

async def add_blackboard_entry(session_id: str, author: str, tag: str, text: str) -> str:
    eid = str(uuid.uuid4())
    async with _connect() as db:
        await db.execute(
            "INSERT INTO blackboard_entries (id,session_id,author,tag,text,ts) VALUES (?,?,?,?,?,?)",
            (eid, session_id, author, tag, text, time.time()),
        )
        await db.commit()
    return eid


async def get_blackboard(session_id: str) -> list[dict]:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM blackboard_entries WHERE session_id=? ORDER BY ts",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ── cross-session memory ────────────────────────────────────────────────────

async def memory_set(key: str, value: str) -> None:
    async with _connect() as db:
        await db.execute(
            """
            INSERT INTO memory_entries (key, value, updated) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated = excluded.updated
            """,
            (key, value, time.time()),
        )
        await db.commit()


async def memory_get(key: str) -> str | None:
    async with _connect() as db:
        async with db.execute("SELECT value FROM memory_entries WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
    return row[0] if row else None


async def memory_list(prefix: str = "") -> list[dict]:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT key, value, updated FROM memory_entries WHERE key LIKE ? ORDER BY key",
            (f"{prefix}%",),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ── scheduled tasks ──────────────────────────────────────────────────────────

async def add_scheduled_task(session_id: str, prompt: str, run_at: float) -> str:
    tid = str(uuid.uuid4())
    async with _connect() as db:
        await db.execute(
            "INSERT INTO scheduled_tasks (id,session_id,prompt,run_at,status,created) VALUES (?,?,?,?,?,?)",
            (tid, session_id, prompt, run_at, "pending", time.time()),
        )
        await db.commit()
    return tid


async def get_due_tasks(now: float) -> list[dict]:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM scheduled_tasks WHERE status='pending' AND run_at<=?", (now,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def mark_task_status(task_id: str, status: str) -> None:
    async with _connect() as db:
        await db.execute("UPDATE scheduled_tasks SET status=? WHERE id=?", (status, task_id))
        await db.commit()


async def list_scheduled_tasks(session_id: str) -> list[dict]:
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM scheduled_tasks WHERE session_id=? ORDER BY run_at", (session_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ── turn/thread registry (checkpoint pruning support) ───────────────────────

async def record_thread(thread_id: str, session_id: str) -> None:
    async with _connect() as db:
        await db.execute(
            "INSERT OR IGNORE INTO turn_threads (thread_id, session_id, created) VALUES (?,?,?)",
            (thread_id, session_id, time.time()),
        )
        await db.commit()


async def get_stale_threads(older_than: float) -> list[dict]:
    """Threads recorded before `older_than` (a Unix timestamp)."""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM turn_threads WHERE created < ?", (older_than,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def delete_thread_record(thread_id: str) -> None:
    async with _connect() as db:
        await db.execute("DELETE FROM turn_threads WHERE thread_id=?", (thread_id,))
        await db.commit()