"""SQLite session + message persistence (aiosqlite)."""
import json
import time
import uuid
import aiosqlite
from backend.config import DB_PATH


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
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
        """)
        await db.commit()


async def create_session(title: str = "New Chat") -> dict:
    sid = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (id,title,created,updated) VALUES (?,?,?,?)",
            (sid, title, now, now),
        )
        await db.commit()
    return {"id": sid, "title": title, "created": now, "updated": now}


async def list_sessions() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM sessions ORDER BY updated DESC"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_session(sid: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions WHERE id=?", (sid,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def update_session_title(sid: str, title: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET title=?, updated=? WHERE id=?",
            (title, time.time(), sid),
        )
        await db.commit()


async def delete_session(sid: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM sessions WHERE id=?", (sid,))
        await db.commit()


async def append_message(session_id: str, role: str, content: str) -> str:
    mid = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
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
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO blackboard_entries (id,session_id,author,tag,text,ts) VALUES (?,?,?,?,?,?)",
            (eid, session_id, author, tag, text, time.time()),
        )
        await db.commit()
    return eid


async def get_blackboard(session_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM blackboard_entries WHERE session_id=? ORDER BY ts",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]
