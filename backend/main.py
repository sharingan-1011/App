"""
main.py — FastAPI application entry point. v2.2.0

Approval gate + subagent deployment additions
──────────────────────────────────────────────
- POST /sessions now optionally accepts {subagents_enabled,
  active_specialists}, persisted once and locked for that session's
  whole life (backend/db.py's session_config table).
- POST /sessions/{id}/approve resolves a pending gated-tool call,
  either a C137-level one (backend/agent.py's PENDING_GRAPH_APPROVALS —
  resumes the LangGraph checkpoint, returns a FRESH StreamingResponse
  continuing the same SSE contract) or a specialist-originated one
  (backend/tools/subagent.py's PENDING_SPECIALIST_APPROVALS — just
  resolves an in-process asyncio.Event; the original /chat connection
  is still open and will pick the result up on its own, so this
  returns a plain JSON ack instead of a new stream).
- GET /sessions/{id}/config and GET /sessions/{id}/blackboard let the
  frontend load the deploy-picker state and the blackboard's history
  on page load, since SSE only pushes NEW events during an active turn.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.agent import PENDING_GRAPH_APPROVALS, resume_agent, stream_agent
from backend.checkpointer import close_checkpointer, get_checkpointer
from backend.db import (
    append_message,
    create_session,
    delete_session,
    get_blackboard,
    get_history_for_llm,
    get_session,
    get_session_config,
    init_db,
    list_scheduled_tasks,
    list_sessions,
    set_session_config,
    update_session_title,
)
from backend.tools.repl import clear_repl
from backend.tools.subagent import PENDING_SPECIALIST_APPROVALS
from backend.tools.scheduler import run_scheduler_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await get_checkpointer()  # warm the approval-gate checkpointer once at startup
    scheduler_task = asyncio.create_task(run_scheduler_loop(stream_agent))
    yield
    scheduler_task.cancel()
    await close_checkpointer()


app = FastAPI(title="FORGE", version="2.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── sessions ──────────────────────────────────────────────────

@app.get("/sessions")
async def get_sessions():
    return await list_sessions()


class NewSessionRequest(BaseModel):
    subagents_enabled: bool = False
    active_specialists: list[str] = []


@app.post("/sessions")
async def new_session(req: NewSessionRequest | None = Body(None)):
    session = await create_session()
    if req is not None:
        valid = {"Ba-56", "La-57", "Ce-58"}
        active = [s for s in req.active_specialists if s in valid]
        await set_session_config(session["id"], req.subagents_enabled and bool(active), active)
    return {"session_id": session["id"]}


@app.get("/sessions/{session_id}")
async def get_session_endpoint(session_id: str):
    session = await get_session(session_id)
    if not session:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return session


@app.get("/sessions/{session_id}/config")
async def get_session_config_endpoint(session_id: str):
    return await get_session_config(session_id)


@app.get("/sessions/{session_id}/blackboard")
async def get_session_blackboard(session_id: str):
    return await get_blackboard(session_id)


@app.get("/sessions/{session_id}/scheduled_tasks")
async def get_scheduled_tasks(session_id: str):
    return await list_scheduled_tasks(session_id)


class RenameRequest(BaseModel):
    title: str


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameRequest):
    await update_session_title(session_id, req.title)
    return {"ok": True}


@app.delete("/sessions/{session_id}")
async def del_session(session_id: str):
    await delete_session(session_id)
    clear_repl(session_id)
    return {"ok": True}


# ── chat ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str


def _stream_and_persist(req: Request, session_id: str, user_msg: str | None, agen):
    """
    Shared tail for /chat and the graph-interrupt branch of /approve:
    watches for client disconnect, forwards every SSE chunk, and on
    completion persists the accumulated "delta" text as the assistant's
    message — same accumulation logic main.py has always used, just
    factored out so both call sites share it.
    """
    cancel_event = asyncio.Event()
    response_parts: list[str] = []

    async def generate():
        async def _watch_disconnect():
            while not cancel_event.is_set():
                if await req.is_disconnected():
                    cancel_event.set()
                    return
                await asyncio.sleep(0.3)

        watcher = asyncio.create_task(_watch_disconnect())

        try:
            async for chunk in agen(cancel_event):
                yield chunk
                try:
                    payload = json.loads(chunk.removeprefix("data: ").strip())
                    if payload.get("type") == "delta":
                        response_parts.append(payload.get("text", ""))
                except Exception:
                    pass
        finally:
            cancel_event.set()
            watcher.cancel()

            full_response = "".join(response_parts)
            if full_response.strip():
                await append_message(session_id, "assistant", full_response)
                if user_msg is not None:
                    session = await get_session(session_id)
                    if session and session.get("title", "").startswith("New Chat"):
                        title = user_msg[:48] + ("…" if len(user_msg) > 48 else "")
                        await update_session_title(session_id, title)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/chat")
async def chat(req: Request, body: ChatRequest = Body(...)):
    session_id = body.session_id
    user_msg   = body.message

    await append_message(session_id, "user", user_msg)
    history = await get_history_for_llm(session_id)

    def agen(cancel_event):
        return stream_agent(session_id, history, cancel_event=cancel_event)

    return _stream_and_persist(req, session_id, user_msg, agen)


# ── approval gate ─────────────────────────────────────────────

class ApproveRequest(BaseModel):
    call_id: str
    decision: str  # "approve" | "deny" | "reassign"
    reassign_to: str | None = None


@app.post("/sessions/{session_id}/approve")
async def approve(req: Request, session_id: str, body: ApproveRequest = Body(...)):
    decision = {"decision": body.decision}
    if body.reassign_to:
        decision["reassign_to"] = body.reassign_to

    # 1) A specialist's gated call — just resolve its asyncio.Event.
    #    The original /chat connection is still open and streaming.
    pending = PENDING_SPECIALIST_APPROVALS.get(body.call_id)
    if pending is not None:
        pending["decision"] = decision
        pending["event"].set()
        return {"ok": True, "mechanism": "event"}

    # 2) A C137-level gated call — resume the checkpointed graph. That
    #    original connection already closed when the interrupt paused
    #    it, so this returns a FRESH SSE stream continuing the turn.
    graph_pending = PENDING_GRAPH_APPROVALS.pop(body.call_id, None)
    if graph_pending is not None:
        if body.decision == "reassign":
            # Reassignment only makes sense for a specialist's task;
            # C137's own direct call has nowhere else to go. Fall back
            # to deny rather than silently approving something the
            # caller meant to redirect.
            decision = {"decision": "deny", "reason": "reassign is not meaningful for a top-level call"}

        thread_id = graph_pending["thread_id"]

        def agen(cancel_event):
            return resume_agent(session_id, thread_id, decision, cancel_event=cancel_event)

        return _stream_and_persist(req, session_id, None, agen)

    return JSONResponse({"error": f"No pending approval for call_id {body.call_id}"}, status_code=404)


# ── upload ────────────────────────────────────────────────────

UPLOAD_DIR = Path(tempfile.gettempdir()) / "forge_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile):
    safe_name = Path(file.filename).name
    dest      = UPLOAD_DIR / safe_name
    counter   = 1
    stem, suffix = dest.stem, dest.suffix
    while dest.exists():
        dest = UPLOAD_DIR / f"{stem}_{counter}{suffix}"
        counter += 1
    contents = await file.read()
    dest.write_bytes(contents)
    return {"path": str(dest), "name": dest.name, "size": len(contents)}
