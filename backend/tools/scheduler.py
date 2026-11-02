"""
tools/scheduler.py — schedule_task: register a background/deferred job,
plus the polling loop that actually runs them.

Minimal, honest scope: a scheduled task is one stored prompt and a due
time. The polling loop (run_scheduler_loop, started from main.py's
FastAPI lifespan) checks backend/db.py's scheduled_tasks table every
POLL_INTERVAL_SECONDS and, for anything due, runs a normal agent turn
with that prompt as a synthetic user message in that task's session,
persists the result the same way a live /chat turn would, and marks
the task done. No cron syntax, no recurrence, no timezone handling
beyond "run_at is a Unix timestamp the caller already resolved" — a
single "run once at this time" primitive, which is what "scheduled/
background jobs" was agreed to mean here; building a full recurring
scheduler was never specified beyond that goal.
"""
from __future__ import annotations

import asyncio
import json
import time

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from backend.db import add_scheduled_task, append_message, get_due_tasks, get_history_for_llm, mark_task_status
from backend.logging_config import get_logger

logger = get_logger("scheduler")

POLL_INTERVAL_SECONDS = 15


class ScheduleTaskArgs(BaseModel):
    prompt: str = Field(..., description="The instruction to run as a fresh turn in this session when due.")
    delay_seconds: int = Field(..., description="How many seconds from now to run this task.", ge=1)


def make_schedule_task_tool(session_id: str) -> StructuredTool:
    async def _schedule(prompt: str, delay_seconds: int) -> dict:
        run_at = time.time() + delay_seconds
        task_id = await add_scheduled_task(session_id, prompt, run_at)
        return {"ok": True, "task_id": task_id, "run_at": run_at}

    return StructuredTool.from_function(
        coroutine=_schedule,
        name="schedule_task",
        description=(
            "Register a one-off background task: a prompt that will run as a fresh turn "
            "in this session after delay_seconds. Not a recurring/cron schedule."
        ),
        args_schema=ScheduleTaskArgs,
    )


async def _run_due_task(task: dict, stream_agent) -> None:
    session_id = task["session_id"]
    prompt = task["prompt"]
    task_id = task["id"]

    try:
        await append_message(session_id, "user", f"[scheduled task fired] {prompt}")
        history = await get_history_for_llm(session_id)

        parts: list[str] = []
        async for chunk in stream_agent(session_id, history):
            try:
                payload = json.loads(chunk.removeprefix("data: ").strip())
                if payload.get("type") == "delta":
                    parts.append(payload.get("text", ""))
            except Exception:
                logger.debug("scheduled task %s: could not parse SSE chunk %r", task_id, chunk)

        full_response = "".join(parts)
        if full_response.strip():
            await append_message(session_id, "assistant", full_response)

        await mark_task_status(task_id, "done")
        logger.info("scheduled task %s (session %s) completed", task_id, session_id)
    except Exception:
        logger.exception("scheduled task %s (session %s) failed", task_id, session_id)
        try:
            await mark_task_status(task_id, "failed")
        except Exception:
            logger.exception("scheduled task %s: also failed to mark status as 'failed'", task_id)


async def run_scheduler_loop(stream_agent) -> None:
    """
    Runs forever (until cancelled) as a background asyncio task started
    from main.py's lifespan. `stream_agent` is passed in rather than
    imported at module load time to avoid a circular import between
    agent.py and this module.

    Previously the per-tick try/except swallowed every exception with
    a bare `pass` — a malfunctioning scheduled task failed completely
    silently, with nothing for an operator to look at. Now logged (with
    traceback) before the loop continues; a bad tick still can't kill
    the loop, but it's no longer invisible.
    """
    while True:
        try:
            due = await get_due_tasks(time.time())
            for task in due:
                await mark_task_status(task["id"], "running")
                asyncio.create_task(_run_due_task(task, stream_agent))
        except Exception:
            logger.exception("scheduler loop tick failed")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
