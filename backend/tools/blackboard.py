"""
tools/blackboard.py — blackboard_post: the shared, session-scoped
findings log every active agent can write to.

Deliberately NOT one of the hard-restricted, per-specialist tool
subsets (see tools/subagent.py's SPECIALIST_TOOL_NAMES) — it's the one
tool every agent in a deployed session gets, specifically so the
tiering walls don't block the one channel meant to cross them.

`session_id` and `author` are bound via closure, not exposed as
model-settable arguments: the model should never be able to post as a
different agent's identity.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.tools import StructuredTool

from backend.db import add_blackboard_entry

VALID_TAGS = ("finding", "decision", "blocker", "context")


class BlackboardPostArgs(BaseModel):
    text: str = Field(..., description="The note to post — keep it short, one idea per post.")
    tag: str = Field(
        "finding",
        description="One of: finding, decision, blocker, context. "
                    "Use 'blocker' when stuck (e.g. waiting on approval) — "
                    "it's the one tag the user can act on by reassigning your task.",
    )


def make_blackboard_tool(session_id: str, author: str) -> StructuredTool:
    async def _post(text: str, tag: str = "finding") -> dict:
        if tag not in VALID_TAGS:
            tag = "finding"
        entry_id = await add_blackboard_entry(session_id, author, tag, text)
        await adispatch_custom_event(
            "blackboard_entry",
            {"id": entry_id, "author": author, "tag": tag, "text": text},
        )
        return {"ok": True, "id": entry_id, "author": author, "tag": tag}

    return StructuredTool.from_function(
        coroutine=_post,
        name="blackboard_post",
        description=(
            "Post a short, tagged note to this session's shared blackboard, visible "
            "to every other active agent and to the user without anyone needing to "
            "open your own tool-call history. tag must be 'finding', 'decision', "
            "'blocker', or 'context'."
        ),
        args_schema=BlackboardPostArgs,
    )
