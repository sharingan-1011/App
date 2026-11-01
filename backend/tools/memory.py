"""
tools/memory.py — memory_read / memory_write: C137's cross-session,
long-term memory.

Deliberately minimal: a flat key/value store, global (not per-session —
this is C137's own memory, not a session-scoped findings log like the
blackboard). No richer spec for this was ever agreed beyond "a
memory_read/memory_write tool should exist" during the tool-tier
design discussion; this is an honest, minimal implementation of
exactly that, not an invented, unscoped design.

Bound in every session regardless of solo/deployed mode (Tier 1,
compulsory). Not exposed to specialists' restricted subsets by
default — see tools/subagent.py — except memory_read, which specialists
also get: reading C137's memory is low-risk; writing to it from an
isolated sub-task could silently surprise the user with state they
didn't see change, so that stays C137-only.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from backend.db import memory_get, memory_list, memory_set


class MemoryReadArgs(BaseModel):
    key: str | None = Field(
        None,
        description="Exact key to read. Omit (or leave blank) to list all stored keys instead of reading one value.",
    )


async def memory_read(key: str | None = None) -> dict:
    if not key:
        entries = await memory_list()
        return {"ok": True, "keys": [e["key"] for e in entries]}
    value = await memory_get(key)
    if value is None:
        return {"ok": False, "error": f"No memory entry for key: {key}"}
    return {"ok": True, "key": key, "value": value}


class MemoryWriteArgs(BaseModel):
    key: str = Field(..., description="Key to store under. Overwrites any existing value for this key.")
    value: str = Field(..., description="Value to store (stored as text).")


async def memory_write(key: str, value: str) -> dict:
    await memory_set(key, value)
    return {"ok": True, "key": key}


def build_memory_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            coroutine=memory_read,
            name="memory_read",
            description="Read a value from C137's persistent, cross-session memory by key, or list all keys if no key is given.",
            args_schema=MemoryReadArgs,
        ),
        StructuredTool.from_function(
            coroutine=memory_write,
            name="memory_write",
            description="Write a key/value pair to C137's persistent, cross-session memory. Overwrites any existing value for that key.",
            args_schema=MemoryWriteArgs,
        ),
    ]
