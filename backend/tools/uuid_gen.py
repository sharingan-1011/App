"""
tools/uuid_gen.py — uuid_gen: minimal reference skill.

Generic harmless skill proving the Stage A -> Stage B wiring end-to-end
(StructuredTool wrapper, TOOLS list, model.bind_tools, SSE tool_result).
No I/O, no timeout needed, no failure mode beyond "can't happen".
"""
from __future__ import annotations
import uuid


async def uuid_gen() -> dict:
    """Return one freshly generated UUID4."""
    return {"ok": True, "uuid": str(uuid.uuid4())}
