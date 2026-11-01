"""
tools/computer.py — computer_action: desktop/OS control.

Honest stub, not a real implementation. "Full-desktop OS control" was
named as a compulsory FORGE goal, but it targets the USER's own
machine, not this backend's host — that requires a remote-control
integration (something like a VNC/RDP client, or an agent process
running on the user's own desktop that this backend talks to over a
side channel) whose design was never specified in any of the design
discussion. Building a fake version that silently does nothing while
reporting success would be worse than not having it: this tool
returns a clear, explicit "not implemented" error every time, with
the gating/registry plumbing wired exactly as it will need to be
once a real integration exists.

Gated (Tier 2, heavy) — see tools/__init__.py's GATED_TOOLS.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


class ComputerActionArgs(BaseModel):
    action: str = Field(..., description="The desktop action requested (click, type, screenshot, etc).")


async def computer_action(action: str) -> dict:
    return {
        "ok": False,
        "error": (
            "computer_action is not implemented. Desktop/OS control requires a "
            "remote-control integration with the user's own machine that has not "
            "been built or specified yet — this is a placeholder, not a working tool."
        ),
    }


def build_computer_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=computer_action,
        name="computer_action",
        description=(
            "Control the user's desktop (click, type, screenshot). NOT YET IMPLEMENTED — "
            "always returns a not-implemented error. Registered so gating/tiering is wired "
            "ahead of a real integration."
        ),
        args_schema=ComputerActionArgs,
    )
