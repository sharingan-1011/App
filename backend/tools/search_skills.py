"""
tools/search_skills.py — the Tier-3 skill registry and its one
discovery tool.

Tier-1 tools (fs_*, shell_exec, repl_run, web_*, blackboard_post,
memory_*, schedule_task, search_skills itself) are always bound.
Tier-3 skills in SKILL_REGISTRY are NOT — they're invisible to the
model until search_skills(query) matches one, at which point agent.py
adds that skill's name to the graph state's unlocked_skills list, and
the NEXT agent-node turn binds it alongside the Tier-1 set. This is
the one tool whose execution has a side effect beyond its own return
value: see agent.py's tools_node, which special-cases this tool's
name to read which skills it matched.

Today SKILL_REGISTRY holds exactly one entry (uuid_gen) — moved here
from the Stage A/B always-bound TOOLS list specifically so this
mechanism has something real to discover, rather than search_skills
existing with nothing in it to find.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from backend.tools.uuid_gen import uuid_gen


class UuidGenArgs(BaseModel):
    """uuid_gen takes no arguments."""


SKILL_REGISTRY: dict[str, StructuredTool] = {
    "uuid_gen": StructuredTool.from_function(
        coroutine=uuid_gen,
        name="uuid_gen",
        description="Generate a single random UUID4. Use when a unique identifier is needed and no other tool supplies one.",
        args_schema=UuidGenArgs,
    ),
}


class SearchSkillsArgs(BaseModel):
    query: str = Field(..., description="Keywords describing what you need. Matched against each skill's name and description.")


def _matches(query: str) -> list[dict]:
    q = query.lower()
    out = []
    for name, tool in SKILL_REGISTRY.items():
        haystack = f"{name} {tool.description}".lower()
        if any(word in haystack for word in q.split()):
            out.append({"name": name, "description": tool.description})
    return out


async def search_skills(query: str) -> dict:
    matches = _matches(query)
    return {
        "ok": True,
        "matches": matches,
        "note": (
            "Matched skills are now available to call directly by name on your next turn."
            if matches else
            "No skill matched that query. Only tools already in your Tier-1 roster are available."
        ),
    }


def build_search_skills_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=search_skills,
        name="search_skills",
        description=(
            "Search for an on-demand skill not already in your tool list, by keyword. "
            "A matched skill becomes callable by name on your NEXT turn — it is not "
            "available in the same turn as this search."
        ),
        args_schema=SearchSkillsArgs,
    )
