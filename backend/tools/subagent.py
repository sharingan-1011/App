"""
tools/subagent.py — spawn_subagent: delegates sub-tasks to specialist
subagents (Ba-56 / La-57 / Ce-58), run in parallel.

Deliberately NOT part of the static TOOLS list in tools/__init__.py:
this is the one tool whose schema is session-scoped (its `specialist`
enum is built from that session's active_specialists) rather than
process-global, and it only exists at all in deployed-mode sessions.
agent.py builds it fresh per turn via make_spawn_subagent_tool(...).

Approval-gate note on two different mechanisms, by design, not oversight
──────────────────────────────────────────────────────────────────────
C137's OWN gated tool calls (handled in agent.py's _tools_node) pause
via LangGraph's checkpointed interrupt() — the graph genuinely halts,
and the original /chat connection closes until POST .../approve starts
a fresh one that resumes the graph.

A SPECIALIST's gated call, running here, does NOT use interrupt(). It
was a deliberate simplification from the design discussion: nesting a
second checkpointed interrupt inside a tool call that's already running
synchronously inside the `tools` graph node is exactly the kind of
"gnarly, contract-breaking" complexity that discussion explicitly chose
to avoid. Instead, a specialist's pause is a plain asyncio.Event wait
with a timeout, tracked in PENDING_SPECIALIST_APPROVALS (module-level,
in-memory, lost on process restart — the same honesty tradeoff the
rest of this app already makes with cancel_event). Because nothing at
the LangGraph level pauses, the original /chat connection stays open
and keeps streaming the whole time — siblings keep running, exactly as
the earlier design settled on.

Reassignment note — a second deliberate simplification
──────────────────────────────────────────────────────
True "swap the specialist live, mid-gather" would mean mutating a
set of already-running asyncio.Tasks in place. What's implemented
here instead: a specialist whose gated call is reassigned stops its
own run immediately and returns a `{"reassigned_to": ...}` marker as
its result. Once the whole parallel batch settles, spawn_subagent
checks for any such markers and runs the replacement specialist on
that same task SEQUENTIALLY, as a follow-up — not re-joined into the
original parallel batch. The user's reassignment is honored and the
work gets done without C137 needing a new reasoning turn, but it is
not literally a live mid-gather swap. Flagging this rather than
overclaiming it.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from backend.config import APPROVAL_TIMEOUT_SECONDS
from backend.db import add_blackboard_entry
from backend.tools.blackboard import make_blackboard_tool

SPECIALIST_TOOL_NAMES: dict[str, list[str]] = {
    "Ba-56": ["shell_exec", "fs_read", "fs_write", "fs_list", "fs_delete", "repl_run", "memory_read"],
    "La-57": ["web_search", "web_fetch", "browser_action", "memory_read"],
    "Ce-58": ["repl_run", "fs_read", "memory_read"],
}

SPECIALIST_SYSTEM_PROMPTS: dict[str, str] = {
    "Ba-56": (
        "You are Ba-56, FORGE's coding specialist, working under C137's direction. "
        "You were given one self-contained task — you have no visibility into the "
        "rest of the conversation. Use shell_exec, fs_read, fs_write, fs_list, "
        "fs_delete, and repl_run as needed. Post important findings to the shared "
        "blackboard with blackboard_post. Be concise; give a final plain-text answer "
        "when done."
    ),
    "La-57": (
        "You are La-57, FORGE's research specialist, working under C137's direction. "
        "You were given one self-contained task — you have no visibility into the "
        "rest of the conversation. Use web_search then web_fetch for full content. "
        "Post important findings to the shared blackboard with blackboard_post. Be "
        "concise; give a final plain-text answer when done."
    ),
    "Ce-58": (
        "You are Ce-58, FORGE's data-analysis specialist, working under C137's "
        "direction. You were given one self-contained task — you have no visibility "
        "into the rest of the conversation. Use repl_run for computation and fs_read "
        "for input data. Post important findings to the shared blackboard with "
        "blackboard_post. Be concise; give a final plain-text answer when done."
    ),
}

MAX_SPECIALIST_ITERATIONS = 6

# call_id -> {"event": asyncio.Event, "decision": dict | None}
# Populated when a specialist hits a gated tool; resolved by main.py's
# POST /sessions/{id}/approve when it finds a matching entry here
# (checked BEFORE falling back to the graph-interrupt resume path).
PENDING_SPECIALIST_APPROVALS: dict[str, dict[str, Any]] = {}


def _gated(name: str, gated_tools: frozenset[str]) -> bool:
    return name in gated_tools


async def _post_blackboard(session_id: str, author: str, tag: str, text: str) -> None:
    """
    Posts directly (bypassing the blackboard_post TOOL, since this is
    this module's own bookkeeping, not a model-issued tool call) but
    still dispatches the same "blackboard_entry" custom event the tool
    wrapper does, so the frontend's live blackboard drawer sees it
    immediately rather than only on its next poll of GET
    /sessions/{id}/blackboard.
    """
    entry_id = await add_blackboard_entry(session_id, author, tag, text)
    await adispatch_custom_event("blackboard_entry", {"id": entry_id, "author": author, "tag": tag, "text": text})


async def _await_specialist_approval(call_id: str, specialist: str, name: str, args: dict) -> dict:
    """
    Blocks (with a timeout) until a decision is posted for this call_id
    via PENDING_SPECIALIST_APPROVALS, or until APPROVAL_TIMEOUT_SECONDS
    elapses, in which case it resolves to {"decision": "deny"} itself —
    same auto-deny-on-timeout contract as the top-level gate.
    """
    event = asyncio.Event()
    entry = {"event": event, "decision": None, "specialist": specialist, "name": name, "args": args, "created": time.time()}
    PENDING_SPECIALIST_APPROVALS[call_id] = entry

    try:
        await asyncio.wait_for(event.wait(), timeout=APPROVAL_TIMEOUT_SECONDS)
        decision = entry["decision"] or {"decision": "deny"}
    except asyncio.TimeoutError:
        decision = {"decision": "deny", "reason": "timeout"}
    finally:
        PENDING_SPECIALIST_APPROVALS.pop(call_id, None)

    return decision


async def _run_specialist(
    specialist: str,
    task: str,
    session_id: str,
    batch_id: str,
    tool_registry: dict,
    get_model,
    gated_tools: frozenset[str],
) -> dict:
    """
    Runs one specialist's own bounded ReAct loop against its restricted
    tool subset + its own blackboard_post, sequential tool execution
    only. Returns either a normal result dict, or
    {"reassigned_to": <specialist>, "task": task} if its one gated call
    was reassigned (see module docstring).
    """
    allowed_names = SPECIALIST_TOOL_NAMES[specialist]
    tools = [tool_registry[n] for n in allowed_names if n in tool_registry]
    tools.append(make_blackboard_tool(session_id, specialist))
    registry = {t.name: t for t in tools}

    model = get_model().bind_tools(tools)
    messages: list = [SystemMessage(content=SPECIALIST_SYSTEM_PROMPTS[specialist]), HumanMessage(content=task)]

    await adispatch_custom_event("subagent_start", {"parent_call_id": batch_id, "subagent": specialist, "task": task})

    tool_call_count = 0
    final_text = ""

    for _ in range(MAX_SPECIALIST_ITERATIONS):
        ai_message: AIMessage = await model.ainvoke(
            messages,
            config={"tags": [f"subagent:{specialist}"], "metadata": {"subagent": specialist, "parent_call_id": batch_id}},
        )
        messages.append(ai_message)
        final_text = ai_message.content if isinstance(ai_message.content, str) else final_text

        if not ai_message.tool_calls:
            break

        for tc in ai_message.tool_calls:
            call_id, name, args = tc["id"], tc["name"], tc["args"]
            tool_call_count += 1

            if name not in registry:
                # Defensive, not merely theoretical: this specialist's
                # model call is bound only to its own restricted tool
                # list, so a well-behaved model can't request anything
                # outside it — but FORGE targets a local, quantized
                # open-source model via llama-server, which is less
                # reliable than a hosted frontier model about strictly
                # respecting function-calling constraints. Skip gating
                # entirely for a name this specialist couldn't run even
                # if approved; there's no point asking the user to
                # approve a call that will fail regardless.
                output = json.dumps({"error": f"Unknown tool: {name}"})
                await adispatch_custom_event(
                    "subagent_tool_result",
                    {"parent_call_id": batch_id, "subagent": specialist, "call_id": call_id, "name": name, "output": output},
                )
                messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))
                continue

            if _gated(name, gated_tools):
                await _post_blackboard(
                    session_id, specialist, "blocker",
                    f"Waiting on approval for {name} before this task can continue.",
                )
                await adispatch_custom_event(
                    "approval_required",
                    {"call_id": call_id, "name": name, "args": args, "subagent": specialist, "parent_call_id": batch_id},
                )
                decision = await _await_specialist_approval(call_id, specialist, name, args)

                if decision.get("decision") == "reassign" and decision.get("reassign_to"):
                    return {"reassigned_to": decision["reassign_to"], "task": task}

                if decision.get("decision") != "approve":
                    output = json.dumps({"ok": False, "error": "Denied by user (or timed out waiting for approval)."})
                    await adispatch_custom_event(
                        "subagent_tool_result",
                        {"parent_call_id": batch_id, "subagent": specialist, "call_id": call_id, "name": name, "output": output},
                    )
                    messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))
                    continue
                # approved — fall through to execution below

            await adispatch_custom_event(
                "subagent_tool_start",
                {"parent_call_id": batch_id, "subagent": specialist, "call_id": call_id, "name": name, "args": args},
            )
            tool = registry.get(name)
            if tool is None:
                output = json.dumps({"error": f"Unknown tool: {name}"})
            else:
                try:
                    result = await tool.ainvoke(args)
                    output = json.dumps(result, ensure_ascii=False)
                except Exception as exc:
                    output = json.dumps({"error": str(exc)})

            await adispatch_custom_event(
                "subagent_tool_result",
                {"parent_call_id": batch_id, "subagent": specialist, "call_id": call_id, "name": name, "output": output},
            )
            messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))

    await adispatch_custom_event("subagent_done", {"parent_call_id": batch_id, "subagent": specialist, "final_text": final_text})

    return {"ok": True, "specialist": specialist, "result": final_text, "tool_calls_made": tool_call_count}


def make_spawn_subagent_tool(
    session_id: str,
    active_specialists: list[str],
    tool_registry: dict,
    get_model,
    gated_tools: frozenset[str],
) -> StructuredTool:
    """
    Builds spawn_subagent for ONE session/turn. The `specialist` enum on
    each task is restricted to this session's active_specialists —
    deliberately session-scoped, unlike every other (process-global)
    tool schema in this app.
    """
    SpecialistLiteral = Literal[tuple(active_specialists)]  # type: ignore[valid-type]

    TaskModel = create_model(
        "SubagentTask",
        specialist=(SpecialistLiteral, Field(..., description="Which active specialist should handle this task.")),
        task=(str, Field(..., description="Self-contained instructions — the specialist sees ONLY this text.")),
    )
    SpawnArgs = create_model(
        "SpawnSubagentArgs",
        tasks=(list[TaskModel], Field(..., min_length=1, description="One or more {specialist, task} pairs — run in parallel.")),
    )

    async def _spawn(tasks: list) -> dict:
        batch_id = str(uuid.uuid4())
        task_list = [(t.specialist if hasattr(t, "specialist") else t["specialist"],
                      t.task if hasattr(t, "task") else t["task"]) for t in tasks]

        results = await asyncio.gather(*[
            _run_specialist(specialist, task_text, session_id, batch_id, tool_registry, get_model, gated_tools)
            for specialist, task_text in task_list
        ])

        out: dict[str, Any] = {}
        for (specialist, task_text), result in zip(task_list, results):
            if "reassigned_to" in result:
                new_specialist = result["reassigned_to"]
                await _post_blackboard(
                    session_id, "C137", "decision",
                    f"{specialist}'s task reassigned to {new_specialist} (approval declined).",
                )
                follow_up = await _run_specialist(new_specialist, result["task"], session_id, batch_id, tool_registry, get_model, gated_tools)
                out[f"{specialist}→{new_specialist}"] = follow_up
            else:
                out[specialist] = result

        return out

    return StructuredTool.from_function(
        coroutine=_spawn,
        name="spawn_subagent",
        description=(
            "Delegate one or more self-contained sub-tasks to active specialist "
            "subagents. Multiple entries run in parallel. Each specialist sees ONLY "
            "the task text given here, not the rest of this conversation."
        ),
        args_schema=SpawnArgs,
    )
