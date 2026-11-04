"""
agent.py — LangGraph ReAct agent loop with live token streaming, the
approval gate, and session-scoped subagent deployment.

Architecture note — why "tools" handles exactly ONE call per node
invocation, not a loop over all of them
────────────────────────────────────────────────────────────────────
LangGraph resumes a paused node by re-running its function from the
top, replaying earlier state via the checkpoint — it does NOT resume
mid-function. Verified directly against this app's installed
langgraph (1.2.19): a node that loops over several tool calls and
calls interrupt() partway through would re-execute every earlier
call's side effects a second time on resume. Confirmed by a scripted
probe before writing this: with each tool call as its own node
invocation (looping back to "tools" via a conditional edge until
`pending_tool_calls` is empty), a 3-call sequence with the middle call
gated executes call 1 once, pauses, and on resume executes call 2 and
call 3 exactly once each — no duplication. That one-call-per-invocation
shape is required for the approval gate to be safe, and is the one
real contract change from the earlier LangGraph migration (Stage A),
which had no approval gate and could safely loop over every call in
one node execution.

Two different approval mechanisms, by design
────────────────────────────────────────────────────────────────────
C137's OWN gated calls pause via this module's checkpointed
interrupt()/Command(resume=...) — the graph genuinely halts, and the
original /chat connection closes until POST /sessions/{id}/approve
opens a new one that resumes it (main.py).

A SPECIALIST's gated call (inside spawn_subagent, backend/tools/
subagent.py) uses a separate, simpler asyncio.Event-based pause — see
that module's docstring for why nesting a second checkpointed
interrupt there was deliberately avoided.

Restart durability (P0 hardening) — why only this gate's index moved to the DB
──────────────────────────────────────────────────────────────────────
A C137-level pause is genuinely resumable after a restart: the actual
paused graph state already lives durably in checkpoints.db
(AsyncSqliteSaver). The only thing that previously lived ONLY in
process memory — and so orphaned every such pending approval on a
restart — was the call_id → thread_id index this module used to keep
in a plain dict. That index now lives entirely in backend/db.py's
pending_graph_approvals table (there is deliberately no in-memory
dict mirroring it anymore: an earlier draft of this kept one as a
"cache," but nothing on the hot path ever read it for a decision, and
main.py's approve handler resolves straight from the DB — so the
dict would only ever have grown, never shrink, since nothing popped
it on that path; dropped rather than shipped half-wired). On startup,
rearm_pending_approvals() reloads anything left over from a previous
process and gives each one a fresh APPROVAL_TIMEOUT_SECONDS window.

PENDING_SPECIALIST_APPROVALS (backend/tools/subagent.py) and this
module's own cancel_event were deliberately NOT moved to the DB: both
are tied to a live coroutine or an open SSE connection in what is, by
definition, the process that just died. There is no checkpoint of a
specialist's mid-task coroutine anywhere, so persisting that dict's
metadata couldn't let anything be resumed — only recorded as orphaned.
That's a real feature, just a different one, and out of scope here.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Annotated, AsyncGenerator, TypedDict

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, convert_to_messages
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from backend.checkpointer import get_checkpointer
from backend.config import APPROVAL_TIMEOUT_SECONDS
from backend.db import (
    add_pending_graph_approval,
    append_message,
    delete_pending_graph_approval,
    get_pending_graph_approval,
    get_session_config,
    list_pending_graph_approvals,
    pop_pending_graph_approval,
    record_thread,
)
from backend.llm import get_model
from backend.logging_config import get_logger
from backend.tools import GATED_TOOLS, TOOL_REGISTRY, TOOLS
from backend.tools.blackboard import make_blackboard_tool
from backend.tools.subagent import make_spawn_subagent_tool
from backend.tools.search_skills import SKILL_REGISTRY
from backend.tools.scheduler import make_schedule_task_tool

logger = get_logger("agent")

BASE_SYSTEM_PROMPT = """You are C137, a capable agentic AI assistant with access to tools.

When you need to reason before acting, wrap your thoughts in <think>…</think>.
After reasoning, either call a tool or respond directly to the user.

Guidelines:
- Use tools only when needed — don't call tools for things you already know.
- For computation or data analysis, prefer repl_run.
- For research, use web_search then web_fetch for full content.
- Never guess file contents — use fs_read.
- Format responses clearly using markdown — use headers, bullet lists, bold, and code blocks where appropriate.
- Be concise and direct in your final response.
- Use uuid_gen only when a new, unique identifier is actually needed; never call it just to decorate a response.
- Post important findings, decisions, blockers, or context to the shared blackboard with blackboard_post so other agents and the user can see them without digging through tool-call history.
- Not every tool you might need is in your list. If something you need isn't there, call search_skills with keywords describing it — a match becomes callable by name on your NEXT turn, not the same one.
- Use memory_read/memory_write for anything that should persist across sessions (not this one conversation). Use schedule_task only when the user asks for something to happen later or in the background, not for anything you can just do now.
"""

DEPLOYED_SYSTEM_PROMPT_ADDENDUM = """
Subagents are deployed this session: {specialists}. You are planner-first —
prefer delegating a task to the specialist whose scope fits it via
spawn_subagent, running multiple specialists in parallel when a turn needs
more than one. If no active specialist's scope fits a task, handle it
yourself directly with your own full toolset rather than forcing a bad
match or blocking.
"""

# ── incremental <think> tag parser ────────────────────────────────────────────
# Kept verbatim from the pre-approval-gate implementation — do not change
# the 7-char / 8-char windows (see the Stage A migration notes: this
# matches the ORIGINAL, pre-migration FORGE code's matching behaviour,
# bug-for-bug, including a verified pre-existing defect in the closing
# "</think>" match — see the migration's git history for details).

class _ThinkParser:
    def __init__(self):
        self._state   = "RESPONSE"   # "RESPONSE" | "IN_THINK"
        self._partial = ""

    def feed(self, token: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        i = 0
        while i < len(token):
            ch = token[i]
            if self._state == "RESPONSE":
                if ch == "<":
                    self._partial = "<"
                    i += 1
                    while i < len(token) and len(self._partial) < 7:
                        self._partial += token[i]
                        i += 1
                    if self._partial == "<think>":
                        self._state   = "IN_THINK"
                        self._partial = ""
                    elif "<think>".startswith(self._partial):
                        pass
                    else:
                        out.append(("delta", self._partial))
                        self._partial = ""
                else:
                    out.append(("delta", ch))
                    i += 1
            else:  # IN_THINK
                if ch == "<":
                    self._partial = "<"
                    i += 1
                    while i < len(token) and len(self._partial) < 7:
                        self._partial += token[i]
                        i += 1
                    if self._partial == "</think>":
                        self._state   = "RESPONSE"
                        self._partial = ""
                    elif "</think>".startswith(self._partial):
                        pass
                    else:
                        out.append(("thinking", self._partial))
                        self._partial = ""
                else:
                    out.append(("thinking", ch))
                    i += 1
        return out

    def flush(self) -> list[tuple[str, str]]:
        if self._partial:
            label = "thinking" if self._state == "IN_THINK" else "delta"
            out = [(label, self._partial)]
            self._partial = ""
            return out
        return []


# ── graph state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_tool_calls: list[dict]
    unlocked_skills: list[str]


# ── per-turn tool/model setup ───────────────────────────────────────────────

async def _build_turn_tools(session_id: str) -> tuple[list, dict, str]:
    """
    Returns (tools, tool_registry, system_prompt) for this session —
    solo mode gets the base roster + blackboard + schedule_task;
    deployed mode also gets spawn_subagent, scoped to that session's
    active_specialists. Tier-3 skills in SKILL_REGISTRY are NOT
    included here regardless of mode — see search_skills.py; they're
    added per-call in agent_node based on state["unlocked_skills"].
    """
    cfg = await get_session_config(session_id)
    blackboard_tool = make_blackboard_tool(session_id, "C137")
    schedule_tool = make_schedule_task_tool(session_id)
    tools = [*TOOLS, blackboard_tool, schedule_tool]
    registry = {**TOOL_REGISTRY, blackboard_tool.name: blackboard_tool, schedule_tool.name: schedule_tool}
    system_prompt = BASE_SYSTEM_PROMPT

    if cfg["subagents_enabled"] and cfg["active_specialists"]:
        spawn_tool = make_spawn_subagent_tool(
            session_id=session_id,
            active_specialists=cfg["active_specialists"],
            tool_registry=TOOL_REGISTRY,
            get_model=get_model,
            gated_tools=GATED_TOOLS,
        )
        tools.append(spawn_tool)
        registry[spawn_tool.name] = spawn_tool
        system_prompt += DEPLOYED_SYSTEM_PROMPT_ADDENDUM.format(specialists=", ".join(cfg["active_specialists"]))

    return tools, registry, system_prompt


def _build_graph(tools: list, tool_registry: dict, checkpointer):
    async def agent_node(state: AgentState) -> dict:
        unlocked = [SKILL_REGISTRY[s] for s in state.get("unlocked_skills", []) if s in SKILL_REGISTRY]
        model = get_model().bind_tools([*tools, *unlocked])
        ai_message = await model.ainvoke(state["messages"])
        pending = ai_message.tool_calls if isinstance(ai_message, AIMessage) else []
        return {"messages": [ai_message], "pending_tool_calls": pending}

    async def tools_node(state: AgentState) -> dict:
        pending = state["pending_tool_calls"]
        if not pending:
            return {}
        tc, *rest = pending
        call_id, name, args = tc["id"], tc["name"], tc["args"]

        newly_unlocked: list[str] = []

        if name in GATED_TOOLS:
            decision = interrupt({"call_id": call_id, "name": name, "args": args})
        else:
            decision = {"decision": "approve"}

        if decision.get("decision") != "approve":
            output = json.dumps({"ok": False, "error": "Denied by user (or timed out waiting for approval)."})
        else:
            await adispatch_custom_event("tool_start", {"call_id": call_id, "name": name, "args": args})
            # SKILL_REGISTRY tools aren't in tool_registry (they're only
            # bound in agent_node once unlocked) — check there too.
            tool = tool_registry.get(name) or SKILL_REGISTRY.get(name)
            if tool is None:
                output = json.dumps({"error": f"Unknown tool: {name}"})
            else:
                try:
                    result = await tool.ainvoke(args)
                    output = json.dumps(result, ensure_ascii=False)
                    if name == "search_skills" and isinstance(result, dict):
                        newly_unlocked = [m["name"] for m in result.get("matches", [])]
                except Exception as exc:
                    logger.exception("tool %s (call_id=%s) raised", name, call_id)
                    output = json.dumps({"error": str(exc)})
            await adispatch_custom_event("tool_result", {"call_id": call_id, "name": name, "output": output})

        tm = ToolMessage(content=output, tool_call_id=call_id, name=name)
        updated_unlocked = list(set(state.get("unlocked_skills", []) + newly_unlocked))
        return {"messages": [tm], "pending_tool_calls": rest, "unlocked_skills": updated_unlocked}

    def route_after_agent(state: AgentState) -> str:
        return "tools" if state["pending_tool_calls"] else END

    def route_after_tools(state: AgentState) -> str:
        return "tools" if state["pending_tool_calls"] else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    builder.add_conditional_edges("tools", route_after_tools, {"tools": "tools", "agent": "agent"})
    return builder.compile(checkpointer=checkpointer)


# ── shared event-translation loop ───────────────────────────────────────────

async def _consume(graph, run_input, config, cancel_event, session_id: str) -> AsyncGenerator[str, None]:
    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    parsers: dict[str, _ThinkParser] = {}

    def parser_for(key: str) -> _ThinkParser:
        if key not in parsers:
            parsers[key] = _ThinkParser()
        return parsers[key]

    try:
        async for event in graph.astream_events(run_input, config=config, version="v2"):
            if cancelled():
                yield sse({"type": "cancelled"})
                return

            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = getattr(chunk, "content", "") or ""
                if isinstance(content, list):
                    text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                else:
                    text = content

                subagent = (event.get("metadata") or {}).get("subagent")
                key = subagent or "__main__"
                if text:
                    for label, seg in parser_for(key).feed(text):
                        if not seg:
                            continue
                        if subagent:
                            parent_call_id = (event.get("metadata") or {}).get("parent_call_id")
                            yield sse({"type": f"subagent_{label}", "parent_call_id": parent_call_id, "subagent": subagent, "text": seg})
                        else:
                            yield sse({"type": label, "text": seg})

            elif kind == "on_chat_model_end":
                subagent = (event.get("metadata") or {}).get("subagent")
                key = subagent or "__main__"
                for label, seg in parser_for(key).flush():
                    if not seg:
                        continue
                    if subagent:
                        parent_call_id = (event.get("metadata") or {}).get("parent_call_id")
                        yield sse({"type": f"subagent_{label}", "parent_call_id": parent_call_id, "subagent": subagent, "text": seg})
                    else:
                        yield sse({"type": label, "text": seg})
                parsers[key] = _ThinkParser()

            elif kind == "on_custom_event":
                yield sse({"type": event["name"], **event["data"]})

        # Loop ended naturally — either the graph finished, or a
        # checkpointed interrupt() paused it. Distinguish the two by
        # inspecting the checkpointed state.
        state = await graph.aget_state(config)
        if state.next:
            interrupt_value = state.tasks[0].interrupts[0].value
            thread_id = config["configurable"]["thread_id"]
            call_id = interrupt_value["call_id"]
            await add_pending_graph_approval(call_id, thread_id, session_id)
            asyncio.create_task(_auto_deny_after_timeout(call_id, thread_id, session_id))
            yield sse({"type": "approval_required", **interrupt_value})
        else:
            yield sse({"type": "done"})

    except GraphRecursionError:
        logger.warning("session %s: hit recursion limit, ending turn", session_id)
        yield sse({"type": "done"})

    except Exception as exc:
        logger.exception("session %s: unhandled error in agent loop", session_id)
        yield sse({"type": "error", "detail": str(exc)})


async def _auto_deny_after_timeout(call_id: str, thread_id: str, session_id: str) -> None:
    """
    Scheduled the moment a C137-level gated call pauses the graph (or,
    after a restart, re-armed fresh by rearm_pending_approvals below).
    If nobody resolves it via POST /sessions/{id}/approve within
    APPROVAL_TIMEOUT_SECONDS, resumes it with a deny itself and
    persists the resulting turn straight to the DB — the original
    /chat connection is long since closed, so there's no live stream
    to forward this through.
    """
    await asyncio.sleep(APPROVAL_TIMEOUT_SECONDS)
    # The DB row is the only source of truth — a real approve/deny via
    # main.py's pop_pending_graph_approval removes it directly, possibly
    # from a different process than the one that scheduled this
    # timeout, if a restart happened in between.
    if await get_pending_graph_approval(call_id) is None:
        return  # already resolved by a real approve/deny
    await delete_pending_graph_approval(call_id)

    parts: list[str] = []
    async for chunk in resume_agent(session_id, thread_id, {"decision": "deny", "reason": "timeout"}):
        try:
            payload = json.loads(chunk.removeprefix("data: ").strip())
            if payload.get("type") == "delta":
                parts.append(payload.get("text", ""))
        except Exception:
            pass

    full_response = "".join(parts)
    if full_response.strip():
        await append_message(session_id, "assistant", full_response)


async def rearm_pending_approvals() -> int:
    """
    Called once from main.py's FastAPI lifespan, after get_checkpointer()
    warms up and before the server starts accepting requests. Reloads
    any pending_graph_approvals rows left behind by a previous process
    (i.e. the server was restarted while a C137-level gated call was
    awaiting approval) and gives each one a FRESH APPROVAL_TIMEOUT_SECONDS
    window — chosen over resuming the original window's remaining time,
    since there's no reliable way to tell a user "you have 40 seconds
    left" across a restart they weren't necessarily watching for.

    The checkpointed graph state for each thread_id is untouched by any
    of this — it's still sitting in checkpoints.db exactly as it was,
    which is what makes this safe to re-arm rather than just re-deny.
    """
    rows = await list_pending_graph_approvals()
    for row in rows:
        call_id, thread_id, session_id = row["call_id"], row["thread_id"], row["session_id"]
        asyncio.create_task(_auto_deny_after_timeout(call_id, thread_id, session_id))
        logger.info("re-armed pending graph approval %s (session %s) after restart", call_id, session_id)
    return len(rows)


# ── public entry points ──────────────────────────────────────────────────────

async def stream_agent(
    session_id: str,
    history: list[dict],
    max_iterations: int = 12,
    cancel_event=None,
) -> AsyncGenerator[str, None]:
    """Starts a fresh turn. See module docstring for the approval-gate shape."""
    tools, registry, system_prompt = await _build_turn_tools(session_id)
    checkpointer = await get_checkpointer()
    graph = _build_graph(tools, registry, checkpointer)

    messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *convert_to_messages(history)]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_iterations * 2}

    # Recorded so backend/checkpointer.py's pruning job can find and
    # clean up this turn's checkpoints.db rows once it's old enough —
    # see db.py's turn_threads table.
    await record_thread(thread_id, session_id)

    initial_state = {"messages": messages, "pending_tool_calls": [], "unlocked_skills": []}
    async for chunk in _consume(graph, initial_state, config, cancel_event, session_id):
        yield chunk


async def resume_agent(
    session_id: str,
    thread_id: str,
    decision: dict,
    max_iterations: int = 12,
    cancel_event=None,
) -> AsyncGenerator[str, None]:
    """
    Continues a turn paused on a C137-level gated call. `decision` is
    e.g. {"decision": "approve"} or {"decision": "deny"}. Rebuilds
    tools/graph fresh from this session's (immutable, locked-at-start)
    config — the paused VALUES live in the checkpointer by thread_id,
    not in the graph object, so rebuilding it here is correct.
    """
    tools, registry, _system_prompt = await _build_turn_tools(session_id)
    checkpointer = await get_checkpointer()
    graph = _build_graph(tools, registry, checkpointer)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_iterations * 2}

    async for chunk in _consume(graph, Command(resume=decision), config, cancel_event, session_id):
        yield chunk