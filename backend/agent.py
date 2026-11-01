"""
agent.py — LangGraph ReAct agent loop with live token streaming.

Phase 3 integration
────────────────────
This replaces the hand-written ReAct loop with a real LangGraph
StateGraph (langchain_openai.ChatOpenAI + LangChain StructuredTool
instances), while preserving the external contract exactly:
  - same SSE event shapes (thinking | delta | tool_start | tool_result |
    done | cancelled | error)
  - _ThinkParser kept verbatim (7-char "<think>" / 8-char "</think>"
    exact-match windows, no tolerance added)
  - tool calls still executed SEQUENTIALLY, never in parallel
  - the ReAct budget is still effectively 12 model calls
  - cancel_event still stops the run and yields exactly one "cancelled"
  - only the final plain-text answer is persisted (main.py does that,
    unchanged, by accumulating "delta" events)

No approval gate, no parallel tool execution, no checkpointing, and no
persistence of individual tool calls are added here — those are
separate, later changes.

Behaviour-change note (called out per the migration guardrails;
verified by actually running both cases against a scripted mock
model, not guessed from reading)
─────────────────────────────────────────────────────────────────
The old code manually accumulated each tool call's raw JSON argument
string and fell back to `{}` on `json.JSONDecodeError`:

    try:
        args = json.loads(tc["args_str"] or "{}")
    except json.JSONDecodeError:
        args = {}

...so a malformed-args tool call still RAN, just with no arguments.

LangChain's tool-call reconstruction is actually more lenient than
that in one respect and less visible in another:

  - Plain truncation (a tag cut mid-string, e.g. `{"path": "forge.db`
    with a missing closing quote/brace) is silently REPAIRED by
    langchain_core's partial-JSON parser — confirmed directly:
    `parse_partial_json('{"path": "forge.db')` returns
    `{'path': 'forge.db'}`. This is strictly better than the old
    `{}`-fallback, which would have discarded the partial path.

  - Genuinely invalid JSON (e.g. the literal text `not json at all`)
    is NOT retried as `{}` and does NOT raise out of
    `model.ainvoke(...)` either. Confirmed by direct inspection: the
    resulting AIMessage puts that call in `.invalid_tool_calls`
    (`[{'name': 'fs_read', 'args': 'not json at all', 'error': None,
    ...}]`) and leaves `.tool_calls` EMPTY. `_route_tools` only looks
    at `.tool_calls`, so this call is silently dropped — no
    `tool_start`/`tool_result`, no error, no text — the turn just
    ends with "done" and nothing else happened. This is a real,
    separate behaviour change: the old code at least ran something
    visible; this silently does nothing. Flagging it as a known gap
    for a follow-up fix (e.g. `_route_tools`/`_tools_node` could also
    walk `ai_message.invalid_tool_calls` and surface each as a
    visible tool_result-shaped error) rather than patching it inside
    this migration, since doing it safely means not breaking the
    next turn's tool_call_id/tool-message pairing contract with the
    upstream API — a change worth its own review, not a one-line fix
    made under this task's time box.
"""
from __future__ import annotations

import json
from typing import Annotated, AsyncGenerator, TypedDict

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, convert_to_messages
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from backend.llm import get_model
from backend.tools import TOOL_REGISTRY, TOOLS

SYSTEM_PROMPT = """You are C137, a capable agentic AI assistant with access to tools.

When you need to reason before acting, wrap your thoughts in <think>…</think>.
After reasoning, either call a tool or respond directly to the user.

Guidelines:
- Use tools only when needed — don't call tools for things you already know.
- For computation or data analysis, prefer repl_run.
- For research, use web_search then web_fetch for full content.
- Never guess file contents — use fs_read.
- Format responses clearly using markdown — use headers, bullet lists, bold, and code blocks where appropriate.
- Be concise and direct in your final response.
- Use uuid_gen only when a new, unique identifier is actually needed (e.g. naming a new record); never call it just to decorate a response.
"""

# ── incremental <think> tag parser ────────────────────────────────────────────
# Tracks streaming state across token chunks so we can emit thinking vs
# response events live without waiting for the full buffer.
# Kept verbatim from the pre-migration implementation — do not change the
# 7-char / 8-char exact-match windows, and do not add case/whitespace
# tolerance.

class _ThinkParser:
    """
    Feed tokens one at a time; get back labelled segments.
    States: RESPONSE | IN_THINK | DRAINING_CLOSE
    """
    def __init__(self):
        self._state   = "RESPONSE"   # "RESPONSE" | "IN_THINK"
        self._partial = ""           # accumulates partial tag chars

    def feed(self, token: str) -> list[tuple[str, str]]:
        """
        Returns list of (label, text) where label ∈ {"thinking","delta"}.
        """
        out: list[tuple[str, str]] = []
        i = 0
        while i < len(token):
            ch = token[i]
            if self._state == "RESPONSE":
                if ch == "<":
                    self._partial = "<"
                    i += 1
                    # consume ahead to see if this is <think>
                    while i < len(token) and len(self._partial) < 7:
                        self._partial += token[i]
                        i += 1
                    if self._partial == "<think>":
                        self._state   = "IN_THINK"
                        self._partial = ""
                    elif "<think>".startswith(self._partial):
                        pass  # still accumulating — will resolve next chunk
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
                        pass  # still accumulating
                    else:
                        out.append(("thinking", self._partial))
                        self._partial = ""
                else:
                    out.append(("thinking", ch))
                    i += 1
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Emit any leftover partial tag as the appropriate type."""
        if self._partial:
            label = "thinking" if self._state == "IN_THINK" else "delta"
            out = [(label, self._partial)]
            self._partial = ""
            return out
        return []


# ── graph state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── nodes ─────────────────────────────────────────────────────────────────────

async def _agent_node(state: AgentState) -> dict:
    model = get_model().bind_tools(TOOLS)
    ai_message = await model.ainvoke(state["messages"])
    return {"messages": [ai_message]}


async def _tools_node(state: AgentState) -> dict:
    """
    Executes every tool call on the last AIMessage SEQUENTIALLY (never
    asyncio.gather, never a parallel ToolNode), emitting a "tool_start"
    custom event before each call and a "tool_result" custom event
    after — picked up by stream_agent's astream_events loop and
    translated 1:1 into the existing SSE contract.
    """
    ai_message = state["messages"][-1]
    tool_messages: list[ToolMessage] = []

    for tc in ai_message.tool_calls:
        call_id = tc["id"]
        name    = tc["name"]
        args    = tc["args"]

        await adispatch_custom_event("tool_start", {"call_id": call_id, "name": name, "args": args})

        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            output = json.dumps({"error": f"Unknown tool: {name}"})
        else:
            try:
                result = await tool.ainvoke(args)
                output = json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                output = json.dumps({"error": str(exc)})

        await adispatch_custom_event("tool_result", {"call_id": call_id, "name": name, "output": output})

        tool_messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))

    return {"messages": tool_messages}


def _route_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _tools_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges("agent", _route_tools, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


# Compiled once at import time — ephemeral, no checkpointer, no persistence.
_GRAPH = _build_graph()


# ── main streaming entry point ────────────────────────────────────────────────

async def stream_agent(
    session_id: str,
    history: list[dict],
    max_iterations: int = 12,
    cancel_event=None,
) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph ReAct loop and yield SSE-formatted strings.
    Tokens are emitted live — no full-buffer wait.
    Pass cancel_event and set() it to abort mid-stream.

    `session_id` is accepted for call-site compatibility with main.py
    (and for the session-scoped tool binding a later change adds); it
    is not otherwise used by this ephemeral, non-checkpointed graph.
    """

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        *convert_to_messages(history),
    ]

    # 12 model calls ≈ 12 agent->tools supersteps -> recursion_limit ~24.
    config = {"recursion_limit": max_iterations * 2}

    parser = _ThinkParser()

    try:
        async for event in _GRAPH.astream_events({"messages": messages}, config=config, version="v2"):
            if cancelled():
                yield sse({"type": "cancelled"})
                return

            kind = event["event"]

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = getattr(chunk, "content", "") or ""
                if isinstance(content, list):
                    # Some providers emit content as a list of blocks
                    # (e.g. {"type": "text", "text": "..."}) rather than
                    # a plain string.
                    text = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                else:
                    text = content

                if text:
                    for label, seg in parser.feed(text):
                        if seg:
                            yield sse({"type": label, "text": seg})

            elif kind == "on_chat_model_end":
                for label, seg in parser.flush():
                    if seg:
                        yield sse({"type": label, "text": seg})
                # Fresh parser for the next model turn in this loop.
                parser = _ThinkParser()

            elif kind == "on_custom_event" and event["name"] in ("tool_start", "tool_result"):
                yield sse({"type": event["name"], **event["data"]})

        yield sse({"type": "done"})

    except GraphRecursionError:
        # Matches the pre-migration behaviour: exhausting the iteration
        # budget produces a successful-looking "done", not an error.
        yield sse({"type": "done"})

    except Exception as exc:
        yield sse({"type": "error", "detail": str(exc)})
