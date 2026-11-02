"""
tests/test_approval_gate.py — regression test for the approval gate's
core safety contract.

backend/agent.py's module docstring describes a one-time, manual,
scripted probe run before the approval gate was trusted: verifying
that a 3-call tool sequence with the middle call gated executes call 1
once, pauses, and on resume executes call 2 and call 3 exactly once
each — no duplication. That was the single load-bearing correctness
property of the whole graph design, and until now it was never
automated — a LangGraph version bump could silently reintroduce
duplicate tool-call side effects (e.g. a shell_exec running twice)
with nothing to catch it. This test makes that probe repeatable.

Uses LangGraph's in-memory checkpointer (not AsyncSqliteSaver) and a
fake chat model (no real LLM call) so this runs fast and without
network/DB dependencies — it exercises backend/agent.py's actual
_build_graph/tools_node/agent_node logic, just with stubbed-out model
and gating.
"""
from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

import backend.agent as agent_module


class _NoArgs(BaseModel):
    pass


def _make_counting_tool(name: str, counts: dict):
    async def _run() -> dict:
        counts[name] = counts.get(name, 0) + 1
        return {"ok": True, "tool": name, "call_number": counts[name]}

    return StructuredTool.from_function(coroutine=_run, name=name, description=f"test tool {name}", args_schema=_NoArgs)


class _FakeModel:
    """
    Stands in for ChatOpenAI. First ainvoke() call (no prior tool
    results in the message list) returns an AIMessage requesting 3
    tool calls in sequence: tool_a (not gated), tool_b (gated),
    tool_c (not gated). Any later call (after tool results are present)
    returns a plain final answer with no further tool calls, ending
    the turn.
    """
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        from langchain_core.messages import ToolMessage
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="All done.")
        return AIMessage(
            content="",
            tool_calls=[
                {"id": "call-a", "name": "tool_a", "args": {}},
                {"id": "call-b", "name": "tool_b", "args": {}},
                {"id": "call-c", "name": "tool_c", "args": {}},
            ],
        )


async def _run_single_call_per_invocation_probe():
    counts: dict[str, int] = {}
    tool_a = _make_counting_tool("tool_a", counts)
    tool_b = _make_counting_tool("tool_b", counts)
    tool_c = _make_counting_tool("tool_c", counts)
    tools = [tool_a, tool_b, tool_c]
    registry = {t.name: t for t in tools}

    original_get_model = agent_module.get_model
    original_gated = agent_module.GATED_TOOLS
    agent_module.get_model = lambda: _FakeModel()
    agent_module.GATED_TOOLS = frozenset({"tool_b"})
    try:
        checkpointer = MemorySaver()
        graph = agent_module._build_graph(tools, registry, checkpointer)

        from langchain_core.messages import SystemMessage
        thread_id = "test-thread-1"
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 24}
        initial_state = {
            "messages": [SystemMessage(content="test"), *agent_module.convert_to_messages([{"role": "user", "content": "go"}])],
            "pending_tool_calls": [],
            "unlocked_skills": [],
        }

        await graph.ainvoke(initial_state, config=config)

        # tool_a (before the gated call) must have run exactly once,
        # and tool_b/tool_c must NOT have run yet — the graph should be
        # paused on tool_b's interrupt().
        assert counts.get("tool_a") == 1, f"tool_a should run exactly once before the pause, got {counts}"
        assert "tool_b" not in counts, "tool_b is gated and should not run before approval"
        assert "tool_c" not in counts, "tool_c comes after the gated call and should not run before it's resolved"

        state = await graph.aget_state(config)
        assert state.next, "graph should be paused (interrupted) on the gated call"

        # Resume with approval — tool_b should now run exactly once,
        # and critically tool_a must NOT be re-executed (the
        # regression this test exists to catch: LangGraph resumes a
        # paused node by re-running it from the top, so a looser graph
        # shape could replay tool_a's side effect a second time here).
        await graph.ainvoke(Command(resume={"decision": "approve"}), config=config)

        assert counts.get("tool_a") == 1, f"tool_a must not be re-executed on resume, got {counts}"
        assert counts.get("tool_b") == 1, f"tool_b should run exactly once after approval, got {counts}"
        assert counts.get("tool_c") == 1, f"tool_c should run exactly once after tool_b resolves, got {counts}"
    finally:
        agent_module.get_model = original_get_model
        agent_module.GATED_TOOLS = original_gated


def test_single_call_per_node_invocation():
    """
    Synchronous pytest entry point — runs the async probe via
    asyncio.run() so this test works even without a pytest-asyncio
    plugin installed.
    """
    asyncio.run(_run_single_call_per_invocation_probe())


def test_denied_gated_call_does_not_run_and_does_not_replay_earlier_calls():
    """A denied gated call should short-circuit (no tool execution),
    and — same regression concern as above — must not cause tool_a to
    re-run on resume."""
    async def _probe():
        counts: dict[str, int] = {}
        tool_a = _make_counting_tool("tool_a", counts)
        tool_b = _make_counting_tool("tool_b", counts)
        tool_c = _make_counting_tool("tool_c", counts)
        tools = [tool_a, tool_b, tool_c]
        registry = {t.name: t for t in tools}

        original_get_model = agent_module.get_model
        original_gated = agent_module.GATED_TOOLS
        agent_module.get_model = lambda: _FakeModel()
        agent_module.GATED_TOOLS = frozenset({"tool_b"})
        try:
            checkpointer = MemorySaver()
            graph = agent_module._build_graph(tools, registry, checkpointer)

            from langchain_core.messages import SystemMessage
            config = {"configurable": {"thread_id": "test-thread-2"}, "recursion_limit": 24}
            initial_state = {
                "messages": [SystemMessage(content="test"), *agent_module.convert_to_messages([{"role": "user", "content": "go"}])],
                "pending_tool_calls": [],
                "unlocked_skills": [],
            }
            await graph.ainvoke(initial_state, config=config)
            assert counts.get("tool_a") == 1

            await graph.ainvoke(Command(resume={"decision": "deny"}), config=config)

            assert counts.get("tool_a") == 1, "tool_a must not replay on a denied resume either"
            assert "tool_b" not in counts, "a denied gated tool must never actually execute"
            assert counts.get("tool_c") == 1, "tool_c (after the denied call) should still run once"
        finally:
            agent_module.get_model = original_get_model
            agent_module.GATED_TOOLS = original_gated

    asyncio.run(_probe())
