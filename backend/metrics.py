"""
backend/metrics.py — structured tracing + cost accounting (P1 hardening).

Wraps the three kinds of timed work this app does — an agent_node LLM
call, a tools_node tool execution, and a specialist's own LLM/tool
calls inside spawn_subagent — in a single async context manager that
records one row per span to backend/db.py's turn_metrics table.

Token counts: ChatOpenAI populates AIMessage.usage_metadata from the
provider's `usage` field when available, but llama-server (llama.cpp)
streaming responses don't reliably return usage data the way OpenAI's
API does — that requires `stream_options.include_usage`, and llama.cpp
server support for it has been inconsistent across versions. So this
always tries real usage first and falls back to the chars÷4 heuristic
already used for compaction decisions (backend/compaction.py's
chars_to_tokens) when it's missing, recording which one happened via
tokens_estimated rather than ever presenting a guess as if it were
measured.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from backend.compaction import chars_to_tokens
from backend.db import record_turn_metric
from backend.logging_config import get_logger

logger = get_logger("metrics")


@asynccontextmanager
async def span(session_id: str | None, span_type: str, name: str | None = None) -> AsyncIterator[dict]:
    """
    Usage:
        async with span(session_id, "agent_node") as m:
            ai_message = await ainvoke_with_retry(model, messages)
            record_llm_usage(m, ai_message, input_text=..., output_text=...)

    `m` is a plain dict the caller fills in before the `with` block
    exits (input_tokens / output_tokens / tokens_estimated) — anything
    left unset is recorded as NULL/False. Records the span (including
    duration and, if the block raised, the exception's str()) even on
    failure, via the `finally`, so a tool/LLM call that errors out
    still shows up in the trace rather than vanishing.

    session_id=None is a deliberate escape hatch for callers that build
    a graph without a real session (tests/test_approval_gate.py's fake
    model, for one) — the wrapped code still runs exactly the same, it
    just doesn't produce a turn_metrics row (there'd be no valid
    session_id to satisfy the table's foreign key against anyway).
    """
    start = time.time()
    metrics: dict = {}
    raised_error: str | None = None
    try:
        yield metrics
    except Exception as exc:
        raised_error = str(exc)
        raise
    finally:
        if session_id is not None:
            end = time.time()
            # A caller whose own try/except already swallows the
            # failure (e.g. agent.py's tools_node, which converts a
            # tool exception into an error-output ToolMessage rather
            # than letting it propagate) can still report it by setting
            # metrics["error"] itself before the block exits — that
            # takes priority over whatever did or didn't propagate out
            # of this context manager, since letting a swallowed tool
            # error show up as error=NULL here would silently lose
            # exactly the information a trace exists to capture.
            final_error = metrics.get("error", raised_error)
            try:
                await record_turn_metric(
                    session_id=session_id,
                    span_type=span_type,
                    name=name,
                    start_ts=start,
                    end_ts=end,
                    input_tokens=metrics.get("input_tokens"),
                    output_tokens=metrics.get("output_tokens"),
                    tokens_estimated=metrics.get("tokens_estimated", False),
                    error=final_error,
                )
            except Exception:
                # Metrics recording must never be why a turn fails — same
                # "best-effort, logged" posture as the rest of this app's
                # non-critical bookkeeping (e.g. checkpoint pruning).
                logger.exception("session %s: failed to record turn_metrics span (%s/%s)", session_id, span_type, name)


def record_llm_usage(metrics: dict, ai_message, input_text: str = "", output_text: str = "") -> None:
    """
    Fills `metrics` (the dict yielded by `span`) from an AIMessage's
    real usage_metadata if present, else from a char-count estimate of
    the given input/output text. Call this once inside the `async with
    span(...)` block, right after getting the model's response.
    """
    usage = getattr(ai_message, "usage_metadata", None)
    if usage and usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
        metrics["input_tokens"] = usage["input_tokens"]
        metrics["output_tokens"] = usage["output_tokens"]
        metrics["tokens_estimated"] = False
    else:
        metrics["input_tokens"] = chars_to_tokens(len(input_text))
        metrics["output_tokens"] = chars_to_tokens(len(output_text))
        metrics["tokens_estimated"] = True