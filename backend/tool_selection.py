"""
backend/tool_selection.py — dynamic per-turn tool binding (P2, off by
default — see backend/config.py's DYNAMIC_TOOL_BINDING_ENABLED comment
for why this ships disabled and what the real risk is).

A deliberately crude keyword heuristic, same spirit as
backend/tools/search_skills.py's _matches: a tool is "relevant" to the
current turn if any non-trivial word from its name/description also
appears somewhere in the conversation so far. Checked against the
WHOLE conversation (every message in the compacted history passed to
this agent_node call), not just the latest message — a tool whose need
only becomes apparent once the model starts reasoning, rather than
being obvious from a vague first message, stays eligible as long as
anything relevant was said earlier in the same turn-chain.

ALWAYS_BOUND_TOOL_NAMES are exempt from filtering entirely: either
because they're cheap, near-universally useful discovery primitives
(fs_read, fs_list, memory_read/write), or because the whole app's
design depends on them always being reachable (search_skills — the
Tier-3 unlock path itself; blackboard_post/schedule_task — compulsory
per the README; spawn_subagent — deployed mode's entire delegation
design assumes C137 can always reach it). Filtering is only ever
applied to the remaining, more narrowly-triggered/heavier tools
(shell_exec, fs_write, fs_delete, repl_run, web_fetch, web_search,
browser_action, computer_action).
"""
from __future__ import annotations

import re

from backend.config import DYNAMIC_TOOL_BINDING_ENABLED

ALWAYS_BOUND_TOOL_NAMES: frozenset[str] = frozenset({
    "fs_read", "fs_list", "memory_read", "memory_write",
    "search_skills", "blackboard_post", "schedule_task", "spawn_subagent",
})

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "this", "that", "it", "as", "by", "be", "if", "your",
    "you", "can", "will", "not", "use", "using", "when", "from",
})

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 3 and w not in _STOPWORDS}


def select_relevant_tools(tools: list, conversation_text: str) -> list:
    """
    Returns the subset of `tools` to actually bind this turn. A no-op
    (returns `tools` unchanged) when DYNAMIC_TOOL_BINDING_ENABLED is
    false, so this function is always safe to call — callers don't
    need their own feature-flag check.

    If filtering would leave NOTHING bound beyond the always-bound
    core (the conversation matched none of the filterable tools' own
    keywords — most likely a very short/early conversation with no
    signal yet), this fails open and returns the full, unfiltered list
    rather than risk starving the model of a tool it needs with no way
    to ask for it back.
    """
    if not DYNAMIC_TOOL_BINDING_ENABLED:
        return tools

    convo_keywords = _keywords(conversation_text)
    kept: list = []
    any_filterable_matched = False

    for tool in tools:
        if tool.name in ALWAYS_BOUND_TOOL_NAMES:
            kept.append(tool)
            continue
        tool_keywords = _keywords(f"{tool.name} {tool.description}")
        if convo_keywords & tool_keywords:
            kept.append(tool)
            any_filterable_matched = True

    if not any_filterable_matched:
        return tools

    return kept