"""
tools/__init__.py — registry of all fundamental tools.

Phase 3 integration: each tool is wrapped as a LangChain StructuredTool so
the LangGraph agent can bind them directly via model.bind_tools(TOOLS).
TOOL_REGISTRY / TOOL_SCHEMAS are kept, derived from TOOLS, for anything
that still reads the old shape (today: nothing in the live path does,
but main.py's SSE re-parsing and app.html's formatToolOutput key off
tool *names*, not off these constants, so nothing else needs to change).

All tools auto-execute — no approval gate at this layer. The approval
gate (Stage: gated-tool approval) sits in agent.py's graph, in front of
the tools node, not here.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from backend.tools.shell import shell_exec
from backend.tools.fs import fs_read, fs_write, fs_list, fs_delete
from backend.tools.repl import repl_run
from backend.tools.web import web_fetch, web_search
from backend.tools.memory import build_memory_tools
from backend.tools.search_skills import build_search_skills_tool
from backend.tools.browser import build_browser_tool
from backend.tools.computer import build_computer_tool


# ── args schemas — mirror the original TOOL_SCHEMAS parameters exactly ──────

class ShellExecArgs(BaseModel):
    cmd: str = Field(..., description="Command to execute")
    cwd: str | None = Field(None, description="Working directory (optional)")
    timeout: int = Field(30, description="Timeout in seconds (default 30)")
    use_shell: bool = Field(
        True,
        description=(
            "True (default): full shell interpretation (pipes, redirects, globbing). "
            "False: argv-exec, no shell metacharacter interpretation — prefer this for "
            "a single program + plain arguments."
        ),
    )


class FsReadArgs(BaseModel):
    path: str = Field(..., description="Absolute or relative file path")
    encoding: str = Field("utf-8", description="File encoding (default utf-8)")


class FsWriteArgs(BaseModel):
    path: str = Field(..., description="File path to write")
    content: str = Field(..., description="Content to write")
    mode: str = Field("w", description="'w' (overwrite) or 'a' (append), default 'w'")


class FsListArgs(BaseModel):
    path: str = Field(..., description="Directory path")
    recursive: bool = Field(False, description="List recursively (default false)")


class FsDeleteArgs(BaseModel):
    path: str = Field(..., description="Path to delete")
    recursive: bool = Field(False, description="Delete directory recursively (default false)")


class ReplRunArgs(BaseModel):
    code: str = Field(..., description="Python code to execute")
    session_id: str = Field("default", description="REPL session ID to maintain state (optional)")
    timeout: int = Field(60, description="Timeout in seconds (default 60)")


class WebFetchArgs(BaseModel):
    url: str = Field(..., description="URL to fetch")
    timeout: int = Field(15, description="Timeout in seconds (default 15)")
    raw_html: bool = Field(False, description="Return raw HTML instead of stripped text (default false)")


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Max results to return (default 5)")


# ── StructuredTool wrappers — descriptions preserved verbatim ───────────────
#
# Tier 1 (compulsory, always bound): fs_*, shell_exec, repl_run, web_*,
# memory_read/memory_write, search_skills. blackboard_post and
# spawn_subagent are session-scoped, built fresh per turn in agent.py —
# not listed here.
#
# Tier 2 (heavy, gated, bound alongside Tier 1 — small enough roster
# today that the "don't blindly load every turn" schema-bloat concern
# from the tiering design doesn't bite yet): browser_action,
# computer_action.
#
# Tier 3 (on-demand, NOT bound here): backend/tools/search_skills.py's
# SKILL_REGISTRY — currently just uuid_gen.

TOOLS: list[StructuredTool] = [
    StructuredTool.from_function(
        coroutine=shell_exec,
        name="shell_exec",
        description="Run a shell command on the local machine. Returns stdout and stderr.",
        args_schema=ShellExecArgs,
    ),
    StructuredTool.from_function(
        coroutine=fs_read,
        name="fs_read",
        description="Read a file and return its contents as text.",
        args_schema=FsReadArgs,
    ),
    StructuredTool.from_function(
        coroutine=fs_write,
        name="fs_write",
        description="Write or overwrite a file with the given content.",
        args_schema=FsWriteArgs,
    ),
    StructuredTool.from_function(
        coroutine=fs_list,
        name="fs_list",
        description="List files and directories at a path.",
        args_schema=FsListArgs,
    ),
    StructuredTool.from_function(
        coroutine=fs_delete,
        name="fs_delete",
        description="Delete a file or directory.",
        args_schema=FsDeleteArgs,
    ),
    StructuredTool.from_function(
        coroutine=repl_run,
        name="repl_run",
        description=(
            "Execute Python code in a persistent REPL session. "
            "Variables persist between calls within the same session."
        ),
        args_schema=ReplRunArgs,
    ),
    StructuredTool.from_function(
        coroutine=web_fetch,
        name="web_fetch",
        description="Fetch a URL and return the page content as plain text (HTML stripped).",
        args_schema=WebFetchArgs,
    ),
    StructuredTool.from_function(
        coroutine=web_search,
        name="web_search",
        description="Search the web using DuckDuckGo. Returns top N results with title, URL, and snippet.",
        args_schema=WebSearchArgs,
    ),
    *build_memory_tools(),
    build_search_skills_tool(),
    build_browser_tool(),
    build_computer_tool(),
]

# name -> StructuredTool, derived from TOOLS — agent.py's tools-node looks
# tool calls up here, same role TOOL_REGISTRY played before.
TOOL_REGISTRY: dict[str, StructuredTool] = {tool.name: tool for tool in TOOLS}

# Kept for compatibility with anything that still reads the old
# OpenAI-function-schema shape. Nothing in the live agent.py/main.py path
# reads this after the migration — model.bind_tools(TOOLS) is what the
# graph actually uses — but it's cheap to keep derived and in sync.
TOOL_SCHEMAS: list[dict] = [convert_to_openai_tool(tool) for tool in TOOLS]

# Gated tools: require explicit human approval before execution (the
# gate in agent.py's tools_node enforces this). computer_action is
# gated too even though it's a stub today — the gating is wired ahead
# of the real implementation, consistent with its own module docstring.
GATED_TOOLS: frozenset[str] = frozenset({
    "shell_exec", "fs_write", "fs_delete", "repl_run",
    "browser_action", "computer_action",
})
