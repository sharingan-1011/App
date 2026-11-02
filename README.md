# FORGE · Phase 3
**Agent C137 — Agentic Harness with Tool Use, Approval Gating & Subagent Delegation**

## Quick start

```bash
# 1. Copy env and fill in your llama-server URL
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt --break-system-packages

# 3. One-time: fetch the headless browser binary for browser_action
playwright install chromium

# 4. Start backend (port 8000)
python forge_run.py

# 5. Open UI in browser
start app.html         # Windows
open  app.html         # macOS
xdg-open app.html      # Linux
```

## Architecture

```
app.html  ──fetch/SSE──►  FastAPI (main.py)
                                │
                     agent.py — LangGraph StateGraph
                     (agent ⇄ tools, approval gate via
                      interrupt()/Command(resume=…))
                                │
              ┌─────────────────┼─────────────────────┐
              │                 │                      │
     Tier 1 — compulsory   Tier 2 — heavy/gated   Tier 3 — on demand
     fs_*, shell_exec,     browser_action,        SKILL_REGISTRY,
     repl_run, web_*,      computer_action         discovered via
     memory_*, search_                             search_skills
     skills, blackboard_
     post, schedule_task
              │
     spawn_subagent (deployed sessions only)
              │
     ┌────────┼────────┐
   Ba-56    La-57     Ce-58
  (coding) (research) (data)
```

A session runs in one of two modes, chosen once on its first message and
locked for that session's life:

- **Solo** — C137 has the full tool roster, no delegation.
- **Deployed** — C137 is planner-first, delegates to whichever of
  Ba-56 / La-57 / Ce-58 the session activated (each hard-restricted to
  its own tool subset), and falls back to its own full toolset only
  when no active specialist's scope fits the task.

## Tools

| Tool | Tier | Gated? | Notes |
|---|---|---|---|
| `shell_exec` | Core | ✅ | |
| `fs_read` | Core | No | |
| `fs_write` | Core | ✅ | |
| `fs_list` | Core | No | |
| `fs_delete` | Core | ✅ | |
| `repl_run` | Core | ✅ | Persistent Python REPL per session |
| `web_fetch` | Core | No | |
| `web_search` | Core | No | DuckDuckGo HTML scrape |
| `memory_read` / `memory_write` | Core | No | Flat key/value store, global across sessions (not per-session) |
| `search_skills` | Core | No | Matches a query against `SKILL_REGISTRY`; a match becomes callable from the *next* agent turn, not the same one |
| `blackboard_post` | Core | No | Session-scoped findings log, available to every active agent regardless of tier — the one tool that deliberately ignores the tiering walls |
| `schedule_task` | Core | No | Registers a one-off prompt to run as a fresh turn after a delay; a background loop polls every 15s |
| `browser_action` | Heavy | ✅ | Real headless-Chromium automation via Playwright |
| `computer_action` | Heavy | ✅ | **Not implemented.** Desktop/OS control needs a remote-control integration with the user's own machine that hasn't been built; always returns an honest not-implemented error. Gated anyway so the plumbing is ready. |
| `uuid_gen` | Skill (Tier 3) | No | Not bound by default — only reachable after `search_skills` matches it |
| `spawn_subagent` | — | — | Only exists in deployed-mode sessions; its `specialist` argument is scoped to that session's active specialists |

**Gated tools** pause the graph via LangGraph's `interrupt()`/`Command(resume=…)`
and require `POST /sessions/{id}/approve` before running. Left unresolved
for 5 minutes, a gated call is auto-denied. A specialist's own gated call
uses a lighter `asyncio.Event`-based pause instead (not a nested
interrupt), so its siblings keep running — see `backend/agent.py` and
`backend/tools/subagent.py`'s docstrings for why the two mechanisms
differ.

## SSE event types

| Event | Payload | Description |
|---|---|---|
| `thinking` | `{text}` | Agent reasoning (`<think>` content) |
| `delta` | `{text}` | Streamed response text |
| `tool_start` | `{call_id, name, args}` | Tool call starting |
| `tool_result` | `{call_id, name, output}` | Tool call finished |
| `approval_required` | `{call_id, name, args, subagent?, parent_call_id?}` | Paused on a gated call — `subagent` is present only when a specialist (not C137) is asking |
| `subagent_start` / `subagent_thinking` / `subagent_delta` / `subagent_tool_start` / `subagent_tool_result` / `subagent_done` | `{parent_call_id, subagent, …}` | A specialist's own turn, nested inside `spawn_subagent`'s `tool_result` |
| `blackboard_entry` | `{id, author, tag, text}` | Live push when any agent posts to the blackboard |
| `done` | — | Turn complete |
| `cancelled` | — | Stopped via the Stop button or a client disconnect |
| `error` | `{detail}` | Unhandled error |

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/sessions` | List sessions |
| POST | `/sessions` | Create a session — optional body `{subagents_enabled, active_specialists}`, locked for that session's life |
| GET | `/sessions/{id}` | Get session + message history |
| PATCH | `/sessions/{id}` | Rename session |
| DELETE | `/sessions/{id}` | Delete session |
| GET | `/sessions/{id}/config` | Read a session's deploy-mode config |
| GET | `/sessions/{id}/blackboard` | Full blackboard history for a session |
| GET | `/sessions/{id}/scheduled_tasks` | A session's scheduled tasks |
| GET | `/skills` | List `SKILL_REGISTRY` (for the frontend's `/` skill picker) |
| POST | `/chat` | Send a message → SSE stream |
| POST | `/sessions/{id}/approve` | Resolve a pending gated call — `{call_id, decision: "approve"\|"deny"\|"reassign", reassign_to?}` |
| POST | `/upload` | Upload a file for the next message to reference |

`reassign` only applies to a specialist's gated call (hand the task to a
different active specialist); on a top-level C137 call it falls back to
`deny`, since there's nowhere else for it to go.

## Folder structure

```
App/
├── app.html                   # Zero-dependency frontend
├── forge_run.py                # python forge_run.py to start
├── requirements.txt
├── pyproject.toml
├── .env.example
├── forge.db                    # sessions/messages/config/blackboard/memory/scheduled_tasks
├── checkpoints.db               # LangGraph approval-gate checkpoints — kept separate from forge.db
└── backend/
    ├── __init__.py
    ├── config.py                # Env vars
    ├── db.py                    # SQLite storage (forge.db)
    ├── checkpointer.py          # AsyncSqliteSaver for the approval gate (checkpoints.db)
    ├── llm.py                   # ChatOpenAI client factory
    ├── agent.py                 # LangGraph StateGraph: agent/tools nodes, approval gate, subagent + skill-unlock wiring
    ├── main.py                  # FastAPI routes
    └── tools/
        ├── __init__.py          # TOOLS / TOOL_REGISTRY / GATED_TOOLS
        ├── shell.py              # shell_exec
        ├── fs.py                 # fs_read/write/list/delete
        ├── repl.py                # persistent Python REPL
        ├── web.py                 # web_fetch + web_search
        ├── memory.py              # memory_read/memory_write
        ├── search_skills.py       # SKILL_REGISTRY + search_skills
        ├── uuid_gen.py             # the one Tier-3 skill registered today
        ├── blackboard.py          # blackboard_post
        ├── scheduler.py           # schedule_task + the polling loop
        ├── subagent.py            # spawn_subagent + Ba-56/La-57/Ce-58
        ├── browser.py              # browser_action (Playwright)
        └── computer.py             # computer_action (stub — see table above)
```

Not part of the live path, left as-is: `backend/chain.py`,
`backend/_verify.py`, `streamlit_app.py`, and `pyproject.toml`'s
`[project.scripts]` entry (malformed — use `python forge_run.py`
instead).

## Known limitations

- Approval/cancellation state (`PENDING_GRAPH_APPROVALS`,
  `PENDING_SPECIALIST_APPROVALS`, `cancel_event`) is in-process memory —
  lost on a server restart.
- `_ThinkParser`'s closing-tag match has a pre-existing off-by-one bug:
  its accumulate loop caps at 7 characters in both branches, one short
  of the 8 needed to match `</think>`, so the closing tag is never
  recognized and everything after it stays labelled "thinking". Present
  in the original Phase 2 code; preserved as-is through the LangGraph
  migration rather than fixed silently alongside it.
- A tool call with genuinely invalid (non-truncated) JSON arguments
  lands in the model response's `invalid_tool_calls` and is silently
  dropped — no visible error, the turn just ends. Flagged, not patched,
  since fixing it safely means not breaking the next turn's
  `tool_call_id` pairing with the upstream API.
- `computer_action` is a stub (see Tools table) — no real desktop
  control exists yet.
- `schedule_task` is a one-off "run this prompt at this time" primitive,
  not a cron/recurrence system.

## Connecting to llama-server (Kaggle)

In the Kaggle notebook:
```bash
./llama-server \
  -m /path/to/model.gguf \
  -ngl 99 -ts 1,1 -c 131072 -np 4 -fa on --jinja \
  --port 8080
```

`-np 4` matches the target concurrency (1 orchestrator + 3 specialist
slots) for deployed-mode sessions. Then expose via Cloudflare tunnel
and paste the URL into `.env`.
