# FORGE · Phase 2
**Agent C137 — Agentic Harness with Tool Use**

## Quick start

```bash
# 1. Copy env and fill in your llama-server URL
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt --break-system-packages

# 3. Start backend (port 8000)
python forge_run.py

# 4. Open UI in browser
start app.html         # Windows
open  app.html         # macOS
xdg-open app.html      # Linux
```

## Architecture

```
app.html  ──fetch/SSE──►  FastAPI (main.py)
                                │
                         agent.py (ReAct loop)
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              shell.py      repl.py      web.py
              fs.py
```

## Tools (always present — no MCP required)

| # | Tool | Category | Gate? |
|---|------|----------|-------|
| 1 | `shell_exec` | Shell | ✅ Yes |
| 2 | `fs_read` | Filesystem | No |
| 3 | `fs_write` | Filesystem | ✅ Yes |
| 4 | `fs_list` | Filesystem | No |
| 5 | `fs_delete` | Filesystem | ✅ Yes |
| 6 | `repl_run` | Python REPL | No |
| 7 | `web_fetch` | Web | No |
| 8 | `web_search` | Web | No |

**Gate tools** (`shell_exec`, `fs_write`, `fs_delete`) require explicit
human approval in the UI before execution. All other tools auto-execute.

## SSE Event Types

| Event | Payload | Description |
|-------|---------|-------------|
| `thinking` | `{text}` | Agent reasoning (`<think>` content) |
| `delta` | `{text}` | Streamed response text |
| `tool_start` | `{call_id, name, args}` | Tool call initiated |
| `tool_gate` | `{call_id, name, args}` | **Paused — needs approval** |
| `tool_result` | `{call_id, name, output}` | Tool execution result |
| `done` | — | Turn complete |
| `error` | `{detail}` | Error occurred |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/sessions` | Create session |
| GET | `/sessions` | List sessions |
| GET | `/sessions/{id}` | Get session |
| PATCH | `/sessions/{id}` | Rename session |
| DELETE | `/sessions/{id}` | Delete session |
| GET | `/sessions/{id}/history` | Get message history |
| POST | `/chat` | Send message → SSE stream |
| POST | `/chat/approve` | Approve/deny gated tool |
| GET | `/health` | Health check |

## Folder structure

```
forge2/
├── app.html              # Zero-dependency frontend
├── forge_run.py          # python forge_run.py to start
├── requirements.txt
├── .env.example
└── backend/
    ├── __init__.py
    ├── config.py         # Env vars
    ├── db.py             # SQLite session/message storage
    ├── llm.py            # AsyncOpenAI client factory
    ├── agent.py          # LangGraph-style ReAct loop
    ├── main.py           # FastAPI routes
    └── tools/
        ├── __init__.py   # TOOL_REGISTRY + TOOL_SCHEMAS + GATE_TOOLS
        ├── shell.py      # shell_exec
        ├── fs.py         # fs_read/write/list/delete
        ├── repl.py       # persistent Python REPL
        └── web.py        # web_fetch + web_search
```

## Connecting to llama-server (Kaggle)

In the Kaggle notebook:
```bash
./llama-server \
  -m /path/to/model.gguf \
  -ngl 99 -ts 1,1 -c 65536 -np 2 -fa on --jinja \
  --port 8080
```

Then expose via Cloudflare tunnel and paste the URL into `.env`.
