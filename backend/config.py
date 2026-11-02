import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL        = os.getenv("FORGE_BASE_URL", "http://localhost:8080")
API_KEY         = os.getenv("FORGE_API_KEY",  "sk-no-key")
MODEL_NAME      = os.getenv("FORGE_MODEL_NAME", "local-model")
DB_PATH         = os.getenv("FORGE_DB_PATH",  "forge.db")

# ── security hardening (P0) ─────────────────────────────────────────────

# Server bind address/port. Previously hardcoded to 0.0.0.0:8000 in
# forge_run.py, which exposes the unauthenticated shell/fs/repl tools to
# the whole LAN by default. Default is now loopback-only; set
# FORGE_BIND_HOST=0.0.0.0 explicitly to opt back into LAN exposure.
BIND_HOST = os.getenv("FORGE_BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.getenv("FORGE_BIND_PORT", "8000"))

# Disable uvicorn's --reload by default: a file-watcher restart silently
# drops PENDING_GRAPH_APPROVALS / PENDING_SPECIALIST_APPROVALS / any live
# cancel_event, orphaning in-flight approvals and turns with no operator
# warning. Opt in explicitly for local development.
DEV_RELOAD = os.getenv("FORGE_DEV_RELOAD", "false").lower() in ("1", "true", "yes")

# Locked-down CORS: previously allow_origins=["*"] with any
# method/header, meaning any page the operator's browser visits can
# drive this backend's unauthenticated tools cross-origin. Default to
# the two conventional local dev origins for app.html; override with a
# comma-separated list for anything else.
CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "FORGE_CORS_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,null",
    ).split(",")
    if o.strip()
]

# Optional shared-secret auth. Empty string (the default) means auth is
# OFF — unchanged behavior for existing single-operator setups. Set
# FORGE_API_AUTH_KEY to require every request to carry a matching
# "X-Forge-Key" header.
API_AUTH_KEY = os.getenv("FORGE_API_AUTH_KEY", "")

# Workspace root for fs_read/fs_write/fs_list/fs_delete. Empty string
# (the default) preserves today's behavior (no containment, full
# filesystem access within the process's own permissions) so existing
# single-operator setups aren't broken by default. Set this to confine
# every fs_* call to one directory tree.
FS_WORKSPACE_ROOT = os.getenv("FORGE_FS_WORKSPACE_ROOT", "")

# Allow-list of shell_exec cwd directories, colon-separated. Empty
# (default) preserves today's behavior (any cwd). Only enforced when
# FS_WORKSPACE_ROOT is also set, so the two knobs move together.
SHELL_ALLOW_UNRESTRICTED_CWD = FS_WORKSPACE_ROOT == ""

# Approval gate + subagent orchestration — kept deliberately separate
# from forge.db (see backend/checkpointer.py).
CHECKPOINTS_DB_PATH      = os.getenv("FORGE_CHECKPOINTS_DB_PATH", "checkpoints.db")
APPROVAL_TIMEOUT_SECONDS = int(os.getenv("FORGE_APPROVAL_TIMEOUT_SECONDS", "300"))

# Informational only in this codebase: the actual concurrency limit is
# set on llama-server itself (its own --parallel flag), not read here.
# Kept so the number lives in one documented place.
SUBAGENT_PARALLEL = int(os.getenv("FORGE_SUBAGENT_PARALLEL", "4"))
