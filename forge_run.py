"""Entry point — run with: python forge_run.py"""
import uvicorn

from backend.config import BIND_HOST, BIND_PORT, DEV_RELOAD

if __name__ == "__main__":
    # Previously hardcoded host="0.0.0.0", reload=True: exposed the
    # unauthenticated shell/fs/repl tools to the whole LAN by default,
    # and a file-watcher restart could silently orphan in-flight
    # approvals/turns. Now defaults to loopback-only, reload off — both
    # overridable via FORGE_BIND_HOST / FORGE_DEV_RELOAD (see
    # backend/config.py).
    uvicorn.run(
        "backend.main:app",
        host=BIND_HOST,
        port=BIND_PORT,
        reload=DEV_RELOAD,
    )
