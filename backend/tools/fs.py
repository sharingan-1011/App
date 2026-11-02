"""
tools/fs.py — file-system tools: read, write, list, delete.

Path containment (security hardening)
──────────────────────────────────────────────────────────────────────
When backend.config.FS_WORKSPACE_ROOT is set, every path passed to
fs_read/fs_write/fs_list/fs_delete is resolved and checked against that
root; anything that resolves outside it is rejected before any syscall
touches it. When FS_WORKSPACE_ROOT is unset (the default), behavior is
unchanged from before this hardening pass — full access within the
process's own OS permissions, no containment — so existing
single-operator setups aren't broken without opting in.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path

from backend.config import FS_WORKSPACE_ROOT


class PathNotAllowed(Exception):
    pass


def _resolve_contained(path: str) -> Path:
    """
    Resolve `path` and, if FS_WORKSPACE_ROOT is configured, verify it
    falls inside that root. Raises PathNotAllowed otherwise. Resolves
    symlinks (Path.resolve()) specifically so a symlink inside the
    workspace can't be used to escape it.
    """
    p = Path(path).expanduser().resolve()
    if FS_WORKSPACE_ROOT:
        root = Path(FS_WORKSPACE_ROOT).expanduser().resolve()
        try:
            p.relative_to(root)
        except ValueError:
            raise PathNotAllowed(
                f"Path '{p}' is outside the configured workspace root '{root}'."
            )
    return p


async def fs_read(path: str, encoding: str = "utf-8") -> dict:
    """Read a file and return its text content."""
    try:
        p = _resolve_contained(path)
        if not p.exists():
            return {"ok": False, "error": f"File not found: {path}"}
        if not p.is_file():
            return {"ok": False, "error": f"Path is not a file: {path}"}
        content = p.read_text(encoding=encoding, errors="replace")
        return {"ok": True, "path": str(p), "content": content, "size_bytes": p.stat().st_size}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def fs_write(path: str, content: str, mode: str = "w") -> dict:
    """Write content to a file. mode='w' overwrites, mode='a' appends."""
    try:
        p = _resolve_contained(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode=mode, encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": str(p), "size_bytes": p.stat().st_size}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def fs_list(path: str, recursive: bool = False) -> dict:
    """List directory contents. Returns file names, sizes, types."""
    try:
        p = _resolve_contained(path)
        if not p.exists():
            return {"ok": False, "error": f"Path not found: {path}"}
        if not p.is_dir():
            return {"ok": False, "error": f"Path is not a directory: {path}"}

        entries = []
        if recursive:
            for item in sorted(p.rglob("*")):
                rel = item.relative_to(p)
                entries.append({
                    "name": str(rel),
                    "type": "file" if item.is_file() else "dir",
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                })
        else:
            for item in sorted(p.iterdir()):
                entries.append({
                    "name": item.name,
                    "type": "file" if item.is_file() else "dir",
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                })

        return {"ok": True, "path": str(p), "entries": entries, "count": len(entries)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def fs_delete(path: str, recursive: bool = False) -> dict:
    """Delete a file or directory."""
    try:
        p = _resolve_contained(path)
        if not p.exists():
            return {"ok": False, "error": f"Path not found: {path}"}
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            if recursive:
                shutil.rmtree(p)
            else:
                p.rmdir()  # fails if non-empty — intentional safety
        return {"ok": True, "path": str(p)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
