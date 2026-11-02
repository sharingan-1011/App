"""
backend/logging_config.py — process-wide structured logging setup.

Previously there was no logging anywhere in backend/: errors were
swallowed into JSON ({"error": str(exc)}) and returned to the model,
with no corresponding server-side log line, and the scheduler loop
caught all per-tick exceptions with a bare `except Exception: pass`.
This module gives every other backend module a `logger` to import and
use, and configures a single root handler so output is consistent
whether it's a tool error, a scheduler failure, or an unhandled
request exception.

Deliberately minimal: stdlib `logging` to stdout, no external
dependency, no log-aggregation integration — this is the floor, not
the ceiling. FORGE_LOG_LEVEL controls verbosity.
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("FORGE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("forge")
    root.setLevel(level)
    root.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    ))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Returns a child of the 'forge' logger, e.g. get_logger(__name__)."""
    configure_logging()
    return logging.getLogger(f"forge.{name}")
