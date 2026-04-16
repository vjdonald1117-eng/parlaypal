"""
Central logging configuration for Parlay Pal (console, structured format).

Call ``configure_app_logging()`` once at process entry (e.g. from ``api`` on import)
so all ``logging.getLogger(__name__)`` loggers share the same handlers and level.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_configured: bool = False


def configure_app_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger for stdout with a fixed format.

    Uses ``logging.basicConfig(..., force=True)`` so re-imports under uvicorn
    --reload still apply the intended format and level.
    """
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    _configured = True
