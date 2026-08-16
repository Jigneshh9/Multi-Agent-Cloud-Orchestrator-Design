"""Structured (JSON) logging configuration."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    try:
        from pythonjsonlogger import jsonlogger

        formatter = jsonlogger.JsonFormatter(  # type: ignore[attr-defined]
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    except ImportError:  # pragma: no cover - fallback for minimal installs
        formatter = logging.Formatter(  # type: ignore[assignment]
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
