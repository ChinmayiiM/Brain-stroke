"""Project-wide logging setup."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent console format (configured once per name)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def add_file_handler(logger_names: list[str], path) -> None:
    """Also write the given loggers to ``path`` (idempotent — safe on Streamlit reruns)."""
    from pathlib import Path

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    for name in logger_names:
        logger = get_logger(name)
        if any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == target for h in logger.handlers):
            continue
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
