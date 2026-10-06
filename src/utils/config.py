"""Configuration loading and project path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Optional path to a config file. Defaults to ``config/config.yaml``.

    Returns:
        The configuration as a nested dictionary.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or malformed: {config_path}")
    return config


def resolve_path(relative: str | Path) -> Path:
    """Resolve a project-relative path (as stored in the config) to an absolute path."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path
