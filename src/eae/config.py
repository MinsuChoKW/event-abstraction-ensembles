# src/eae/config.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML config file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return cfg


def get_project_root(cfg: Dict[str, Any]) -> Path:
    """
    Return project root directory from config.
    """
    return Path(cfg.get("project", {}).get("root_dir", "."))


def resolve_path(cfg: Dict[str, Any], path: str | Path) -> Path:
    """
    Resolve a path relative to project.root_dir unless it is absolute.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return get_project_root(cfg) / path
