"""Filesystem and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(payload: dict[str, Any] | list[Any], path: str | Path) -> Path:
    """Save JSON with readable indentation."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return output


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_joblib(payload: Any, path: str | Path) -> Path:
    """Save a Python artifact with joblib."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output)
    return output


def load_joblib(path: str | Path) -> Any:
    """Load a joblib artifact."""
    return joblib.load(path)
