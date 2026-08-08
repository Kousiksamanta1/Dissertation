"""Configuration loading helpers for the SOC-ready IDS project."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    """Thin wrapper around the loaded YAML configuration."""

    values: dict[str, Any]
    root: Path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return a nested config value using dot notation."""
        current: Any = self.values
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def path(self, dotted_key: str) -> Path:
        """Return a project-relative path from the config."""
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"Missing path config value: {dotted_key}")
        dataset = str(self.get("data.dataset", "cicids2017"))
        path = Path(str(value).format(dataset=dataset))
        return path if path.is_absolute() else self.root / path

    def for_dataset(self, dataset: str) -> ProjectConfig:
        """Return an isolated configuration for one dataset."""
        values = copy.deepcopy(self.values)
        values.setdefault("data", {})["dataset"] = dataset
        return ProjectConfig(values=values, root=self.root)


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest ancestor containing config.yaml."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "config.yaml").exists():
            return candidate
    return current


def load_config(config_path: str | Path | None = None) -> ProjectConfig:
    """Load config.yaml and return a typed wrapper."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised before deps install
        raise RuntimeError(
            "PyYAML is required to load config.yaml. Run: pip install -r requirements.txt"
        ) from exc

    path = Path(config_path) if config_path else find_project_root() / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return ProjectConfig(values=values, root=path.parent.resolve())
