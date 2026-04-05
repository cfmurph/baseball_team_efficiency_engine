from __future__ import annotations

from pathlib import Path
import yaml


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
