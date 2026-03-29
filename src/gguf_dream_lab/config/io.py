from __future__ import annotations

import json
from pathlib import Path

from .models import AppConfig


def load_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)


def save_config(config: AppConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
