from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gguf_dream_lab.backend.dream.controller import DreamTick


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_ticks(self, run_id: str, ticks: list[DreamTick], metadata: dict | None = None) -> Path:
        path = self.root / f"{run_id}_ticks.parquet"
        rows = [t.__dict__ for t in ticks]
        pd.DataFrame(rows).to_parquet(path, index=False)
        if metadata is not None:
            (self.root / f"{run_id}_meta.json").write_text(json.dumps(metadata, indent=2))
        return path

    def load_ticks(self, run_id: str) -> pd.DataFrame:
        path = self.root / f"{run_id}_ticks.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
