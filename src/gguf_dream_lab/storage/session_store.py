from __future__ import annotations

from pathlib import Path

import pandas as pd

from gguf_dream_lab.backend.dream.controller import DreamTick


class SessionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_ticks(self, run_id: str, ticks: list[DreamTick]) -> Path:
        path = self.root / f"{run_id}_ticks.parquet"
        rows = [t.__dict__ for t in ticks]
        pd.DataFrame(rows).to_parquet(path, index=False)
        return path
