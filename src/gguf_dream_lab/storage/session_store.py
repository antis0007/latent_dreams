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

    def load_timeline(self, run_id: str) -> dict:
        frame = self.load_ticks(run_id)
        if frame.empty:
            return {"run_id": run_id, "ticks": [], "roots": [], "children": {}, "paths": []}

        frame = frame.sort_values(["step_idx", "state_id"]).reset_index(drop=True)
        ticks = frame.to_dict("records")
        nodes = {str(t["state_id"]): t for t in ticks if t.get("state_id")}
        children: dict[str, list[str]] = {}
        roots: list[str] = []

        for tick in ticks:
            state_id = str(tick.get("state_id") or "")
            parent_state_id = str(tick.get("parent_state_id") or "")
            if not state_id:
                continue
            if parent_state_id and parent_state_id in nodes:
                children.setdefault(parent_state_id, []).append(state_id)
            else:
                roots.append(state_id)

        ordered_roots: list[str] = []
        for state_id in roots:
            if state_id not in ordered_roots:
                ordered_roots.append(state_id)

        paths: list[dict] = []

        def dfs(path_so_far: list[str]) -> None:
            cur = path_so_far[-1]
            next_nodes = children.get(cur, [])
            if not next_nodes:
                path_ticks = [nodes[state] for state in path_so_far if state in nodes]
                if path_ticks:
                    first_tick = path_ticks[0]
                    last_tick = path_ticks[-1]
                    path_id = f"{first_tick.get('branch_id', 'root')}::{last_tick.get('state_id', cur)}"
                    paths.append(
                        {
                            "id": path_id,
                            "label": (
                                f"step {int(first_tick.get('step_idx', 0))}"
                                f"→{int(last_tick.get('step_idx', 0))}"
                                f" · {str(last_tick.get('branch_id', 'branch'))}"
                            ),
                            "state_ids": list(path_so_far),
                            "start_step": int(first_tick.get("step_idx", 0)),
                            "end_step": int(last_tick.get("step_idx", 0)),
                        }
                    )
                return
            for child_state in next_nodes:
                dfs(path_so_far + [child_state])

        for root_state in ordered_roots:
            dfs([root_state])

        return {"run_id": run_id, "ticks": ticks, "roots": ordered_roots, "children": children, "paths": paths}
