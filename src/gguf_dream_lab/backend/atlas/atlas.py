from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from gguf_dream_lab.backend.dream.state import LatentState


@dataclass
class StatePoint:
    state_id: str
    run_id: str
    run_label: str
    basin: str
    step_idx: int
    preview: str
    committed: str
    coherence: float
    entropy: float
    embedding: np.ndarray
    phase: str = "HYPNAGOGIC"
    density: float = 0.0
    recurrence: int = 0


@dataclass
class TransitionEdge:
    src_state_id: str
    dst_state_id: str
    run_id: str
    step_idx: int
    distance: float
    curvature: float
    speed: float


@dataclass
class LatentAtlas:
    points: list[StatePoint] = field(default_factory=list)
    edges: list[TransitionEdge] = field(default_factory=list)
    projection_2d: np.ndarray | None = None
    nn: NearestNeighbors | None = None
    cluster_labels: np.ndarray | None = None
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def append_points(self, new_points: list[StatePoint]) -> None:
        with self._lock:
            self.points.extend(new_points)
            self._rebuild_indexes_if_needed()

    def append_transition(self, edge: TransitionEdge) -> None:
        with self._lock:
            self.edges.append(edge)

    def append_latent_state(self, state: LatentState, *, run_label: str, step_idx: int) -> StatePoint:
        point = StatePoint(
            state_id=state.state_id,
            run_id=state.run_id,
            run_label=run_label,
            basin=state.basin,
            step_idx=step_idx,
            preview=state.preview_text,
            committed=state.committed_prefix,
            coherence=state.coherence,
            entropy=state.entropy,
            embedding=np.asarray(state.latent_vector, dtype=np.float32),
            phase=state.phase.value,
            density=state.density,
        )
        self.append_points([point])
        return point

    def _rebuild_indexes_if_needed(self) -> None:
        if len(self.points) < 3 or len(self.points) % 8 == 0:
            self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        if not self.points:
            self.projection_2d = None
            self.nn = None
            self.cluster_labels = None
            return
        mat = np.stack([p.embedding for p in self.points], axis=0)
        if len(self.points) < 2:
            self.projection_2d = np.zeros((len(self.points), 2), dtype=float)
            self.nn = None
            self.cluster_labels = np.zeros((len(self.points),), dtype=int)
            return
        pca = PCA(n_components=2)
        self.projection_2d = pca.fit_transform(mat)
        self.nn = NearestNeighbors(n_neighbors=min(8, len(self.points))).fit(mat)
        n_clusters = min(max(2, len(self.points) // 20), 12)
        self.cluster_labels = MiniBatchKMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit_predict(mat)

    def neighbors(self, index: int, k: int = 6) -> list[int]:
        with self._lock:
            if self.nn is None or not self.points:
                return []
            mat = np.stack([p.embedding for p in self.points], axis=0)
            fitted_count = int(getattr(self.nn, "n_samples_fit_", 0) or 0)
            if fitted_count <= 0:
                return []
            query = mat[index].reshape(1, -1)
            neighbor_count = min(k, len(self.points), fitted_count)
            if neighbor_count <= 0:
                return []
            _, indices = self.nn.kneighbors(query, n_neighbors=neighbor_count)
            return [int(i) for i in indices[0].tolist() if i < len(self.points)]

    def local_density(self, index: int, k: int = 6) -> float:
        with self._lock:
            if self.nn is None or len(self.points) < 2:
                return 0.0
            mat = np.stack([p.embedding for p in self.points], axis=0)
            fitted_count = int(getattr(self.nn, "n_samples_fit_", 0) or 0)
            if fitted_count <= 0:
                return 0.0
            query = mat[index].reshape(1, -1)
            neighbor_count = min(k, len(self.points), fitted_count)
            if neighbor_count <= 0:
                return 0.0
            distances, _ = self.nn.kneighbors(query, n_neighbors=neighbor_count)
            avg_dist = float(np.mean(distances[0][1:])) if distances.shape[1] > 1 else float(np.mean(distances[0]))
            return 1.0 / (avg_dist + 1e-6)

    def sample_seed_from_basin(self, basin: str) -> np.ndarray | None:
        basin_points = [p for p in self.points if p.basin == basin]
        if not basin_points:
            return None
        chosen = basin_points[np.random.randint(0, len(basin_points))]
        return chosen.embedding

    def candidate_attractors(self, basin: str | None = None, top_k: int = 5) -> list[StatePoint]:
        candidates = [p for p in self.points if basin is None or p.basin == basin]
        candidates.sort(key=lambda p: (p.recurrence, p.coherence, -p.entropy), reverse=True)
        return candidates[:top_k]

    def to_frame(self) -> pd.DataFrame:
        with self._lock:
            rows = []
            for i, p in enumerate(self.points):
                if self.projection_2d is not None and i < len(self.projection_2d):
                    x, y = self.projection_2d[i]
                else:
                    x, y = (0.0, 0.0)
                cluster = int(self.cluster_labels[i]) if self.cluster_labels is not None and i < len(self.cluster_labels) else -1
                rows.append(
                    {
                        "state_id": p.state_id,
                        "run_id": p.run_id,
                        "run_label": p.run_label,
                        "basin": p.basin,
                        "step_idx": p.step_idx,
                        "preview": p.preview,
                        "committed": p.committed,
                        "phase": p.phase,
                        "coherence": p.coherence,
                        "entropy": p.entropy,
                        "density": p.density,
                        "x": float(x),
                        "y": float(y),
                        "cluster": cluster,
                    }
                )
            return pd.DataFrame(rows)


class AtlasStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def parquet_path(self) -> Path:
        return self.root / "atlas_points.parquet"

    @property
    def embeddings_path(self) -> Path:
        return self.root / "atlas_embeddings.joblib"

    @property
    def edges_path(self) -> Path:
        return self.root / "atlas_edges.parquet"

    def save(self, atlas: LatentAtlas) -> None:
        df = atlas.to_frame()
        df.to_parquet(self.parquet_path, index=False)
        embeddings = np.stack([p.embedding for p in atlas.points], axis=0) if atlas.points else np.empty((0, 0))
        joblib.dump(embeddings, self.embeddings_path)
        edge_df = pd.DataFrame([e.__dict__ for e in atlas.edges])
        edge_df.to_parquet(self.edges_path, index=False)

    def load(self) -> LatentAtlas:
        atlas = LatentAtlas()
        if not self.parquet_path.exists() or not self.embeddings_path.exists():
            return atlas
        df = pd.read_parquet(self.parquet_path)
        embeddings = joblib.load(self.embeddings_path)
        points = []
        for i, row in df.iterrows():
            points.append(
                StatePoint(
                    state_id=row["state_id"],
                    run_id=row["run_id"],
                    run_label=row["run_label"],
                    basin=row["basin"],
                    step_idx=int(row["step_idx"]),
                    preview=row["preview"],
                    committed=row["committed"],
                    phase=row.get("phase", "HYPNAGOGIC"),
                    coherence=float(row["coherence"]),
                    entropy=float(row["entropy"]),
                    density=float(row.get("density", 0.0)),
                    embedding=np.array(embeddings[i], dtype=np.float32),
                )
            )
        atlas.points = points
        if self.edges_path.exists():
            e_df = pd.read_parquet(self.edges_path)
            atlas.edges = [TransitionEdge(**r) for r in e_df.to_dict(orient="records")]
        atlas._rebuild_indexes()
        return atlas
