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
    latent_source: str = "embedding_proxy"
    density: float = 0.0
    recurrence: int = 0
    attractor_id: str = ""
    visit_count: int = 1
    dwell_time: float = 0.0
    return_count: int = 0
    return_frequency: float = 0.0
    attractor_strength: float = 0.0


@dataclass
class AttractorStats:
    visit_count: int = 0
    dwell_time: float = 0.0
    return_count: int = 0
    return_frequency: float = 0.0
    last_step_idx: int | None = None
    last_timestamp: float | None = None


@dataclass
class TransitionEdge:
    src_state_id: str
    dst_state_id: str
    run_id: str
    step_idx: int
    distance: float
    curvature: float
    speed: float
    branch_id: str = ""
    branch_seed: int = 0
    branch_score: float = 0.0


@dataclass
class LatentAtlas:
    points: list[StatePoint] = field(default_factory=list)
    edges: list[TransitionEdge] = field(default_factory=list)
    projection_2d: np.ndarray | None = None
    nn: NearestNeighbors | None = None
    cluster_labels: np.ndarray | None = None
    recurrence_distance_threshold: float = 0.75
    _attractor_stats: dict[str, AttractorStats] = field(default_factory=dict, init=False, repr=False, compare=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def append_points(self, new_points: list[StatePoint]) -> None:
        with self._lock:
            self.points.extend(new_points)
            self._rebuild_attractor_stats()
            self._rebuild_indexes_if_needed()

    def clear(self) -> None:
        with self._lock:
            self.points = []
            self.edges = []
            self._attractor_stats = {}
            self._rebuild_indexes()

    def remove_runs(self, run_ids: set[str]) -> None:
        if not run_ids:
            return
        with self._lock:
            self.points = [p for p in self.points if p.run_id not in run_ids]
            self.edges = [e for e in self.edges if e.run_id not in run_ids]
            self._rebuild_attractor_stats()
            self._rebuild_indexes()

    def append_transition(self, edge: TransitionEdge) -> None:
        with self._lock:
            self.edges.append(edge)

    def append_latent_state(self, state: LatentState, *, run_label: str, step_idx: int) -> StatePoint:
        with self._lock:
            embedding = np.asarray(state.latent_vector, dtype=np.float32)
            matched_idx = self._nearest_prior_index(embedding, state.basin)
            attractor_id = state.state_id
            recurrence = 0
            if matched_idx is not None:
                matched = self.points[matched_idx]
                matched.recurrence += 1
                recurrence = matched.recurrence
                attractor_id = matched.attractor_id or matched.state_id
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
                embedding=embedding,
                phase=state.phase.value,
                latent_source=state.latent_source.value,
                density=state.density,
                recurrence=recurrence,
                attractor_id=attractor_id,
            )
            self._update_attractor_metrics(point, timestamp=state.timestamp)
            self.points.append(point)
            self._rebuild_indexes_if_needed()
            return point

    def _nearest_prior_index(self, embedding: np.ndarray, basin: str) -> int | None:
        if not self.points:
            return None
        candidate_indices = [i for i, p in enumerate(self.points) if p.basin == basin]
        if not candidate_indices:
            return None
        mat = np.stack([self.points[i].embedding for i in candidate_indices], axis=0)
        if mat.shape[1] != embedding.shape[0]:
            return None
        dists = np.linalg.norm(mat - embedding.reshape(1, -1), axis=1)
        nearest_local_idx = int(np.argmin(dists))
        if float(dists[nearest_local_idx]) > self.recurrence_distance_threshold:
            return None
        return candidate_indices[nearest_local_idx]

    def _strength_score(self, stats: AttractorStats) -> float:
        return float(stats.visit_count + (0.25 * stats.dwell_time) + (2.0 * stats.return_frequency))

    def _update_attractor_metrics(self, point: StatePoint, *, timestamp: float) -> None:
        attractor_id = point.attractor_id or point.state_id
        stats = self._attractor_stats.setdefault(attractor_id, AttractorStats())
        stats.visit_count += 1
        if stats.last_step_idx is not None:
            gap = point.step_idx - stats.last_step_idx
            if gap == 1 and stats.last_timestamp is not None:
                stats.dwell_time += max(0.0, float(timestamp - stats.last_timestamp))
            elif gap > 1:
                stats.return_count += 1
        denom = max(stats.visit_count - 1, 1)
        return_frequency = stats.return_count / denom
        stats.last_step_idx = point.step_idx
        stats.last_timestamp = timestamp
        stats.return_frequency = return_frequency

        point.visit_count = stats.visit_count
        point.dwell_time = stats.dwell_time
        point.return_count = stats.return_count
        point.return_frequency = return_frequency
        point.attractor_strength = self._strength_score(stats)
        point.attractor_id = attractor_id

        for existing in self.points:
            if (existing.attractor_id or existing.state_id) == attractor_id:
                existing.visit_count = stats.visit_count
                existing.dwell_time = stats.dwell_time
                existing.return_count = stats.return_count
                existing.return_frequency = return_frequency
                existing.attractor_strength = point.attractor_strength

    def _rebuild_attractor_stats(self) -> None:
        self._attractor_stats = {}
        if not self.points:
            return
        points = sorted(self.points, key=lambda p: (p.step_idx, p.state_id))
        for point in points:
            stats = self._attractor_stats.setdefault(point.attractor_id or point.state_id, AttractorStats())
            stats.visit_count += 1
            if stats.last_step_idx is not None:
                gap = point.step_idx - stats.last_step_idx
                if gap > 1:
                    stats.return_count += 1
            stats.last_step_idx = point.step_idx
            stats.last_timestamp = None
            denom = max(stats.visit_count - 1, 1)
            return_frequency = stats.return_count / denom
            stats.return_frequency = return_frequency
            point.visit_count = stats.visit_count
            point.return_count = stats.return_count
            point.return_frequency = return_frequency
            point.attractor_strength = self._strength_score(stats)

    def _rebuild_indexes_if_needed(self) -> None:
        # Keep projection and neighbor index in sync on every append so the UI
        # never renders newly added points as (0, 0) placeholders between
        # periodic rebuild windows.
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
            return [int(i) for i in indices[0].tolist() if i < len(self.points) and int(i) != index]

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
        best_by_attractor: dict[str, StatePoint] = {}
        for point in candidates:
            key = point.attractor_id or point.state_id
            current = best_by_attractor.get(key)
            if current is None or point.attractor_strength > current.attractor_strength:
                best_by_attractor[key] = point
        ranked = list(best_by_attractor.values())
        ranked.sort(
            key=lambda p: (p.attractor_strength, p.visit_count, p.return_frequency, p.recurrence, p.coherence, -p.entropy),
            reverse=True,
        )
        return ranked[:top_k]

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
                        "latent_source": p.latent_source,
                        "coherence": p.coherence,
                        "entropy": p.entropy,
                        "density": p.density,
                        "recurrence": p.recurrence,
                        "attractor_id": p.attractor_id or p.state_id,
                        "visit_count": p.visit_count,
                        "dwell_time": p.dwell_time,
                        "return_count": p.return_count,
                        "return_frequency": p.return_frequency,
                        "attractor_strength": p.attractor_strength,
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
        for i, (_, row) in enumerate(df.iterrows()):
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
                    latent_source=row.get("latent_source", "embedding_proxy"),
                    coherence=float(row["coherence"]),
                    entropy=float(row["entropy"]),
                    density=float(row.get("density", 0.0)),
                    recurrence=int(row.get("recurrence", 0)),
                    attractor_id=row.get("attractor_id", row["state_id"]),
                    visit_count=int(row.get("visit_count", 1)),
                    dwell_time=float(row.get("dwell_time", 0.0)),
                    return_count=int(row.get("return_count", 0)),
                    return_frequency=float(row.get("return_frequency", 0.0)),
                    attractor_strength=float(row.get("attractor_strength", 0.0)),
                    embedding=np.array(embeddings[i], dtype=np.float32),
                )
            )
        atlas.points = points
        if self.edges_path.exists():
            e_df = pd.read_parquet(self.edges_path)
            atlas.edges = [TransitionEdge(**r) for r in e_df.to_dict(orient="records")]
        atlas._rebuild_attractor_stats()
        atlas._rebuild_indexes()
        return atlas
