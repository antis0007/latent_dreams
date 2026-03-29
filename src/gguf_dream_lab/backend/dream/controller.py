from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from gguf_dream_lab.backend.atlas.atlas import LatentAtlas, TransitionEdge
from gguf_dream_lab.backend.dream.coherence import score_coherence
from gguf_dream_lab.backend.dream.state import DreamMode, DreamPhase, LatentState
from gguf_dream_lab.backend.runtime.base import RuntimeBackend
from gguf_dream_lab.config.models import Basin, DreamConfig

SEED_BASIN_PREFIX = {
    Basin.NULL_PRIOR: "",
    Basin.INTROSPECTIVE: "I am inside a half-remembered thought where",
    Basin.NARRATIVE: "In a dimly lit scene,",
    Basin.MEMORY: "I remember something that almost happened:",
    Basin.AFFECTIVE: "A feeling arrives first, then",
    Basin.PROMPT_CONDITIONED: "",
}


@dataclass
class DreamTick:
    run_id: str
    step_idx: int
    preview_text: str
    committed_text: str
    coherence: float
    entropy: float
    local_density: float
    token_stability: float
    smoothness: float
    state_id: str
    phase: str
    mode: str
    status: str


@dataclass
class DreamSessionState:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = False
    paused: bool = False
    step_idx: int = 0
    preview_text: str = ""
    committed_text: str = ""
    status: str = "idle"
    status_detail: str = ""
    latest_tick: DreamTick | None = None


class DreamController:
    def __init__(self, runtime: RuntimeBackend, atlas: LatentAtlas):
        self.runtime = runtime
        self.atlas = atlas
        self.state = DreamSessionState()
        self.tick_history: list[DreamTick] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def start(self, cfg: DreamConfig) -> DreamSessionState:
        if self.state.active:
            return self.state
        self.state = DreamSessionState(active=True, paused=False, status="loading_model", status_detail="Loading model...")
        self.tick_history = []
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._loop, args=(cfg,), daemon=True)
        self._thread.start()
        return self.state

    def pause(self) -> None:
        if self.state.active:
            self.state.paused = True
            self._pause_event.set()
            self.state.status = "paused"

    def resume(self) -> None:
        if self.state.active:
            self.state.paused = False
            self._pause_event.clear()
            self.state.status = "running"

    def stop(self) -> None:
        self._stop_event.set()
        self.state.active = False
        self.state.status = "stopped"

    def _loop(self, cfg: DreamConfig) -> None:
        try:
            self.runtime.load()
        except Exception as exc:
            self.state.active = False
            self.state.status = "error"
            self.state.status_detail = f"Model load failed: {exc}"
            return

        self.state.status = "running"
        period = 1.0 / max(cfg.tick_hz, 0.1)
        preview_history: deque[str] = deque(maxlen=max(cfg.stability_window, 2))
        basin_prefix = SEED_BASIN_PREFIX[cfg.basin]
        prompt = (basin_prefix + " " + cfg.prompt).strip()
        latent = self._seed_latent_state(cfg, prompt)
        prev_vec = latent.latent_vector.copy()
        anneal = 1.0

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            tick_start = time.perf_counter()
            phase = self._phase_for_step(self.state.step_idx)
            candidates = self._branch_candidates(latent, cfg, anneal, phase)
            latent = self._choose_candidate(candidates)
            preview = self.runtime.decode_preview_from_latent(latent, max_tokens=cfg.preview_decode_cadence * 4)
            self.state.preview_text = preview[-cfg.max_preview_len :]
            preview_history.append(self.state.preview_text)
            token_stability = self._token_stability(preview_history)

            local_density = self._estimate_local_density(latent.latent_vector)
            smoothness = self._smoothness(prev_vec, latent.latent_vector)
            prev_vec = latent.latent_vector.copy()
            latent.density = local_density
            latent.entropy = max(0.01, 1.0 - min(local_density / 4.0, 0.9))
            latent.coherence = score_coherence(
                entropy=latent.entropy,
                local_density=local_density,
                token_stability=token_stability,
                smoothness=smoothness,
                branch_agreement=latent.metadata.get("branch_agreement", 1.0),
                known_state_similarity=min(local_density / 5.0, 1.0),
                weights=cfg.weights,
            )
            latent.preview_text = self.state.preview_text
            latent.committed_prefix = self.state.committed_text

            if self._should_commit(latent.coherence, preview_history, cfg):
                committed_chunk = self.runtime.decode_commit_from_latent(latent, max_tokens=12)
                self.state.committed_text = self._merge_committed_chunk(self.state.committed_text, committed_chunk, cfg.max_committed_len)
                latent.committed_prefix = self.state.committed_text

            point = self.atlas.append_latent_state(latent, run_label=cfg.run_label, step_idx=self.state.step_idx)
            if len(self.atlas.points) > 1:
                prev = self.atlas.points[-2]
                dist = float(np.linalg.norm(point.embedding - prev.embedding))
                self.atlas.append_transition(
                    TransitionEdge(
                        src_state_id=prev.state_id,
                        dst_state_id=point.state_id,
                        run_id=self.state.run_id,
                        step_idx=self.state.step_idx,
                        distance=dist,
                        curvature=abs(1.0 - smoothness),
                        speed=dist * cfg.tick_hz,
                    )
                )

            tick = DreamTick(
                run_id=self.state.run_id,
                step_idx=self.state.step_idx,
                preview_text=self.state.preview_text,
                committed_text=self.state.committed_text,
                coherence=latent.coherence,
                entropy=latent.entropy,
                local_density=local_density,
                token_stability=token_stability,
                smoothness=smoothness,
                state_id=latent.state_id,
                phase=phase.value,
                mode=latent.mode.value,
                status=self.state.status,
            )
            self.state.latest_tick = tick
            self.tick_history.append(tick)
            self.state.step_idx += 1
            anneal *= cfg.anneal_rate
            elapsed = time.perf_counter() - tick_start
            time.sleep(max(period - elapsed, 0.0))

        self.state.active = False

    def _seed_latent_state(self, cfg: DreamConfig, prompt: str) -> LatentState:
        basin_vec = self.atlas.sample_seed_from_basin(cfg.basin.value)
        latent = self._normalize_latent_state(self.runtime.capture_latent_state(self.state.run_id, cfg.basin.value, prompt))
        if basin_vec is not None:
            basin_vec = np.asarray(basin_vec, dtype=np.float32).reshape(-1)
            basin_vec = self._align_vector_shape(basin_vec, latent.latent_vector)
            blended = 0.65 * basin_vec + 0.35 * latent.latent_vector
            latent = latent.clone_with_vector(blended, phase=DreamPhase.HYPNAGOGIC)
        latent.metadata["prompt_seed"] = prompt
        return latent

    def _branch_candidates(self, latent: LatentState, cfg: DreamConfig, anneal: float, phase: DreamPhase) -> list[LatentState]:
        neighbors = self._neighbor_vector(latent.latent_vector)
        candidates = []
        for _ in range(max(cfg.branch_count, 1)):
            target = 0.7 * neighbors + 0.3 * latent.latent_vector
            proposal = self.runtime.evolve_latent_state(latent, target_vector=target, noise_scale=cfg.noise_amplitude * anneal)
            proposal = self._normalize_latent_state(proposal)
            proposal.phase = phase
            proposal.metadata["branch_agreement"] = 1.0 if cfg.branch_count == 1 else 1.0 - (cfg.noise_amplitude * anneal * 0.25)
            candidates.append(proposal)
        return candidates

    def _choose_candidate(self, candidates: list[LatentState]) -> LatentState:
        if len(candidates) == 1:
            return candidates[0]
        scores = [float(np.linalg.norm(c.latent_vector)) for c in candidates]
        return candidates[int(np.argmax(scores))]

    def _neighbor_vector(self, current: np.ndarray) -> np.ndarray:
        if not self.atlas.points:
            return current
        idx = len(self.atlas.points) - 1
        nn_idx = self.atlas.neighbors(idx, k=4)
        nn_idx = [i for i in nn_idx if i != idx]
        if not nn_idx:
            return current
        vecs = [self._align_vector_shape(self.atlas.points[i].embedding, current) for i in nn_idx]
        return np.mean(np.stack(vecs, axis=0), axis=0)

    def _estimate_local_density(self, vec: np.ndarray) -> float:
        if not self.atlas.points:
            return 0.0
        mat = np.stack([p.embedding for p in self.atlas.points], axis=0)
        if mat.shape[1] != vec.shape[0]:
            return 0.0
        dists = np.linalg.norm(mat - vec.reshape(1, -1), axis=1)
        top = np.sort(dists)[: min(8, len(dists))]
        return float(1.0 / (np.mean(top) + 1e-6))

    @staticmethod
    def _smoothness(prev: np.ndarray, cur: np.ndarray) -> float:
        prev = np.asarray(prev, dtype=np.float32).reshape(-1)
        cur = np.asarray(cur, dtype=np.float32).reshape(-1)
        prev = DreamController._align_vector_shape(prev, cur)
        cos = float(np.dot(cur, prev) / (np.linalg.norm(cur) * np.linalg.norm(prev) + 1e-9))
        return float((cos + 1.0) / 2.0)

    @staticmethod
    def _align_vector_shape(vec: np.ndarray, ref: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        ref = np.asarray(ref, dtype=np.float32).reshape(-1)
        if vec.shape == ref.shape:
            return vec
        aligned = np.zeros_like(ref)
        upto = min(vec.shape[0], ref.shape[0])
        aligned[:upto] = vec[:upto]
        return aligned

    @staticmethod
    def _normalize_latent_state(state: LatentState) -> LatentState:
        vec = np.asarray(state.latent_vector, dtype=np.float32)
        if vec.ndim == 1:
            return state
        return state.clone_with_vector(vec.reshape(-1))

    @staticmethod
    def _phase_for_step(step_idx: int) -> DreamPhase:
        stage = step_idx % 24
        if stage < 6:
            return DreamPhase.HYPNAGOGIC
        if stage < 13:
            return DreamPhase.SCENE_FORMATION
        if stage < 20:
            return DreamPhase.CONSOLIDATION
        return DreamPhase.DRIFT_RESET

    @staticmethod
    def _token_stability(history: deque[str]) -> float:
        if len(history) < 2:
            return 0.0
        tails = [" ".join(h.split()[-3:]) for h in history]
        counts = {w: tails.count(w) for w in set(tails)}
        return max(counts.values()) / len(tails)

    @staticmethod
    def _should_commit(coherence: float, history: deque[str], cfg: DreamConfig) -> bool:
        if coherence < cfg.coherence_threshold or len(history) < cfg.stability_window:
            return False
        last = list(history)[-cfg.stability_window :]
        unique = len(set(last))
        return unique <= max(2, cfg.stability_window // 2)

    @staticmethod
    def _merge_committed_chunk(current_committed: str, chunk: str, max_len: int) -> str:
        chunk = chunk.strip()
        if not chunk:
            return current_committed
        updated = (current_committed + " " + chunk).strip()
        return updated[-max_len:]
