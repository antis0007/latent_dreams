from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
import hashlib

import numpy as np

from gguf_dream_lab.backend.atlas.atlas import LatentAtlas, TransitionEdge
from gguf_dream_lab.backend.dream.coherence import score_coherence
from gguf_dream_lab.backend.dream.state import DreamMode, DreamPhase, LatentState
from gguf_dream_lab.backend.runtime.base import RuntimeBackend
from gguf_dream_lab.config.models import Basin, CoherenceWeights, DreamConfig

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
    latent_source: str
    branch_id: str
    branch_seed: int
    branch_score: float
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
            candidates = self._branch_candidates(latent, cfg, anneal, phase, step_idx=self.state.step_idx)
            latent = self._choose_candidate(candidates, prev_latent=latent)
            preview = self._decode_preview(latent, max_tokens=max(12, cfg.preview_decode_cadence * 16))
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
                committed_chunk = self.runtime.decode_commit_from_latent(latent, max_tokens=24)
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
                        branch_id=str(latent.metadata.get("branch_id", "")),
                        branch_seed=int(latent.metadata.get("branch_seed", 0)),
                        branch_score=float(latent.metadata.get("branch_score", 0.0)),
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
                latent_source=latent.latent_source.value,
                branch_id=str(latent.metadata.get("branch_id", "")),
                branch_seed=int(latent.metadata.get("branch_seed", 0)),
                branch_score=float(latent.metadata.get("branch_score", 0.0)),
                status=self.state.status,
            )
            self.state.latest_tick = tick
            self.tick_history.append(tick)
            self.state.step_idx += 1
            anneal *= cfg.anneal_rate
            elapsed = time.perf_counter() - tick_start
            time.sleep(max(period - elapsed, 0.0))

        self.state.active = False

    def _decode_preview(self, latent: LatentState, max_tokens: int) -> str:
        caps = self.runtime.capabilities()
        if caps.supports_instrumented_latents and caps.supports_true_latent_readout:
            return self.runtime.decode_true_latent_readout_preview(latent, max_tokens=max_tokens)
        return self.runtime.decode_prompt_conditioned_preview_from_latent(latent, max_tokens=max_tokens)

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

    def _branch_candidates(
        self,
        latent: LatentState,
        cfg: DreamConfig,
        anneal: float,
        phase: DreamPhase,
        *,
        step_idx: int,
    ) -> list[LatentState]:
        neighbors = self._neighbor_vector(latent.latent_vector)
        candidates = []
        branch_count = max(cfg.branch_count, 1)
        for branch_idx in range(branch_count):
            branch_id = f"{step_idx:06d}-b{branch_idx:02d}"
            branch_seed = self._branch_seed(self.state.run_id, step_idx, branch_idx)
            rng = np.random.default_rng(branch_seed)
            target = 0.7 * neighbors + 0.3 * latent.latent_vector
            perturb = rng.normal(scale=cfg.noise_amplitude * anneal * 0.5, size=target.shape).astype(np.float32)
            proposal = self.runtime.evolve_latent_state(
                latent,
                target_vector=target + perturb,
                noise_scale=cfg.noise_amplitude * anneal,
                noise_seed=branch_seed,
            )
            proposal = self._normalize_latent_state(proposal)
            proposal.phase = phase
            branch_agreement = 1.0 if branch_count == 1 else 1.0 - (cfg.noise_amplitude * anneal * 0.25)
            branch_smoothness = self._smoothness(latent.latent_vector, proposal.latent_vector)
            branch_density = self._estimate_local_density(proposal.latent_vector)
            branch_coherence_estimate = self._branch_coherence_estimate(
                local_density=branch_density,
                smoothness=branch_smoothness,
                branch_agreement=branch_agreement,
                weights=cfg.weights,
            )
            distance_to_attractor = self._distance_to_attractor(proposal.latent_vector, basin=proposal.basin)
            branch_score = self._branch_score(
                coherence_estimate=branch_coherence_estimate,
                distance_to_attractor=distance_to_attractor,
                temporal_smoothness=branch_smoothness,
            )
            proposal.metadata["branch_agreement"] = branch_agreement
            proposal.metadata["branch_id"] = branch_id
            proposal.metadata["branch_seed"] = branch_seed
            proposal.metadata["branch_density_estimate"] = branch_density
            proposal.metadata["branch_coherence_estimate"] = branch_coherence_estimate
            proposal.metadata["distance_to_attractor"] = distance_to_attractor
            proposal.metadata["temporal_smoothness"] = branch_smoothness
            proposal.metadata["branch_score"] = branch_score
            candidates.append(proposal)
        return candidates

    def _choose_candidate(self, candidates: list[LatentState], *, prev_latent: LatentState) -> LatentState:
        if len(candidates) == 1:
            return candidates[0]
        scored = sorted(
            enumerate(candidates),
            key=lambda item: float(item[1].metadata.get("branch_score", float("-inf"))),
            reverse=True,
        )
        for rank, (_, candidate) in enumerate(scored, start=1):
            candidate.metadata["branch_rank"] = rank
        selected = scored[0][1]
        selected.metadata["selected_from_state_id"] = prev_latent.state_id
        return selected

    def _distance_to_attractor(self, vec: np.ndarray, *, basin: str) -> float:
        attractors = self.atlas.candidate_attractors(basin=basin, top_k=5)
        if not attractors:
            return 0.0
        dists = [float(np.linalg.norm(self._align_vector_shape(p.embedding, vec) - vec)) for p in attractors]
        return min(dists) if dists else 0.0

    def _branch_coherence_estimate(
        self,
        *,
        local_density: float,
        smoothness: float,
        branch_agreement: float,
        weights: CoherenceWeights,
    ) -> float:
        entropy = max(0.01, 1.0 - min(local_density / 4.0, 0.9))
        return score_coherence(
            entropy=entropy,
            local_density=local_density,
            token_stability=0.5,
            smoothness=smoothness,
            branch_agreement=branch_agreement,
            known_state_similarity=min(local_density / 5.0, 1.0),
            weights=weights,
        )

    @staticmethod
    def _branch_score(*, coherence_estimate: float, distance_to_attractor: float, temporal_smoothness: float) -> float:
        distance_penalty = min(distance_to_attractor / 3.0, 1.0)
        score = (0.65 * coherence_estimate) + (0.25 * temporal_smoothness) - (0.35 * distance_penalty)
        return float(score)

    @staticmethod
    def _branch_seed(run_id: str, step_idx: int, branch_idx: int) -> int:
        payload = f"{run_id}:{step_idx}:{branch_idx}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

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
        token_sets = [set(h.lower().split()) for h in history if h.strip()]
        if len(token_sets) < 2:
            return 0.0
        overlaps: list[float] = []
        for i in range(1, len(token_sets)):
            a = token_sets[i - 1]
            b = token_sets[i]
            union = len(a | b)
            if union == 0:
                continue
            overlaps.append(len(a & b) / union)
        if not overlaps:
            return 0.0
        return float(sum(overlaps) / len(overlaps))

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
