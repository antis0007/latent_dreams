from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from gguf_dream_lab.backend.atlas.atlas import LatentAtlas, TransitionEdge
from gguf_dream_lab.backend.dream.coherence import score_coherence, score_coherence_breakdown
from gguf_dream_lab.backend.dream.state import DreamMode, DreamPhase, LatentState
from gguf_dream_lab.backend.runtime.base import RuntimeBackend
from gguf_dream_lab.config.models import Basin, BranchSelectionPolicy, CoherenceWeights, DreamConfig

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
    parent_state_id: str
    phase: str
    mode: str
    commit_source: str
    latent_source: str
    branch_id: str
    branch_seed: int
    branch_score: float
    candidate_scores: str
    rejected_candidates: str
    selected_branch_trace: str
    basin_sample_count: int
    basin_mean_coherence: float
    basin_mean_density: float
    basin_force_magnitude: float
    basin_prior_spread: float
    navigation_decision: str
    penalty_breakdown: str
    decode_provenance: str
    coherence_component_entropy: float
    coherence_component_density: float
    coherence_component_token_stability: float
    coherence_component_smoothness: float
    coherence_component_branch_agreement: float
    coherence_component_known_state_similarity: float
    adaptive_noise: float
    adaptive_attractor_weight: float
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
        self._branch_policy_temperature = float(cfg.branch_policy_temperature)
        self._branch_policy_exploration = float(cfg.branch_policy_exploration)
        policy_seed = self._branch_seed(self.state.run_id, 0, 10_001)
        self._policy_random = np.random.default_rng(policy_seed)
        basin_prefix = SEED_BASIN_PREFIX[cfg.basin]
        prompt = (basin_prefix + " " + cfg.prompt).strip()
        latent = self._seed_latent_state(cfg, prompt)
        prev_vec = latent.latent_vector.copy()
        self._previous_direction = np.zeros_like(prev_vec)
        anneal = 1.0
        coherence_window: deque[float] = deque(maxlen=8)

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            tick_start = time.perf_counter()
            phase = self._phase_for_step(self.state.step_idx)
            recent_coherence = float(np.mean(coherence_window)) if coherence_window else latent.coherence
            adaptive_noise = self._adaptive_noise(cfg, anneal=anneal, recent_coherence=recent_coherence)
            adaptive_attractor = self._adaptive_attractor_weight(cfg, recent_coherence=recent_coherence)
            candidates = self._branch_candidates(
                latent,
                cfg,
                anneal,
                phase,
                step_idx=self.state.step_idx,
                adaptive_noise=adaptive_noise,
                adaptive_attractor_weight=adaptive_attractor,
            )
            latent.metadata["branch_selection_policy"] = cfg.branch_selection_policy.value
            latent = self._choose_candidate(candidates, prev_latent=latent)
            candidate_scores = str(latent.metadata.get("candidate_scores", "[]"))
            should_decode_preview = (
                self.state.step_idx == 0 or (self.state.step_idx % max(1, int(cfg.preview_decode_cadence)) == 0)
            )
            if should_decode_preview:
                preview = self._decode_preview(latent, max_tokens=max(12, cfg.preview_decode_cadence * 16))
                preview_decode_source = self._preview_decode_source()
                self.state.preview_text = preview.strip()
            else:
                preview_decode_source = "preview_cache_hold"
            preview_history.append(self.state.preview_text)
            token_stability = self._token_stability(preview_history)

            local_density = self._estimate_local_density(latent.latent_vector)
            smoothness = self._smoothness(prev_vec, latent.latent_vector)
            prev_vec = latent.latent_vector.copy()
            latent.density = local_density
            latent.entropy = max(0.01, 1.0 - min(local_density / 4.0, 0.9))
            coherence_breakdown = score_coherence_breakdown(
                entropy=latent.entropy,
                local_density=local_density,
                token_stability=token_stability,
                smoothness=smoothness,
                branch_agreement=latent.metadata.get("branch_agreement", 1.0),
                known_state_similarity=min(local_density / 5.0, 1.0),
                weights=cfg.weights,
            )
            latent.coherence = coherence_breakdown.score
            latent.metadata["coherence_components"] = coherence_breakdown.components
            latent.metadata["coherence_weighted_contributions"] = coherence_breakdown.weighted_contributions
            latent.metadata["adaptive_noise"] = adaptive_noise
            latent.metadata["adaptive_attractor_weight"] = adaptive_attractor
            latent.preview_text = self.state.preview_text
            latent.committed_prefix = self.state.committed_text

            if self._should_commit(latent.coherence, preview_history, cfg):
                committed_chunk = self.runtime.decode_commit_from_latent(latent, max_tokens=24)
                self.state.committed_text = self._merge_committed_chunk(self.state.committed_text, committed_chunk, cfg.max_committed_len)
                latent.committed_prefix = self.state.committed_text
                commit_decode_source = str(latent.metadata.get("commit_source", "decode_commit_from_latent"))
            else:
                commit_decode_source = "none"

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
                parent_state_id=str(latent.metadata.get("selected_from_state_id", "")),
                phase=phase.value,
                mode=latent.mode.value,
                commit_source=str(latent.metadata.get("commit_source", "none")),
                latent_source=latent.latent_source.value,
                branch_id=str(latent.metadata.get("branch_id", "")),
                branch_seed=int(latent.metadata.get("branch_seed", 0)),
                branch_score=float(latent.metadata.get("branch_score", 0.0)),
                candidate_scores=candidate_scores,
                rejected_candidates=str(latent.metadata.get("rejected_candidates", "[]")),
                selected_branch_trace=str(latent.metadata.get("selected_branch_trace", "{}")),
                basin_sample_count=int(latent.metadata.get("basin_sample_count", 0)),
                basin_mean_coherence=float(latent.metadata.get("basin_mean_coherence", 0.0)),
                basin_mean_density=float(latent.metadata.get("basin_mean_density", 0.0)),
                basin_force_magnitude=float(latent.metadata.get("basin_force_magnitude", 0.0)),
                basin_prior_spread=float(latent.metadata.get("basin_prior_spread", 0.0)),
                navigation_decision=str(latent.metadata.get("navigation_decision", "{}")),
                penalty_breakdown=str(latent.metadata.get("penalty_breakdown", "{}")),
                decode_provenance=(
                    f"preview={preview_decode_source};commit={commit_decode_source};"
                    f"preview_cadence={cfg.preview_decode_cadence}"
                ),
                coherence_component_entropy=float(coherence_breakdown.weighted_contributions.get("entropy", 0.0)),
                coherence_component_density=float(coherence_breakdown.weighted_contributions.get("density", 0.0)),
                coherence_component_token_stability=float(
                    coherence_breakdown.weighted_contributions.get("token_stability", 0.0)
                ),
                coherence_component_smoothness=float(coherence_breakdown.weighted_contributions.get("smoothness", 0.0)),
                coherence_component_branch_agreement=float(
                    coherence_breakdown.weighted_contributions.get("branch_agreement", 0.0)
                ),
                coherence_component_known_state_similarity=float(
                    coherence_breakdown.weighted_contributions.get("known_state_similarity", 0.0)
                ),
                adaptive_noise=float(latent.metadata.get("adaptive_noise", cfg.noise_amplitude * anneal)),
                adaptive_attractor_weight=float(
                    latent.metadata.get("adaptive_attractor_weight", cfg.attractor_force_weight)
                ),
                status=self.state.status,
            )
            self.state.latest_tick = tick
            self.tick_history.append(tick)
            coherence_window.append(float(latent.coherence))
            self.state.step_idx += 1
            anneal *= cfg.anneal_rate
            elapsed = time.perf_counter() - tick_start
            time.sleep(max(period - elapsed, 0.0))

        self.state.active = False

    def _decode_preview(self, latent: LatentState, max_tokens: int) -> str:
        if self._decode_capability_level() == "true_latent_readout":
            return self.runtime.decode_true_latent_readout_preview(latent, max_tokens=max_tokens)
        return self.runtime.decode_approximate_prompt_synthesis_preview(latent, max_tokens=max_tokens)

    def _seed_latent_state(self, cfg: DreamConfig, prompt: str) -> LatentState:
        latent = self._normalize_latent_state(self.runtime.capture_latent_state(self.state.run_id, cfg.basin.value, prompt))
        prior = self.atlas.basin_prior_stats(cfg.basin.value, ref_vector=latent.latent_vector)
        if prior:
            basin_centroid = self._align_vector_shape(np.asarray(prior["centroid"], dtype=np.float32), latent.latent_vector)
            basin_force = self._align_vector_shape(np.asarray(prior["force_vector"], dtype=np.float32), latent.latent_vector)
            retention = float(cfg.basin_retention)
            drift = float(cfg.basin_drift)
            force_weight = float(cfg.basin_force_weight)
            blended = (
                ((1.0 - retention) * latent.latent_vector)
                + (retention * basin_centroid)
                + (force_weight * basin_force)
                + (drift * (latent.latent_vector - basin_centroid))
            )
            latent = latent.clone_with_vector(blended, phase=DreamPhase.HYPNAGOGIC)
            latent.metadata["basin_sample_count"] = int(prior.get("sample_count", 0))
            latent.metadata["basin_mean_coherence"] = float(prior.get("mean_coherence", 0.0))
            latent.metadata["basin_mean_density"] = float(prior.get("mean_density", 0.0))
            latent.metadata["basin_prior_spread"] = float(prior.get("prior_spread", 0.0))
            latent.metadata["basin_force_magnitude"] = float(np.linalg.norm(basin_force))
        else:
            latent.metadata["basin_sample_count"] = 0
            latent.metadata["basin_mean_coherence"] = 0.0
            latent.metadata["basin_mean_density"] = 0.0
            latent.metadata["basin_prior_spread"] = 0.0
            latent.metadata["basin_force_magnitude"] = 0.0
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
        adaptive_noise: float,
        adaptive_attractor_weight: float,
    ) -> list[LatentState]:
        neighbors = self._neighbor_vector(latent.latent_vector)
        prior = self.atlas.basin_prior_stats(latent.basin, ref_vector=latent.latent_vector)
        basin_centroid = (
            self._align_vector_shape(np.asarray(prior["centroid"], dtype=np.float32), latent.latent_vector)
            if prior
            else latent.latent_vector
        )
        basin_force = (
            self._align_vector_shape(np.asarray(prior["force_vector"], dtype=np.float32), latent.latent_vector)
            if prior
            else np.zeros_like(latent.latent_vector, dtype=np.float32)
        )
        attractor_stats = self.atlas.compute_attractor_vector(
            basin=latent.basin,
            ref_vector=latent.latent_vector,
            top_k=cfg.attractor_top_k,
        )
        attractor_force = (
            self._align_vector_shape(np.asarray(attractor_stats["force_vector"], dtype=np.float32), latent.latent_vector)
            if attractor_stats
            else np.zeros_like(latent.latent_vector, dtype=np.float32)
        )
        candidates = []
        branch_count = max(cfg.branch_count, 1)
        for branch_idx in range(branch_count):
            branch_id = f"{step_idx:06d}-b{branch_idx:02d}"
            branch_seed = self._branch_seed(self.state.run_id, step_idx, branch_idx)
            rng = np.random.default_rng(branch_seed)
            target = 0.7 * neighbors + 0.3 * latent.latent_vector
            target = (
                ((1.0 - cfg.basin_retention) * target)
                + (cfg.basin_retention * basin_centroid)
                + (cfg.basin_force_weight * basin_force)
                + (adaptive_attractor_weight * attractor_force)
                + (cfg.basin_drift * (latent.latent_vector - basin_centroid))
            )
            constrained_target, target_distance, target_distance_penalty = self._apply_step_constraint(
                latent.latent_vector,
                target,
                max_distance=cfg.max_step_distance,
            )
            perturb = rng.normal(scale=adaptive_noise * 0.5, size=target.shape).astype(np.float32)
            proposal = self.runtime.evolve_latent_state(
                latent,
                target_vector=constrained_target + perturb,
                noise_scale=adaptive_noise,
                noise_seed=branch_seed,
            )
            proposal = self._normalize_latent_state(proposal)
            proposal.phase = phase
            branch_agreement = 1.0 if branch_count == 1 else 1.0 - (adaptive_noise * 0.25)
            branch_smoothness = self._smoothness(latent.latent_vector, proposal.latent_vector)
            branch_density = self._estimate_local_density(proposal.latent_vector)
            branch_coherence_estimate = self._branch_coherence_estimate(
                local_density=branch_density,
                smoothness=branch_smoothness,
                branch_agreement=branch_agreement,
                weights=cfg.weights,
            )
            distance_to_attractor = self._distance_to_attractor(proposal.latent_vector, basin=proposal.basin)
            proposed_distance = float(np.linalg.norm(proposal.latent_vector - latent.latent_vector))
            step_distance_penalty = max(0.0, (proposed_distance - cfg.max_step_distance) / max(cfg.max_step_distance, 1e-6))
            curvature_penalty = self._curvature_penalty(
                prev_direction=getattr(self, "_previous_direction", np.zeros_like(proposal.latent_vector)),
                current=latent.latent_vector,
                proposed=proposal.latent_vector,
            )
            basin_boundary_cost = self._basin_boundary_cost(
                vec=proposal.latent_vector,
                centroid=basin_centroid,
                spread=float(prior.get("prior_spread", 0.0)) if prior else 0.0,
            )
            branch_score = self._branch_score(
                coherence_estimate=branch_coherence_estimate,
                distance_to_attractor=distance_to_attractor,
                temporal_smoothness=branch_smoothness,
                step_distance_penalty=step_distance_penalty + target_distance_penalty,
                curvature_penalty=curvature_penalty,
                curvature_weight=cfg.curvature_penalty_weight,
                basin_boundary_cost=basin_boundary_cost,
                basin_boundary_weight=cfg.basin_boundary_cost_weight,
            )
            distance_penalty = min(distance_to_attractor / 3.0, 1.0)
            proposal.metadata["branch_agreement"] = branch_agreement
            proposal.metadata["branch_id"] = branch_id
            proposal.metadata["branch_seed"] = branch_seed
            proposal.metadata["branch_density_estimate"] = branch_density
            proposal.metadata["branch_coherence_estimate"] = branch_coherence_estimate
            proposal.metadata["distance_to_attractor"] = distance_to_attractor
            proposal.metadata["temporal_smoothness"] = branch_smoothness
            proposal.metadata["branch_score"] = branch_score
            proposal.metadata["penalty_breakdown"] = json.dumps(
                {
                    "distance_penalty": float(distance_penalty),
                    "step_distance_penalty": float(step_distance_penalty + target_distance_penalty),
                    "curvature_penalty": float(curvature_penalty),
                    "basin_boundary_cost": float(basin_boundary_cost),
                }
            )
            proposal.metadata["navigation_decision"] = json.dumps(
                {
                    "target_distance": float(target_distance),
                    "max_step_distance": float(cfg.max_step_distance),
                    "adaptive_noise": float(adaptive_noise),
                    "adaptive_attractor_weight": float(adaptive_attractor_weight),
                    "attractor_sample_count": int(attractor_stats.get("sample_count", 0)) if attractor_stats else 0,
                    "attractor_mean_recurrence": float(attractor_stats.get("mean_recurrence", 0.0))
                    if attractor_stats
                    else 0.0,
                    "attractor_mean_dwell_time": float(attractor_stats.get("mean_dwell_time", 0.0))
                    if attractor_stats
                    else 0.0,
                }
            )
            proposal.metadata["branch_objective_scores"] = {
                "coherence_estimate": float(branch_coherence_estimate),
                "temporal_smoothness": float(branch_smoothness),
                "distance_penalty": float(distance_penalty),
                "step_distance_penalty": float(step_distance_penalty + target_distance_penalty),
                "curvature_penalty": float(curvature_penalty),
                "basin_boundary_cost": float(basin_boundary_cost),
                "total_score": float(branch_score),
            }
            proposal.metadata["basin_sample_count"] = int(prior.get("sample_count", 0)) if prior else 0
            proposal.metadata["basin_mean_coherence"] = float(prior.get("mean_coherence", 0.0)) if prior else 0.0
            proposal.metadata["basin_mean_density"] = float(prior.get("mean_density", 0.0)) if prior else 0.0
            proposal.metadata["basin_prior_spread"] = float(prior.get("prior_spread", 0.0)) if prior else 0.0
            proposal.metadata["basin_force_magnitude"] = float(np.linalg.norm(basin_force))
            candidates.append(proposal)
        return candidates

    def _choose_candidate(self, candidates: list[LatentState], *, prev_latent: LatentState) -> LatentState:
        if len(candidates) == 1:
            candidate = candidates[0]
            candidate.metadata["branch_rank"] = 1
            candidate.metadata["candidate_scores"] = str(
                [
                    {
                        "branch_id": candidate.metadata.get("branch_id", ""),
                        "score": float(candidate.metadata.get("branch_score", 0.0)),
                        "objective_scores": candidate.metadata.get("branch_objective_scores", {}),
                    }
                ]
            )
            candidate.metadata["rejected_candidates"] = "[]"
            candidate.metadata["selected_branch_trace"] = json.dumps(
                {
                    "policy": "single_candidate",
                    "winner_branch_id": candidate.metadata.get("branch_id", ""),
                    "winner_score": float(candidate.metadata.get("branch_score", 0.0)),
                }
            )
            candidate.metadata["selected_from_state_id"] = prev_latent.state_id
            return candidate
        scored = sorted(
            enumerate(candidates),
            key=lambda item: float(item[1].metadata.get("branch_score", float("-inf"))),
            reverse=True,
        )
        selected_rank = self._select_rank_by_policy(scored=scored, policy=prev_latent.metadata.get("branch_selection_policy"))
        for rank, (_, candidate) in enumerate(scored, start=1):
            candidate.metadata["branch_rank"] = rank
        candidate_scores = []
        for rank, (_, candidate) in enumerate(scored, start=1):
            candidate_scores.append(
                {
                    "branch_id": str(candidate.metadata.get("branch_id", "")),
                    "score": float(candidate.metadata.get("branch_score", 0.0)),
                    "rank": rank,
                    "objective_scores": candidate.metadata.get("branch_objective_scores", {}),
                }
            )
        selected = scored[selected_rank][1]
        rejected_candidates = [
            {
                "branch_id": str(candidate.metadata.get("branch_id", "")),
                "score": float(candidate.metadata.get("branch_score", 0.0)),
                "rank": rank,
                "objective_scores": candidate.metadata.get("branch_objective_scores", {}),
                "rejected_reason": "not_selected_by_policy",
            }
            for rank, (_, candidate) in enumerate(scored, start=1)
            if candidate is not selected
        ]
        selected.metadata["selected_from_state_id"] = prev_latent.state_id
        selected.metadata["candidate_scores"] = str(candidate_scores)
        selected.metadata["rejected_candidates"] = str(rejected_candidates)
        selected.metadata["selected_branch_trace"] = json.dumps(
            {
                "policy": str(prev_latent.metadata.get("branch_selection_policy", BranchSelectionPolicy.GREEDY.value)),
                "winner_branch_id": str(selected.metadata.get("branch_id", "")),
                "winner_rank": int(selected.metadata.get("branch_rank", 1)),
                "winner_score": float(selected.metadata.get("branch_score", 0.0)),
            }
        )
        prev_vec = np.asarray(prev_latent.latent_vector, dtype=np.float32).reshape(-1)
        next_vec = np.asarray(selected.latent_vector, dtype=np.float32).reshape(-1)
        prev_vec = self._align_vector_shape(prev_vec, next_vec)
        self._previous_direction = next_vec - prev_vec
        return selected

    def _select_rank_by_policy(self, *, scored: list[tuple[int, LatentState]], policy: str | None) -> int:
        try:
            effective_policy = BranchSelectionPolicy(str(policy or BranchSelectionPolicy.GREEDY.value))
        except ValueError:
            effective_policy = BranchSelectionPolicy.GREEDY
        if effective_policy == BranchSelectionPolicy.GREEDY:
            return 0
        best_idx = 0
        if effective_policy == BranchSelectionPolicy.EPSILON_GREEDY:
            epsilon = float(getattr(self, "_branch_policy_exploration", 0.1))
            if self._policy_rng().random() < epsilon:
                return int(self._policy_rng().integers(0, len(scored)))
            return best_idx
        temperature = max(1e-6, float(getattr(self, "_branch_policy_temperature", 0.75)))
        scores = np.asarray([float(candidate.metadata.get("branch_score", 0.0)) for _, candidate in scored], dtype=np.float64)
        adjusted = scores - np.max(scores)
        probs = np.exp(adjusted / temperature)
        probs_sum = float(np.sum(probs))
        if probs_sum <= 0:
            return best_idx
        probs = probs / probs_sum
        selected = int(self._policy_rng().choice(np.arange(len(scored)), p=probs))
        return selected

    def _policy_rng(self) -> np.random.Generator:
        if not hasattr(self, "_policy_random"):
            self._policy_random = np.random.default_rng()
        return self._policy_random

    def _preview_decode_source(self) -> str:
        if self._decode_capability_level() == "true_latent_readout":
            return "decode_true_latent_readout_preview"
        return "decode_approximate_prompt_synthesis_preview"

    @staticmethod
    def _adaptive_noise(cfg: DreamConfig, *, anneal: float, recent_coherence: float) -> float:
        coherence_term = max(0.0, min(1.0, recent_coherence))
        reduction = cfg.coherence_gain * coherence_term
        floor = max(0.01, cfg.noise_amplitude * cfg.exploration_floor)
        adaptive = (cfg.noise_amplitude * anneal) * (1.0 - reduction)
        return float(max(floor, adaptive))

    @staticmethod
    def _adaptive_attractor_weight(cfg: DreamConfig, *, recent_coherence: float) -> float:
        coherence_term = max(0.0, min(1.0, recent_coherence))
        # As coherence grows, nudge slightly harder toward attractors to stabilize narratives.
        return float(min(1.0, cfg.attractor_force_weight + (0.25 * coherence_term)))

    def _decode_capability_level(self) -> str:
        caps = self.runtime.capabilities()
        if caps.supports_instrumented_latents and caps.supports_true_latent_readout:
            return "true_latent_readout"
        if caps.supports_instrumented_latents:
            return "instrumented_capture_only"
        return "approximate_prompt_synthesis_only"

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
    def _branch_score(
        *,
        coherence_estimate: float,
        distance_to_attractor: float,
        temporal_smoothness: float,
        step_distance_penalty: float = 0.0,
        curvature_penalty: float = 0.0,
        curvature_weight: float = 0.0,
        basin_boundary_cost: float = 0.0,
        basin_boundary_weight: float = 0.0,
    ) -> float:
        distance_penalty = min(distance_to_attractor / 3.0, 1.0)
        score = (
            (0.65 * coherence_estimate)
            + (0.25 * temporal_smoothness)
            - (0.35 * distance_penalty)
            - (0.2 * step_distance_penalty)
            - (curvature_weight * curvature_penalty)
            - (basin_boundary_weight * basin_boundary_cost)
        )
        return float(score)

    @staticmethod
    def _apply_step_constraint(current: np.ndarray, target: np.ndarray, *, max_distance: float) -> tuple[np.ndarray, float, float]:
        current = np.asarray(current, dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        target = DreamController._align_vector_shape(target, current)
        delta = target - current
        dist = float(np.linalg.norm(delta))
        if dist <= max_distance:
            return target, dist, 0.0
        scaled = current + (delta / max(dist, 1e-9)) * max_distance
        overflow_penalty = (dist - max_distance) / max(max_distance, 1e-6)
        return scaled.astype(np.float32), dist, float(overflow_penalty)

    @staticmethod
    def _curvature_penalty(*, prev_direction: np.ndarray, current: np.ndarray, proposed: np.ndarray) -> float:
        prev_direction = np.asarray(prev_direction, dtype=np.float32).reshape(-1)
        current = np.asarray(current, dtype=np.float32).reshape(-1)
        proposed = np.asarray(proposed, dtype=np.float32).reshape(-1)
        proposed_direction = proposed - DreamController._align_vector_shape(current, proposed)
        prev_direction = DreamController._align_vector_shape(prev_direction, proposed_direction)
        prev_norm = float(np.linalg.norm(prev_direction))
        next_norm = float(np.linalg.norm(proposed_direction))
        if prev_norm <= 1e-9 or next_norm <= 1e-9:
            return 0.0
        cos = float(np.dot(prev_direction, proposed_direction) / (prev_norm * next_norm + 1e-9))
        cos = max(-1.0, min(1.0, cos))
        return float((1.0 - cos) / 2.0)

    @staticmethod
    def _basin_boundary_cost(*, vec: np.ndarray, centroid: np.ndarray, spread: float) -> float:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        centroid = np.asarray(centroid, dtype=np.float32).reshape(-1)
        centroid = DreamController._align_vector_shape(centroid, vec)
        if spread <= 1e-9:
            return 0.0
        distance = float(np.linalg.norm(vec - centroid))
        return float(max(0.0, (distance - spread) / (spread + 1e-9)))

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
        if not any(chunk.strip() for chunk in last):
            return False
        unique = len(set(last))
        return unique <= max(2, cfg.stability_window - 1)

    @staticmethod
    def _merge_committed_chunk(current_committed: str, chunk: str, max_len: int) -> str:
        chunk = chunk.strip()
        if not chunk:
            return current_committed
        updated = (current_committed + " " + chunk).strip()
        return updated[-max_len:]
