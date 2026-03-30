from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gguf_dream_lab.config.models import CoherenceWeights


@dataclass(frozen=True)
class CoherenceBreakdown:
    score: float
    components: dict[str, float]
    weighted_contributions: dict[str, float]


def score_coherence_breakdown(
    entropy: float,
    local_density: float,
    token_stability: float,
    smoothness: float,
    branch_agreement: float,
    known_state_similarity: float,
    weights: CoherenceWeights,
) -> CoherenceBreakdown:
    components = {
        "entropy": max(0.0, 1.0 - float(entropy)),
        "density": float(np.tanh(local_density / 5.0)),
        "token_stability": float(np.clip(token_stability, 0.0, 1.0)),
        "smoothness": float(np.clip(smoothness, 0.0, 1.0)),
        "branch_agreement": float(np.clip(branch_agreement, 0.0, 1.0)),
        "known_state_similarity": float(np.clip(known_state_similarity, 0.0, 1.0)),
    }
    weighted_raw = {
        "entropy": weights.entropy * components["entropy"],
        "density": weights.density * components["density"],
        "token_stability": weights.token_stability * components["token_stability"],
        "smoothness": weights.smoothness * components["smoothness"],
        "branch_agreement": weights.branch_agreement * components["branch_agreement"],
        "known_state_similarity": weights.known_state_similarity * components["known_state_similarity"],
    }
    denom = (
        weights.entropy
        + weights.density
        + weights.token_stability
        + weights.smoothness
        + weights.branch_agreement
        + weights.known_state_similarity
    )
    safe_denom = max(denom, 1e-9)
    normalized = {name: float(val / safe_denom) for name, val in weighted_raw.items()}
    score = float(sum(normalized.values()))
    return CoherenceBreakdown(score=score, components=components, weighted_contributions=normalized)


def score_coherence(
    entropy: float,
    local_density: float,
    token_stability: float,
    smoothness: float,
    branch_agreement: float,
    known_state_similarity: float,
    weights: CoherenceWeights,
) -> float:
    return score_coherence_breakdown(
        entropy=entropy,
        local_density=local_density,
        token_stability=token_stability,
        smoothness=smoothness,
        branch_agreement=branch_agreement,
        known_state_similarity=known_state_similarity,
        weights=weights,
    ).score
