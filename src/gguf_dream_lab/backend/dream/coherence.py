from __future__ import annotations

import numpy as np

from gguf_dream_lab.config.models import CoherenceWeights


def score_coherence(
    entropy: float,
    local_density: float,
    token_stability: float,
    smoothness: float,
    branch_agreement: float,
    known_state_similarity: float,
    weights: CoherenceWeights,
) -> float:
    components = {
        "entropy": max(0.0, 1.0 - float(entropy)),
        "density": float(np.tanh(local_density / 5.0)),
        "token_stability": float(np.clip(token_stability, 0.0, 1.0)),
        "smoothness": float(np.clip(smoothness, 0.0, 1.0)),
        "branch_agreement": float(np.clip(branch_agreement, 0.0, 1.0)),
        "known_state_similarity": float(np.clip(known_state_similarity, 0.0, 1.0)),
    }
    numerator = (
        weights.entropy * components["entropy"]
        + weights.density * components["density"]
        + weights.token_stability * components["token_stability"]
        + weights.smoothness * components["smoothness"]
        + weights.branch_agreement * components["branch_agreement"]
        + weights.known_state_similarity * components["known_state_similarity"]
    )
    denom = (
        weights.entropy
        + weights.density
        + weights.token_stability
        + weights.smoothness
        + weights.branch_agreement
        + weights.known_state_similarity
    )
    return float(numerator / max(denom, 1e-9))
