import pytest

from gguf_dream_lab.backend.dream.coherence import score_coherence, score_coherence_breakdown
from gguf_dream_lab.config.models import CoherenceWeights


def test_coherence_score_range():
    w = CoherenceWeights()
    score = score_coherence(
        entropy=0.4,
        local_density=1.2,
        token_stability=0.7,
        smoothness=0.8,
        branch_agreement=0.6,
        known_state_similarity=0.5,
        weights=w,
    )
    assert 0.0 <= score <= 1.0


def test_coherence_breakdown_contributions_sum_to_score():
    w = CoherenceWeights()
    breakdown = score_coherence_breakdown(
        entropy=0.3,
        local_density=0.9,
        token_stability=0.5,
        smoothness=0.4,
        branch_agreement=0.8,
        known_state_similarity=0.2,
        weights=w,
    )

    contribution_sum = sum(breakdown.weighted_contributions.values())
    assert breakdown.score == pytest.approx(contribution_sum)
    assert set(breakdown.components) == {
        "entropy",
        "density",
        "token_stability",
        "smoothness",
        "branch_agreement",
        "known_state_similarity",
    }
