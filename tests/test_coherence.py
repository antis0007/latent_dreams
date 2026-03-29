from gguf_dream_lab.backend.dream.coherence import score_coherence
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
