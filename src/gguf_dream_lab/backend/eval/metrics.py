from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnMetrics:
    novelty: float
    coherence: float
    relevance: float
    consistency: float
    groundedness: float
    initiative_quality: float
    safety_pass: float


def aggregate(turns: list[TurnMetrics]) -> dict[str, float]:
    if not turns:
        return {
            "novelty": 0.0,
            "coherence": 0.0,
            "relevance": 0.0,
            "consistency": 0.0,
            "groundedness": 0.0,
            "initiative_quality": 0.0,
            "safety_pass": 0.0,
        }
    count = float(len(turns))
    return {
        "novelty": sum(t.novelty for t in turns) / count,
        "coherence": sum(t.coherence for t in turns) / count,
        "relevance": sum(t.relevance for t in turns) / count,
        "consistency": sum(t.consistency for t in turns) / count,
        "groundedness": sum(t.groundedness for t in turns) / count,
        "initiative_quality": sum(t.initiative_quality for t in turns) / count,
        "safety_pass": sum(t.safety_pass for t in turns) / count,
    }
