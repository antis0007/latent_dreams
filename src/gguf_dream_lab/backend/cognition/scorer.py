from __future__ import annotations

from collections.abc import Sequence

from gguf_dream_lab.backend.cognition.models import RankedCandidate, ScoreBreakdown, ScoreWeights, ThoughtCandidate

FACTUAL_CUES = ("is", "are", "was", "were", "data", "report", "study")


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _jaccard(lhs: set[str], rhs: set[str]) -> float:
    if not lhs and not rhs:
        return 1.0
    if not lhs or not rhs:
        return 0.0
    return len(lhs & rhs) / len(lhs | rhs)


def _relevance(candidate: ThoughtCandidate, user_goal: str) -> float:
    return _jaccard(_tokenize(candidate.text), _tokenize(user_goal))


def _coherence(candidate: ThoughtCandidate) -> float:
    words = candidate.text.split()
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    punctuation_penalty = 0.15 if candidate.text.count("?") > 2 else 0.0
    return max(0.0, min(1.0, unique_ratio - punctuation_penalty + 0.25))


def _novelty(candidate: ThoughtCandidate, recent_outputs: Sequence[str]) -> float:
    if not recent_outputs:
        return 1.0
    candidate_tokens = _tokenize(candidate.text)
    similarities = [_jaccard(candidate_tokens, _tokenize(text)) for text in recent_outputs]
    return max(0.0, 1.0 - max(similarities, default=0.0))


def _groundedness(candidate: ThoughtCandidate) -> float:
    return 1.0 if candidate.evidence_refs else 0.45


def _safety_margin(candidate: ThoughtCandidate) -> float:
    return 0.0 if candidate.safety_flags else 1.0


def _hallucination_risk(candidate: ThoughtCandidate) -> float:
    lower = candidate.text.lower()
    looks_factual = any(cue in lower for cue in FACTUAL_CUES)
    if looks_factual and not candidate.evidence_refs:
        return 0.75
    if not candidate.evidence_refs:
        return 0.35
    return 0.1


def rank_candidates(
    candidates: Sequence[ThoughtCandidate],
    *,
    user_goal: str,
    recent_outputs: Sequence[str],
    weights: ScoreWeights | None = None,
) -> list[RankedCandidate]:
    weights = weights or ScoreWeights()
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        relevance = _relevance(candidate, user_goal)
        coherence = _coherence(candidate)
        novelty = _novelty(candidate, recent_outputs)
        groundedness = _groundedness(candidate)
        safety_margin = _safety_margin(candidate)
        hallucination_risk = _hallucination_risk(candidate)
        total = (
            relevance * weights.relevance
            + coherence * weights.coherence
            + novelty * weights.novelty
            + groundedness * weights.groundedness
            + safety_margin * weights.safety_margin
            - hallucination_risk * weights.hallucination_risk
        )
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                score=ScoreBreakdown(
                    relevance=relevance,
                    coherence=coherence,
                    novelty=novelty,
                    groundedness=groundedness,
                    safety_margin=safety_margin,
                    hallucination_risk=hallucination_risk,
                    total=total,
                ),
            )
        )
    return sorted(ranked, key=lambda item: item.score.total, reverse=True)
