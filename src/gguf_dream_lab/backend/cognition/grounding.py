from __future__ import annotations

from gguf_dream_lab.backend.cognition.models import ThoughtCandidate

FACTUAL_CUES = ("is", "are", "was", "were", "data", "study", "report")
SPECULATIVE_PREFIX = "Speculative idea: "


def classify_claim(candidate: ThoughtCandidate) -> str:
    text = candidate.text.lower()
    if any(cue in text for cue in FACTUAL_CUES):
        return "factual"
    if "imagine" in text or "metaphor" in text:
        return "imaginative"
    if "should" in text:
        return "opinion"
    return "speculative"


def enforce_grounding(candidate: ThoughtCandidate) -> ThoughtCandidate:
    claim_type = classify_claim(candidate)
    if claim_type == "factual" and not candidate.evidence_refs:
        candidate.text = f"{SPECULATIVE_PREFIX}{candidate.text}"
        candidate.metadata["grounding_labeled"] = "1"
    candidate.metadata["claim_type"] = claim_type
    return candidate
