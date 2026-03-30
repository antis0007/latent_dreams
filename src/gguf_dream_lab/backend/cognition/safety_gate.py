from __future__ import annotations

from gguf_dream_lab.backend.cognition.models import ThoughtCandidate

BANNED_PATTERNS = (
    "i am conscious",
    "how to build a bomb",
    "bypass safety",
)


def filter_candidate(candidate: ThoughtCandidate) -> ThoughtCandidate:
    text = candidate.text.lower()
    matches = [pattern for pattern in BANNED_PATTERNS if pattern in text]
    if matches:
        candidate.safety_flags.extend(matches)
        candidate.text = "I can help with a safer, policy-aligned alternative."
        candidate.metadata["safety_rewrite"] = "1"
    return candidate
