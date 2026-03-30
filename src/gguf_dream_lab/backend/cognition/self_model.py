from __future__ import annotations

from collections.abc import Sequence

from gguf_dream_lab.backend.cognition.models import SelfModelState
from gguf_dream_lab.backend.memory.retrieval import MemoryHit
from gguf_dream_lab.backend.memory.schemas import ShortTermMemory


def update_self_model(
    *,
    user_msg: str,
    short_term: ShortTermMemory,
    memory_hits: Sequence[MemoryHit],
) -> SelfModelState:
    lower = user_msg.lower()
    intent = "answer"
    if any(token in lower for token in ("brainstorm", "creative", "poem", "story")):
        intent = "create"
    elif "?" in user_msg and len(user_msg.split()) < 10:
        intent = "ask"
    elif any(token in lower for token in ("analyze", "compare", "tradeoff", "evaluate")):
        intent = "analyze"

    uncertainty = "low"
    if "uncertain" in lower or "not sure" in lower:
        uncertainty = "interpretive"

    memory_quality = max((hit.score for hit in memory_hits), default=0.0)
    confidence = max(0.2, min(0.95, 0.45 + 0.4 * memory_quality))

    if intent == "create":
        creative_drive = 0.85
        initiative_budget = 0.6
    elif intent == "ask":
        creative_drive = 0.2
        initiative_budget = 0.35
    else:
        creative_drive = 0.45
        initiative_budget = 0.5

    tone = "warm" if short_term.user_objective else "neutral"
    risk_level = 0.2 if "medical" not in lower and "legal" not in lower else 0.75

    return SelfModelState(
        intent=intent,
        confidence=confidence,
        uncertainty=uncertainty,
        tone=tone,
        initiative_budget=initiative_budget,
        creative_drive=creative_drive,
        risk_level=risk_level,
    )
