from __future__ import annotations

from dataclasses import dataclass

from gguf_dream_lab.backend.cognition.models import SelfModelState
from gguf_dream_lab.backend.memory.retrieval import MemoryHit


@dataclass
class IdleThought:
    prompt: str
    salience: float
    surface_next_turn: bool


def schedule_idle_thoughts(*, self_state: SelfModelState, memory_hits: list[MemoryHit], max_items: int = 3) -> list[IdleThought]:
    base_prompts = [
        "What does the user likely need next?",
        "Which unresolved thread is highest value?",
        "Is there a creative suggestion that stays relevant?",
    ]
    salience_boost = max((hit.score for hit in memory_hits), default=0.2)
    thoughts: list[IdleThought] = []
    for prompt in base_prompts[: max(1, max_items)]:
        salience = min(1.0, 0.3 + 0.4 * self_state.initiative_budget + 0.3 * salience_boost)
        thoughts.append(
            IdleThought(
                prompt=prompt,
                salience=salience,
                surface_next_turn=salience > 0.65 and self_state.risk_level < 0.6,
            )
        )
    return thoughts
