from __future__ import annotations

from dataclasses import dataclass

from gguf_dream_lab.backend.memory.long_term import LongTermMemoryStore
from gguf_dream_lab.backend.memory.short_term import ShortTermMemoryStore


@dataclass
class MemoryHit:
    text: str
    score: float
    source: str


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _overlap(lhs: str, rhs: str) -> float:
    left_tokens = _tokenize(lhs)
    right_tokens = _tokenize(rhs)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class MemoryRetrieval:
    def __init__(self, short_term: ShortTermMemoryStore, long_term: LongTermMemoryStore):
        self.short_term = short_term
        self.long_term = long_term

    def search(self, query: str, session_id: str, *, top_k: int = 12) -> list[MemoryHit]:
        short = self.short_term.load(session_id)
        profile = self.long_term.semantic(session_id)
        hits: list[MemoryHit] = []

        if short.user_objective:
            hits.append(MemoryHit(text=short.user_objective, score=_overlap(query, short.user_objective), source="short_term"))

        for thread in short.open_threads:
            hits.append(MemoryHit(text=thread, score=_overlap(query, thread), source="open_thread"))

        for key, value in profile.stable_preferences.items():
            text = f"{key}: {value}"
            hits.append(MemoryHit(text=text, score=_overlap(query, text), source="semantic_profile"))

        for episode in self.long_term.episodes(session_id):
            overlap = _overlap(query, episode.summary)
            hits.append(
                MemoryHit(
                    text=episode.summary,
                    score=overlap * 0.6 + episode.salience * 0.4,
                    source="episodic",
                )
            )

        hits = sorted(hits, key=lambda item: item.score, reverse=True)
        return hits[:top_k]
