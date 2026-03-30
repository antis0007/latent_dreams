from __future__ import annotations

from collections import defaultdict

from gguf_dream_lab.backend.memory.schemas import EpisodicMemoryRecord, SemanticProfile


class LongTermMemoryStore:
    def __init__(self):
        self._episodes: dict[str, list[EpisodicMemoryRecord]] = defaultdict(list)
        self._semantic: dict[str, SemanticProfile] = {}

    def write_episode(self, record: EpisodicMemoryRecord) -> None:
        self._episodes[record.session_id].append(record)
        profile = self._semantic.setdefault(record.session_id, SemanticProfile(session_id=record.session_id))
        profile.stable_preferences.update(record.user_preference_updates)

    def episodes(self, session_id: str) -> list[EpisodicMemoryRecord]:
        return list(self._episodes.get(session_id, []))

    def semantic(self, session_id: str) -> SemanticProfile:
        return self._semantic.setdefault(session_id, SemanticProfile(session_id=session_id))

    def decay_salience(self, *, half_life_turns: int = 24) -> None:
        if half_life_turns <= 0:
            return
        decay = 0.5 ** (1.0 / half_life_turns)
        for session_episodes in self._episodes.values():
            for episode in session_episodes:
                episode.salience *= decay

    def consolidate(self, session_id: str, *, min_salience: float = 0.05) -> None:
        episodes = self._episodes.get(session_id, [])
        self._episodes[session_id] = [episode for episode in episodes if episode.salience >= min_salience]
