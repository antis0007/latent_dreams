from __future__ import annotations

from gguf_dream_lab.backend.memory.schemas import ShortTermMemory


class ShortTermMemoryStore:
    def __init__(self):
        self._sessions: dict[str, ShortTermMemory] = {}

    def load(self, session_id: str) -> ShortTermMemory:
        return self._sessions.setdefault(session_id, ShortTermMemory(session_id=session_id))

    def update(
        self,
        session_id: str,
        *,
        user_objective: str | None = None,
        constraints: list[str] | None = None,
        open_threads: list[str] | None = None,
        assistant_commitment: str | None = None,
        unresolved_uncertainty: str | None = None,
    ) -> ShortTermMemory:
        current = self.load(session_id)
        if user_objective:
            current.user_objective = user_objective
        if constraints:
            current.constraints = constraints[-8:]
        if open_threads:
            current.open_threads = open_threads[-12:]
        if assistant_commitment:
            current.assistant_commitments = (current.assistant_commitments + [assistant_commitment])[-12:]
        if unresolved_uncertainty:
            current.unresolved_uncertainties = (current.unresolved_uncertainties + [unresolved_uncertainty])[-12:]
        return current
