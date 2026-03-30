from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ShortTermMemory:
    session_id: str
    user_objective: str = ""
    constraints: list[str] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)
    assistant_commitments: list[str] = field(default_factory=list)
    unresolved_uncertainties: list[str] = field(default_factory=list)


@dataclass
class EpisodicMemoryRecord:
    session_id: str
    timestamp: datetime
    summary: str
    salience: float
    confidence: float
    user_preference_updates: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        summary: str,
        salience: float,
        confidence: float,
        user_preference_updates: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> "EpisodicMemoryRecord":
        return cls(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            summary=summary,
            salience=max(0.0, min(1.0, salience)),
            confidence=max(0.0, min(1.0, confidence)),
            user_preference_updates=user_preference_updates or {},
            tags=tags or [],
        )


@dataclass
class SemanticProfile:
    session_id: str
    stable_preferences: dict[str, str] = field(default_factory=dict)
    topic_familiarity: dict[str, float] = field(default_factory=dict)
    preferred_tone: str = "neutral"
    recurring_goals: list[str] = field(default_factory=list)
