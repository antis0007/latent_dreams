from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    RESPOND = "respond"
    ASK_USER = "ask_user"
    REFLECT_ONLY = "reflect_only"


@dataclass
class SelfModelState:
    intent: str = "answer"
    confidence: float = 0.5
    uncertainty: str = "unknown"
    tone: str = "neutral"
    initiative_budget: float = 0.25
    creative_drive: float = 0.25
    risk_level: float = 0.1


@dataclass
class ThoughtCandidate:
    text: str
    evidence_refs: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    metadata: dict[str, str | float] = field(default_factory=dict)


@dataclass
class ScoreWeights:
    relevance: float = 0.30
    coherence: float = 0.20
    novelty: float = 0.18
    groundedness: float = 0.14
    safety_margin: float = 0.10
    hallucination_risk: float = 0.08


@dataclass
class ScoreBreakdown:
    relevance: float
    coherence: float
    novelty: float
    groundedness: float
    safety_margin: float
    hallucination_risk: float
    total: float


@dataclass
class RankedCandidate:
    candidate: ThoughtCandidate
    score: ScoreBreakdown


@dataclass
class DecodeConfig:
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int


@dataclass
class TurnResult:
    action: Action
    response: str
    selected: RankedCandidate | None
