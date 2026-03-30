from __future__ import annotations

from collections.abc import Sequence

from gguf_dream_lab.backend.cognition.models import Action, RankedCandidate, SelfModelState


def decide_action(ranked: Sequence[RankedCandidate], self_state: SelfModelState) -> Action:
    if not ranked:
        return Action.ASK_USER
    top = ranked[0].score
    if top.safety_margin < 0.5 or self_state.risk_level > 0.7:
        return Action.ASK_USER
    if self_state.confidence < 0.35 and top.groundedness < 0.6:
        return Action.ASK_USER
    if top.total < 0.45:
        return Action.REFLECT_ONLY
    return Action.RESPOND
