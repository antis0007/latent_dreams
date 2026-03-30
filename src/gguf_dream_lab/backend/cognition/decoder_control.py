from __future__ import annotations

from gguf_dream_lab.backend.cognition.models import DecodeConfig, SelfModelState


def config_from_state(state: SelfModelState) -> DecodeConfig:
    if state.intent in {"analyze", "answer"}:
        return DecodeConfig(temperature=0.35, top_p=0.75, top_k=40, max_tokens=420)
    if state.intent in {"brainstorm", "create"}:
        return DecodeConfig(temperature=0.85, top_p=0.95, top_k=120, max_tokens=520)
    if state.intent == "ask":
        return DecodeConfig(temperature=0.45, top_p=0.8, top_k=50, max_tokens=240)
    return DecodeConfig(temperature=0.55, top_p=0.85, top_k=60, max_tokens=360)
