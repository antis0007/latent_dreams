from __future__ import annotations

import random
from collections.abc import Sequence

from gguf_dream_lab.backend.cognition.models import SelfModelState, ThoughtCandidate
from gguf_dream_lab.backend.memory.retrieval import MemoryHit

SEED_INTENTS = ("analytic", "poetic", "questioning", "concise")


def _mutate(text: str, mode: str) -> str:
    if mode == "analytic":
        return f"Breakdown: {text}"
    if mode == "poetic":
        return f"Imagery angle: {text}"
    if mode == "questioning":
        return f"Clarify: {text}?"
    return f"Summary: {text}"


def _base_prompt(prompt: str) -> str:
    return prompt.strip() or "autonomous latent drift snapshot"


def _recombine(candidates: list[ThoughtCandidate]) -> ThoughtCandidate | None:
    if len(candidates) < 3:
        return None
    first = candidates[0].text.split("|", maxsplit=1)[0].strip()
    second = candidates[1].text.split("|", maxsplit=1)[0].strip()
    third = candidates[2].text.split("|", maxsplit=1)[0].strip()
    merged = f"Synthesis: {first}. Structure: {second}. Novel hook: {third}."
    evidence = list({ref for candidate in candidates[:3] for ref in candidate.evidence_refs})
    return ThoughtCandidate(text=merged, evidence_refs=evidence, metadata={"seed_intent": "recombined", "steps": 1.0})


def generate_candidates(
    prompt: str,
    memory_hits: Sequence[MemoryHit],
    self_state: SelfModelState,
    n_paths: int = 8,
    steps: int = 3,
) -> list[ThoughtCandidate]:
    prompt = _base_prompt(prompt)
    rng = random.Random(f"{prompt}:{len(memory_hits)}:{self_state.intent}:{n_paths}:{steps}")
    references = [hit.text for hit in memory_hits[:3]]
    seeds = list(SEED_INTENTS)
    rng.shuffle(seeds)

    candidates: list[ThoughtCandidate] = []
    for idx in range(max(1, n_paths)):
        seed = seeds[idx % len(seeds)]
        text = f"{seed} response to: {prompt}"
        for _ in range(max(1, steps)):
            text = _mutate(text, seed)
        if self_state.creative_drive > 0.7 and seed == "poetic":
            text += " with one novel metaphor"
        if references:
            text += f" | context: {references[idx % len(references)]}"
        candidates.append(
            ThoughtCandidate(
                text=text,
                evidence_refs=[f"mem:{index}" for index in range(len(references))],
                metadata={"seed_intent": seed, "steps": float(steps)},
            )
        )

    recombined = _recombine(candidates)
    if recombined is not None:
        candidates.append(recombined)
    return candidates
