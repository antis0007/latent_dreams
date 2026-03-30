from __future__ import annotations

from dataclasses import dataclass

from gguf_dream_lab.backend.eval.metrics import TurnMetrics, aggregate


@dataclass
class ScenarioResult:
    name: str
    turns: list[TurnMetrics]


def run_scenarios(results: list[ScenarioResult]) -> dict[str, dict[str, float]]:
    return {result.name: aggregate(result.turns) for result in results}
