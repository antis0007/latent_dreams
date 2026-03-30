from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gguf_dream_lab.backend.cognition.arbitration import decide_action
from gguf_dream_lab.backend.cognition.decoder_control import config_from_state
from gguf_dream_lab.backend.cognition.grounding import enforce_grounding
from gguf_dream_lab.backend.cognition.models import Action, DecodeConfig, SelfModelState, ThoughtCandidate, TurnResult
from gguf_dream_lab.backend.cognition.scheduler import schedule_idle_thoughts
from gguf_dream_lab.backend.cognition.scorer import rank_candidates
from gguf_dream_lab.backend.cognition.self_model import update_self_model
from gguf_dream_lab.backend.cognition.workspace import generate_candidates
from gguf_dream_lab.backend.memory.retrieval import MemoryHit, MemoryRetrieval
from gguf_dream_lab.backend.memory.schemas import EpisodicMemoryRecord
from gguf_dream_lab.backend.memory.short_term import ShortTermMemoryStore


@dataclass
class TurnContext:
    session_id: str
    user_message: str
    recent_outputs: list[str]


class CognitiveOrchestrator:
    """Main runtime loop for hidden-workspace cognition around a frozen model."""

    def __init__(
        self,
        *,
        short_term: ShortTermMemoryStore,
        retrieval: MemoryRetrieval,
        workspace_generate: Callable[[str, list[MemoryHit], SelfModelState], list[ThoughtCandidate]] = generate_candidates,
        self_model_update: Callable[..., SelfModelState] = update_self_model,
        grounding_enforce: Callable[[ThoughtCandidate], ThoughtCandidate] = enforce_grounding,
        safety_filter: Callable[[ThoughtCandidate], ThoughtCandidate],
        decode_visible: Callable[[ThoughtCandidate, DecodeConfig, Action], str],
        write_episode: Callable[[EpisodicMemoryRecord], None],
    ):
        self.short_term = short_term
        self.retrieval = retrieval
        self.workspace_generate = workspace_generate
        self.self_model_update = self_model_update
        self.grounding_enforce = grounding_enforce
        self.safety_filter = safety_filter
        self.decode_visible = decode_visible
        self.write_episode = write_episode

    def handle_turn(self, context: TurnContext) -> TurnResult:
        return self._run_loop(context)

    def handle_autonomous_tick(self, *, session_id: str, recent_outputs: list[str], max_items: int = 1) -> list[TurnResult]:
        """Idle cognition path: generate internal prompts when there is no user input."""
        short_memory = self.short_term.load(session_id)
        memory_hits = self.retrieval.search(short_memory.user_objective or "", session_id, top_k=6)
        self_state = self.self_model_update(user_msg="", short_term=short_memory, memory_hits=memory_hits)
        thoughts = schedule_idle_thoughts(self_state=self_state, memory_hits=memory_hits, max_items=max_items)
        results: list[TurnResult] = []
        for thought in thoughts:
            context = TurnContext(session_id=session_id, user_message=thought.prompt if thought.surface_next_turn else "", recent_outputs=recent_outputs)
            results.append(self._run_loop(context))
        return results

    def _run_loop(self, context: TurnContext) -> TurnResult:
        short_memory = self.short_term.load(context.session_id)
        memory_hits = self.retrieval.search(context.user_message, context.session_id, top_k=12)
        self_state = self.self_model_update(user_msg=context.user_message, short_term=short_memory, memory_hits=memory_hits)

        candidates = self.workspace_generate(context.user_message, memory_hits, self_state)
        ranked = rank_candidates(candidates, user_goal=context.user_message, recent_outputs=context.recent_outputs)
        action = decide_action(ranked, self_state)

        if not ranked:
            response = "Could you share a bit more detail so I can respond precisely?"
            self._write_memory(context, response, self_state)
            return TurnResult(action=Action.ASK_USER, response=response, selected=None)

        selected = ranked[0]
        grounded = self.grounding_enforce(selected.candidate)
        safe = self.safety_filter(grounded)
        decode_cfg = config_from_state(self_state)
        response = self.decode_visible(safe, decode_cfg, action)

        self._write_memory(context, response, self_state)
        return TurnResult(action=action, response=response, selected=selected)

    def _write_memory(self, context: TurnContext, response: str, self_state: SelfModelState) -> None:
        self.short_term.update(
            context.session_id,
            user_objective=context.user_message or self.short_term.load(context.session_id).user_objective,
            assistant_commitment=response,
            unresolved_uncertainty=self_state.uncertainty if self_state.uncertainty != "low" else None,
        )
        self.write_episode(
            EpisodicMemoryRecord.create(
                session_id=context.session_id,
                summary=f"User: {context.user_message} | Assistant: {response}",
                salience=0.45 + self_state.initiative_budget * 0.35,
                confidence=self_state.confidence,
            )
        )
