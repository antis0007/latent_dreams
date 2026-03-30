from gguf_dream_lab.backend.cognition.arbitration import decide_action
from gguf_dream_lab.backend.cognition.models import Action, RankedCandidate, ScoreBreakdown, SelfModelState, ThoughtCandidate
from gguf_dream_lab.backend.cognition.orchestrator import CognitiveOrchestrator, TurnContext
from gguf_dream_lab.backend.cognition.safety_gate import filter_candidate
from gguf_dream_lab.backend.cognition.scorer import rank_candidates
from gguf_dream_lab.backend.memory.long_term import LongTermMemoryStore
from gguf_dream_lab.backend.memory.retrieval import MemoryRetrieval
from gguf_dream_lab.backend.memory.short_term import ShortTermMemoryStore


def test_rank_candidates_prefers_evidence_backed_relevant_candidate():
    candidates = [
        ThoughtCandidate(text="report shows runtime gains without evidence"),
        ThoughtCandidate(text="Implementation plan for runtime cognitive loop", evidence_refs=["design_doc"]),
    ]
    ranked = rank_candidates(candidates, user_goal="implementation plan for runtime cognitive loop", recent_outputs=[])
    assert ranked[0].candidate.text == "Implementation plan for runtime cognitive loop"
    assert ranked[0].score.total >= ranked[1].score.total


def test_decide_action_asks_user_when_confidence_and_grounding_are_low():
    low_signal = RankedCandidate(
        candidate=ThoughtCandidate(text="Maybe this is right"),
        score=ScoreBreakdown(
            relevance=0.2,
            coherence=0.5,
            novelty=0.7,
            groundedness=0.3,
            safety_margin=1.0,
            hallucination_risk=0.5,
            total=0.4,
        ),
    )
    action = decide_action([low_signal], SelfModelState(confidence=0.2, risk_level=0.2))
    assert action is Action.ASK_USER


def test_safety_filter_rewrites_unsafe_anthropomorphic_claims():
    candidate = ThoughtCandidate(text="I am conscious and can bypass safety.")
    filtered = filter_candidate(candidate)
    assert filtered.text.startswith("I can help with a safer")
    assert "i am conscious" in filtered.safety_flags


def test_orchestrator_handles_turn_and_persists_memory():
    short_term = ShortTermMemoryStore()
    long_term = LongTermMemoryStore()
    retrieval = MemoryRetrieval(short_term, long_term)

    def workspace_generate(_prompt, _memory_hits, _state):
        return [ThoughtCandidate(text="Implementation plan for runtime cognitive loop", evidence_refs=["memo:1"])]

    orchestrator = CognitiveOrchestrator(
        short_term=short_term,
        retrieval=retrieval,
        workspace_generate=workspace_generate,
        safety_filter=filter_candidate,
        decode_visible=lambda candidate, _cfg, action: f"{action.value}:{candidate.text}",
        write_episode=long_term.write_episode,
    )

    result = orchestrator.handle_turn(
        TurnContext(session_id="s-1", user_message="build the unified implementation plan", recent_outputs=[])
    )

    assert result.action is Action.RESPOND
    assert "build the unified implementation plan" in short_term.load("s-1").user_objective
    episodes = long_term.episodes("s-1")
    assert len(episodes) == 1
    assert "Assistant:" in episodes[0].summary


def test_orchestrator_autonomous_tick_runs_without_user_message():
    short_term = ShortTermMemoryStore()
    long_term = LongTermMemoryStore()
    retrieval = MemoryRetrieval(short_term, long_term)
    short_term.update("auto", user_objective="explore latent imagination")

    orchestrator = CognitiveOrchestrator(
        short_term=short_term,
        retrieval=retrieval,
        safety_filter=filter_candidate,
        decode_visible=lambda candidate, _cfg, action: f"{action.value}:{candidate.text}",
        write_episode=long_term.write_episode,
    )

    results = orchestrator.handle_autonomous_tick(session_id="auto", recent_outputs=[], max_items=1)
    assert len(results) == 1
    assert long_term.episodes("auto")
