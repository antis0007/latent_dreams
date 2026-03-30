from gguf_dream_lab.backend.cognition.models import SelfModelState
from gguf_dream_lab.backend.cognition.scheduler import schedule_idle_thoughts
from gguf_dream_lab.backend.memory.long_term import LongTermMemoryStore
from gguf_dream_lab.backend.memory.retrieval import MemoryRetrieval
from gguf_dream_lab.backend.memory.schemas import EpisodicMemoryRecord
from gguf_dream_lab.backend.memory.short_term import ShortTermMemoryStore


def test_memory_retrieval_combines_short_and_long_term_hits():
    short_term = ShortTermMemoryStore()
    long_term = LongTermMemoryStore()
    retrieval = MemoryRetrieval(short_term, long_term)

    short_term.update("s1", user_objective="build a cognition runtime", open_threads=["memory salience policy"])
    long_term.write_episode(
        EpisodicMemoryRecord.create(
            session_id="s1",
            summary="discussed arbitration thresholds for cognition runtime",
            salience=0.9,
            confidence=0.8,
        )
    )

    hits = retrieval.search("cognition runtime memory", "s1", top_k=5)
    assert hits
    assert {hit.source for hit in hits} >= {"short_term", "episodic"}


def test_schedule_idle_thoughts_flags_surface_when_safe_and_salient():
    thoughts = schedule_idle_thoughts(
        self_state=SelfModelState(initiative_budget=0.9, risk_level=0.2),
        memory_hits=[],
        max_items=2,
    )
    assert len(thoughts) == 2
    assert all(thought.salience > 0 for thought in thoughts)
