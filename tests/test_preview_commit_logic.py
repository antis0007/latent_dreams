from collections import deque

from gguf_dream_lab.backend.dream.controller import DreamController


class _Cfg:
    coherence_threshold = 0.6
    stability_window = 3


def test_commit_gate_on_stability_window():
    stable = deque(["echo one", "echo one", "echo one"], maxlen=3)
    unstable = deque(["echo one", "echo two", "echo three"], maxlen=3)

    assert DreamController._should_commit(0.7, stable, _Cfg())
    assert not DreamController._should_commit(0.7, unstable, _Cfg())
