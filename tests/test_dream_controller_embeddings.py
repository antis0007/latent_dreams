from collections import deque

import numpy as np

from gguf_dream_lab.backend.atlas.atlas import LatentAtlas, StatePoint
from gguf_dream_lab.backend.dream.controller import DreamController
from gguf_dream_lab.backend.dream.state import DreamMode, LatentState


class _DummyRuntime:
    def load(self) -> None:
        return None

    def capture_latent_state(self, run_id: str, basin: str, prompt: str) -> LatentState:
        return LatentState(run_id=run_id, basin=basin, mode=DreamMode.ENHANCED_LATENT, latent_vector=np.ones(8, dtype=np.float32))

    def evolve_latent_state(self, state: LatentState, target_vector: np.ndarray, noise_scale: float) -> LatentState:
        return state.clone_with_vector(target_vector)

    def decode_preview_from_latent(self, state: LatentState, max_tokens: int = 16) -> str:
        return "alpha beta gamma"

    def decode_commit_from_latent(self, state: LatentState, max_tokens: int = 24) -> str:
        return "alpha beta"


def _controller() -> DreamController:
    return DreamController(runtime=_DummyRuntime(), atlas=LatentAtlas())


def test_estimate_local_density_returns_zero_for_shape_mismatch():
    controller = _controller()
    controller.atlas.append_points(
        [
            StatePoint(
                state_id="s0",
                run_id="r0",
                run_label="run",
                basin="narrative",
                step_idx=0,
                preview="",
                committed="",
                coherence=0.0,
                entropy=0.0,
                embedding=np.ones(16, dtype=np.float32),
            )
        ]
    )

    density = controller._estimate_local_density(np.ones(8, dtype=np.float32))

    assert density == 0.0


def test_should_commit_requires_threshold_and_stability_window():
    history = deque(["a b", "a b", "a b", "a b"], maxlen=4)

    class _Cfg:
        coherence_threshold = 0.6
        stability_window = 4

    assert DreamController._should_commit(0.7, history, _Cfg())
    assert not DreamController._should_commit(0.3, history, _Cfg())


def test_merge_committed_chunk_appends_unique_chunk():
    updated = DreamController._merge_committed_chunk("the dream", "becomes stable", max_len=200)
    assert "becomes stable" in updated


def test_smoothness_handles_shape_mismatch_without_crashing():
    prev = np.ones(8, dtype=np.float32)
    cur = np.ones(16, dtype=np.float32)

    smoothness = DreamController._smoothness(prev, cur)

    assert 0.0 <= smoothness <= 1.0


def test_align_vector_shape_pads_and_preserves_prefix():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ref = np.zeros(5, dtype=np.float32)

    aligned = DreamController._align_vector_shape(vec, ref)

    assert aligned.shape == ref.shape
    assert np.allclose(aligned[:3], vec)
    assert np.allclose(aligned[3:], 0.0)


def test_smoothness_flattens_row_vectors():
    prev = np.ones((1, 4), dtype=np.float32)
    cur = np.ones((1, 4), dtype=np.float32)

    smoothness = DreamController._smoothness(prev, cur)

    assert 0.99 <= smoothness <= 1.0


def test_neighbor_vector_ignores_self_neighbor():
    controller = _controller()
    current = np.ones(8, dtype=np.float32)
    controller.atlas.append_points(
        [
            StatePoint(
                state_id="s0",
                run_id="r0",
                run_label="run",
                basin="narrative",
                step_idx=0,
                preview="",
                committed="",
                coherence=0.0,
                entropy=0.0,
                embedding=current.copy(),
            )
        ]
    )

    neighbor = controller._neighbor_vector(current)

    assert np.allclose(neighbor, current)
