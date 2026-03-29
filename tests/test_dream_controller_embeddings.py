import numpy as np

from gguf_dream_lab.backend.atlas.atlas import LatentAtlas, StatePoint
from gguf_dream_lab.backend.dream.controller import DreamController
from gguf_dream_lab.config.models import DreamConfig


class _DummyRuntime:
    def load(self) -> None:
        return None


def _controller() -> DreamController:
    return DreamController(runtime=_DummyRuntime(), atlas=LatentAtlas())


def test_normalize_embedding_averages_token_matrix_to_vector():
    controller = _controller()
    matrix = np.arange(24 * 8, dtype=np.float32).reshape(24, 8)

    vec = controller._normalize_embedding(matrix)

    assert vec.shape == (8,)
    np.testing.assert_allclose(vec, matrix.mean(axis=0))


def test_normalize_embedding_uses_fallback_shape_when_missing():
    controller = _controller()
    fallback = np.ones(12, dtype=np.float32)

    vec = controller._normalize_embedding(None, fallback=fallback)

    assert vec.shape == fallback.shape
    assert np.all(vec == 0.0)


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


def test_merge_preview_token_replaces_repeated_tail_instead_of_appending():
    updated = DreamController._merge_preview_token("I I", "everything", max_len=100)
    assert updated == "I everything"


def test_merge_committed_token_skips_duplicate_tail_token():
    updated = DreamController._merge_committed_token("the dream", "dream", max_len=100)
    assert updated == "the dream"


def test_evolve_prompt_includes_recent_context():
    controller = _controller()
    controller.state.preview_text = "I the everything words"
    controller.state.committed_text = "a coherent thought appears"
    cfg = DreamConfig(prompt="seed")

    prompt = controller._evolve_prompt(base_prompt="seed", cfg=cfg)

    assert "seed" in prompt
    assert "coherent thought appears" in prompt
    assert "everything words" in prompt
