import numpy as np

from gguf_dream_lab.backend.dream.controller import DreamController
from gguf_dream_lab.backend.dream.state import DreamMode, LatentSource, LatentState
from gguf_dream_lab.backend.runtime.llama_backend import LlamaCppBackend
from gguf_dream_lab.config.models import DreamConfig, RuntimeConfig


class _RecorderLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return {"choices": [{"text": "decoded latent scene"}]}


def test_adaptive_noise_respects_floor_and_decreases_with_coherence():
    cfg = DreamConfig(noise_amplitude=0.4, exploration_floor=0.3, coherence_gain=0.8)

    low = DreamController._adaptive_noise(cfg, anneal=1.0, recent_coherence=0.0)
    high = DreamController._adaptive_noise(cfg, anneal=1.0, recent_coherence=1.0)

    assert low > high
    assert high >= cfg.noise_amplitude * cfg.exploration_floor


def test_adaptive_attractor_weight_increases_with_coherence():
    cfg = DreamConfig(attractor_force_weight=0.2)

    low = DreamController._adaptive_attractor_weight(cfg, recent_coherence=0.1)
    high = DreamController._adaptive_attractor_weight(cfg, recent_coherence=0.9)

    assert high > low


def test_llama_preview_prompt_uses_latent_anchors_not_prompt_seed():
    backend = LlamaCppBackend(RuntimeConfig(model_path=None))
    recorder = _RecorderLLM()
    backend._loaded = True
    backend._llm = recorder

    state = LatentState(
        run_id="r1",
        basin="null_prior",
        mode=DreamMode.BASELINE_APPROXIMATE,
        latent_vector=np.array([0.1, -0.8, 0.3, 0.05], dtype=np.float32),
        latent_source=LatentSource.EMBEDDING_PROXY,
        metadata={"prompt_seed": "legacy seed text"},
    )

    backend.decode_approximate_prompt_synthesis_preview(state, max_tokens=12)

    assert recorder.prompts
    sent_prompt = recorder.prompts[-1]
    assert "Latent anchors:" in sent_prompt
    assert "Dream prompt:" not in sent_prompt
