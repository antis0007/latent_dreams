from gguf_dream_lab.backend.runtime.llama_backend import LlamaCppBackend
from gguf_dream_lab.config.models import RuntimeConfig


def test_missing_model_path_fallback():
    backend = LlamaCppBackend(RuntimeConfig(model_path=None))
    step = backend.sample_step("test")
    assert step.token
    assert step.embedding is not None
