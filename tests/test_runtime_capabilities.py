from gguf_dream_lab.backend.runtime.llama_backend import LlamaCppBackend
from gguf_dream_lab.config.models import RuntimeConfig


def test_runtime_capabilities_report():
    backend = LlamaCppBackend(RuntimeConfig(model_path=None))
    caps = backend.capabilities()
    assert caps.supports_streaming
    assert isinstance(caps.warnings, list)
