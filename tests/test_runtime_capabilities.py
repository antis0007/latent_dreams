from gguf_dream_lab.backend.runtime.llama_backend import LlamaCppBackend
from gguf_dream_lab.config.models import RuntimeConfig


def test_runtime_capabilities_report_mode_and_sites():
    backend = LlamaCppBackend(RuntimeConfig(model_path=None, instrumented_backend=True))
    caps = backend.capabilities()
    assert caps.supports_streaming
    assert caps.active_mode.value in {"enhanced_latent", "true_latent_instrumented", "baseline_approximate"}
    assert isinstance(caps.capture_sites, list)
    assert isinstance(caps.warnings, list)
