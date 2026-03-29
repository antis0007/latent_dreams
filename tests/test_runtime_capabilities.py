import sys
import time
import types

from gguf_dream_lab.backend.runtime.llama_backend import LlamaCppBackend
from gguf_dream_lab.config.models import RuntimeConfig


def test_runtime_capabilities_report_mode_and_sites():
    backend = LlamaCppBackend(RuntimeConfig(model_path=None, instrumented_backend=True))
    caps = backend.capabilities()
    assert caps.supports_streaming
    assert caps.active_mode.value in {"enhanced_latent", "true_latent_instrumented", "baseline_approximate"}
    assert isinstance(caps.capture_sites, list)
    assert isinstance(caps.warnings, list)


def test_runtime_load_timeout_falls_back_to_synthetic(tmp_path):
    model = tmp_path / "fake.gguf"
    model.write_text("not-a-real-model")

    fake_module = types.ModuleType("llama_cpp")

    class SlowLlama:
        def __init__(self, **kwargs):
            time.sleep(0.2)

    fake_module.Llama = SlowLlama
    original = sys.modules.get("llama_cpp")
    sys.modules["llama_cpp"] = fake_module
    try:
        backend = LlamaCppBackend(RuntimeConfig(model_path=model, load_timeout_sec=0.05))
        backend.load()
        caps = backend.capabilities()
    finally:
        if original is None:
            del sys.modules["llama_cpp"]
        else:
            sys.modules["llama_cpp"] = original

    assert backend._llm is None
    assert caps.backend_name == "synthetic-fallback"
    assert any("timed out" in warning for warning in caps.warnings)
