from gguf_dream_lab.backend.runtime.manager import RuntimeManager
from gguf_dream_lab.config.models import RuntimeConfig


class DummyRuntime:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.closed = False

    def teardown(self) -> None:
        self.closed = True


def test_runtime_manager_returns_singleton_per_config():
    manager = RuntimeManager(runtime_factory=DummyRuntime)
    cfg = RuntimeConfig(model_path=None, n_ctx=1024)

    runtime_a = manager.get_runtime(cfg)
    runtime_b = manager.get_runtime(cfg)

    assert runtime_a is runtime_b
    diag = manager.diagnostics()
    assert diag.model_load_count == 1
    assert diag.reuse_hits == 1
    assert diag.reuse_hit_rate == 0.5


def test_runtime_manager_teardown_removes_instance_and_closes_runtime():
    manager = RuntimeManager(runtime_factory=DummyRuntime)
    cfg = RuntimeConfig(model_path=None, n_ctx=1024)

    runtime = manager.get_runtime(cfg)
    manager.teardown(cfg)

    assert runtime.closed
    assert manager.diagnostics().active_instances == 0


def test_runtime_manager_can_recreate_after_teardown():
    manager = RuntimeManager(runtime_factory=DummyRuntime)
    cfg = RuntimeConfig(model_path=None, n_ctx=1024)

    first = manager.get_runtime(cfg)
    manager.teardown(cfg)
    second = manager.get_runtime(cfg)

    assert first is not second
    assert manager.diagnostics().model_load_count == 2
