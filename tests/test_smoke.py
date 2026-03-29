from gguf_dream_lab.config.models import AppConfig
from gguf_dream_lab.ui.app import create_dash_app


def test_app_smoke():
    cfg = AppConfig()
    app = create_dash_app(cfg)
    assert app.title == "GGUF Dream Lab"


def test_control_ack_is_not_a_dependency_input():
    cfg = AppConfig()
    app = create_dash_app(cfg)
    for callback in app.callback_map.values():
        inputs = callback.get("inputs", [])
        assert all(inp.get("id") != "control-ack" for inp in inputs)
