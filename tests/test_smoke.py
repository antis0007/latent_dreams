from gguf_dream_lab.config.models import AppConfig
from gguf_dream_lab.ui.app import create_dash_app


def test_app_smoke():
    cfg = AppConfig()
    app = create_dash_app(cfg)
    assert app.title == "GGUF Dream Lab"
