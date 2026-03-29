from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from gguf_dream_lab.backend.atlas.atlas import AtlasStorage, LatentAtlas
from gguf_dream_lab.backend.dream.controller import DreamController
from gguf_dream_lab.backend.runtime.llama_backend import LlamaCppBackend
from gguf_dream_lab.config.io import load_config, save_config
from gguf_dream_lab.config.models import AppConfig
from gguf_dream_lab.logging_utils import configure_logging
from gguf_dream_lab.ui.app import run_ui

app = typer.Typer(help="GGUF Dream Lab CLI")


@app.command("init-config")
def init_config(path: Path = Path("sample_configs/default.json")) -> None:
    cfg = AppConfig()
    save_config(cfg, path)
    print(f"[green]Wrote config to {path}")


@app.command("diagnostics")
def diagnostics(config_path: Path | None = typer.Option(None, "--config")) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.storage_dir / "logs", cfg.log_level)
    runtime = LlamaCppBackend(cfg.runtime)
    runtime.load()
    caps = runtime.capabilities()
    report = {
        "backend": caps.backend_name,
        "supports_embeddings": caps.supports_embeddings,
        "supports_logits_all": caps.supports_logits_all,
        "supports_streaming": caps.supports_streaming,
        "supports_instrumented_latents": caps.supports_instrumented_latents,
        "warnings": caps.warnings,
    }
    print(json.dumps(report, indent=2))


@app.command("benchmark")
def benchmark(config_path: Path | None = typer.Option(None, "--config"), prompt: str = "dreaming in latent space") -> None:
    cfg = load_config(config_path)
    runtime = LlamaCppBackend(cfg.runtime)
    report = runtime.benchmark(prompt=prompt, steps=16)
    print(json.dumps(report, indent=2))


@app.command("atlas-build")
def atlas_build(config_path: Path | None = typer.Option(None, "--config"), steps: int = 24) -> None:
    cfg = load_config(config_path)
    runtime = LlamaCppBackend(cfg.runtime)
    atlas = LatentAtlas()
    controller = DreamController(runtime=runtime, atlas=atlas)
    controller.start(cfg.dream)
    import time

    while controller.state.step_idx < steps:
        time.sleep(0.05)
    controller.stop()
    AtlasStorage(cfg.storage_dir / "atlas").save(atlas)
    print(f"Built atlas with {len(atlas.points)} points")


@app.command("atlas-append")
def atlas_append(config_path: Path | None = typer.Option(None, "--config"), steps: int = 12) -> None:
    cfg = load_config(config_path)
    store = AtlasStorage(cfg.storage_dir / "atlas")
    atlas = store.load()
    runtime = LlamaCppBackend(cfg.runtime)
    controller = DreamController(runtime=runtime, atlas=atlas)
    controller.start(cfg.dream)
    import time

    start = controller.state.step_idx
    while controller.state.step_idx < start + steps:
        time.sleep(0.05)
    controller.stop()
    store.save(atlas)
    print(f"Appended atlas. Total points: {len(atlas.points)}")


@app.command("dream-run")
def dream_run(config_path: Path | None = typer.Option(None, "--config"), seconds: int = 15) -> None:
    cfg = load_config(config_path)
    runtime = LlamaCppBackend(cfg.runtime)
    atlas = LatentAtlas()
    controller = DreamController(runtime=runtime, atlas=atlas)
    controller.start(cfg.dream)
    import time

    time.sleep(seconds)
    controller.stop()
    tick = controller.state.latest_tick
    if tick:
        print(f"Preview: {tick.preview_text}")
        print(f"Committed: {tick.committed_text}")


@app.command("app")
def launch_app(config_path: Path | None = typer.Option(None, "--config"), host: str = "127.0.0.1", port: int = 8050) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.storage_dir / "logs", cfg.log_level)
    run_ui(cfg, host=host, port=port)


if __name__ == "__main__":
    app()
