from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from gguf_dream_lab.backend.atlas.atlas import AtlasStorage, LatentAtlas
from gguf_dream_lab.backend.dream.controller import DreamController
from gguf_dream_lab.backend.runtime.capability_contracts import MODE_CONTRACTS, RuntimeBehaviorSnapshot, validate_mode_contract
from gguf_dream_lab.backend.runtime.manager import get_runtime_manager
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
    runtime_manager = get_runtime_manager()
    runtime = runtime_manager.get_runtime(cfg.runtime)
    try:
        runtime.load()
        caps = runtime.capabilities()
        behavior_snapshot = RuntimeBehaviorSnapshot(
            capture=caps.supports_capture,
            reinject=caps.supports_reinject,
            decode_provenance=caps.supports_decode_provenance,
            control_authority=caps.supports_control_authority,
        )
        contract = validate_mode_contract(caps.active_mode, behavior_snapshot)
        manager_diag = runtime_manager.diagnostics()
        report = {
            "backend": caps.backend_name,
            "supports_embeddings": caps.supports_embeddings,
            "supports_logits_all": caps.supports_logits_all,
            "supports_streaming": caps.supports_streaming,
            "supports_instrumented_latents": caps.supports_instrumented_latents,
            "active_mode": caps.active_mode.value,
            "capture_sites": caps.capture_sites,
            "runtime_manager": {
                "active_instances": manager_diag.active_instances,
                "model_load_count": manager_diag.model_load_count,
                "reuse_hits": manager_diag.reuse_hits,
                "reuse_hit_rate": manager_diag.reuse_hit_rate,
            },
            "instrumentation_verification": {
                "source": caps.instrumentation_verification_source,
                "metadata": caps.instrumentation_verification_metadata,
                "downgrade_reasons": caps.instrumentation_downgrade_reasons,
                "verified_for_true_mode": caps.supports_instrumented_latents,
            },
            "runtime_behaviors": {
                "capture": caps.supports_capture,
                "reinject": caps.supports_reinject,
                "decode_provenance": caps.supports_decode_provenance,
                "control_authority": caps.supports_control_authority,
            },
            "mode_contract": {
                "claimed_mode": contract.claimed_mode.value,
                "effective_mode": contract.effective_mode.value,
                "required_behaviors_for_effective_mode": list(MODE_CONTRACTS[contract.effective_mode]),
                "missing_behaviors_for_claim": contract.missing_behaviors,
                "downgraded": contract.downgraded,
            },
            "warnings": caps.warnings,
        }
        print(json.dumps(report, indent=2))
    finally:
        runtime_manager.teardown(cfg.runtime)


@app.command("benchmark")
def benchmark(config_path: Path | None = typer.Option(None, "--config"), prompt: str = "dreaming in latent space") -> None:
    cfg = load_config(config_path)
    runtime_manager = get_runtime_manager()
    runtime = runtime_manager.get_runtime(cfg.runtime)
    try:
        report = runtime.benchmark(prompt=prompt, steps=16)
        print(json.dumps(report, indent=2))
    finally:
        runtime_manager.teardown(cfg.runtime)


@app.command("atlas-build")
def atlas_build(config_path: Path | None = typer.Option(None, "--config"), steps: int = 24) -> None:
    cfg = load_config(config_path)
    runtime_manager = get_runtime_manager()
    runtime = runtime_manager.get_runtime(cfg.runtime)
    atlas = LatentAtlas()
    controller = DreamController(runtime=runtime, atlas=atlas)
    try:
        controller.start(cfg.dream)
        import time

        while controller.state.step_idx < steps:
            time.sleep(0.05)
        controller.stop()
        AtlasStorage(cfg.storage_dir / "atlas").save(atlas)
        print(f"Built atlas with {len(atlas.points)} points")
    finally:
        runtime_manager.teardown(cfg.runtime)


@app.command("atlas-append")
def atlas_append(config_path: Path | None = typer.Option(None, "--config"), steps: int = 12) -> None:
    cfg = load_config(config_path)
    store = AtlasStorage(cfg.storage_dir / "atlas")
    atlas = store.load()
    runtime_manager = get_runtime_manager()
    runtime = runtime_manager.get_runtime(cfg.runtime)
    controller = DreamController(runtime=runtime, atlas=atlas)
    try:
        controller.start(cfg.dream)
        import time

        start = controller.state.step_idx
        while controller.state.step_idx < start + steps:
            time.sleep(0.05)
        controller.stop()
        store.save(atlas)
        print(f"Appended atlas. Total points: {len(atlas.points)}")
    finally:
        runtime_manager.teardown(cfg.runtime)


@app.command("dream-run")
def dream_run(config_path: Path | None = typer.Option(None, "--config"), seconds: int = 15) -> None:
    cfg = load_config(config_path)
    runtime_manager = get_runtime_manager()
    runtime = runtime_manager.get_runtime(cfg.runtime)
    atlas = LatentAtlas()
    controller = DreamController(runtime=runtime, atlas=atlas)
    try:
        controller.start(cfg.dream)
        import time

        time.sleep(seconds)
        controller.stop()
        tick = controller.state.latest_tick
        if tick:
            print(f"Preview: {tick.preview_text}")
            print(f"Committed: {tick.committed_text}")
    finally:
        runtime_manager.teardown(cfg.runtime)


@app.command("app")
def launch_app(config_path: Path | None = typer.Option(None, "--config"), host: str = "127.0.0.1", port: int = 8050) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg.storage_dir / "logs", cfg.log_level)
    run_ui(cfg, host=host, port=port)


if __name__ == "__main__":
    app()
