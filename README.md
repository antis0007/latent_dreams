# GGUF Dream Lab

GGUF Dream Lab is a local-first latent-dreaming research app for GGUF models (including 4-bit quants) that separates:

- **Provisional dream preview** (live, unstable, ghost-like)
- **Committed dream transcript** (only when coherence is high)

It is **not** a chatbot wrapper; it is a dream-controller + latent atlas tool with baseline and enhanced backend modes.

## Features (MVP)

- Load any local GGUF model path via `llama-cpp-python` (graceful synthetic fallback if unavailable)
- Runtime tuning for consumer hardware (`n_ctx`, `n_gpu_layers`, `n_batch`, `n_ubatch`, `logits_all`, `embedding`, `offload_kqv`, `flash_attn`)
- Seed basins: `null_prior`, `introspective`, `narrative`, `memory`, `affective_style`, `custom`
- Realtime dream loop at configurable Hz
- Coherence-gated commit lane with explicit weighted score components
- Latent atlas cache with append mode, projections, neighborhood density, clustering
- Dash UI with dark theme, state metrics, latent trajectory visualization
- CLI for diagnostics, benchmark, atlas build/append, dream run, app launch

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
# Optional local GGUF runtime extras:
pip install -e .[llm]
```

Create a sample config:

```bash
gguf-dream-lab init-config
```

Edit `sample_configs/default.json` and set `runtime.model_path` to your `.gguf` file.

Run diagnostics:

```bash
gguf-dream-lab diagnostics --config sample_configs/default.json
```

Launch UI:

```bash
gguf-dream-lab app --config sample_configs/default.json --host 127.0.0.1 --port 8050
```

## Commands

- `gguf-dream-lab init-config`
- `gguf-dream-lab diagnostics --config ...`
- `gguf-dream-lab benchmark --config ...`
- `gguf-dream-lab atlas-build --config ...`
- `gguf-dream-lab atlas-append --config ...`
- `gguf-dream-lab dream-run --config ...`
- `gguf-dream-lab app --config ...`

## Windows-focused setup

Use the helper script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

## Baseline vs enhanced latent modes

See:
- `docs/architecture.md`
- `docs/backend_modes.md`
- `docs/limitations.md`

## Sample prompts

See `sample_configs/sample_prompts.md`.
