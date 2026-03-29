# GGUF Dream Lab

GGUF Dream Lab is a local-first latent-dreaming research app for GGUF models with **explicit dreaming modes**:

- `baseline_approximate`: token/logit/embedding proxy only (fallback).
- `enhanced_latent`: latent-state-first flow backed by embedding proxies and atlas priors.
- `true_latent_instrumented`: latent-state-first flow with instrumented capture/reinjection hooks.

## Key distinction: dreaming vs text generation

This repo now treats dreaming as **latent state evolution**, with text as a readout:

- **Preview lane** = provisional recall from latent state (`decode_preview_from_latent`).
- **Committed transcript** = crystallized recall after coherence + stability window.
- **Dream process** = atlas-constrained latent drift with phase machine and branch candidates.

Baseline mode remains available but is not presented as equivalent to true latent dreaming.

## Features

- GGUF runtime via `llama-cpp-python` with graceful fallback.
- Runtime capability matrix and capture-site reporting.
- Seed basins: `null_prior`, `introspective`, `narrative`, `memory`, `affective`, `prompt_conditioned`.
- Latent phases: `HYPNAGOGIC`, `SCENE_FORMATION`, `CONSOLIDATION`, `DRIFT_RESET`.
- Atlas as navigation substrate: points, edges, density, neighbors, attractor candidates.
- Session persistence + replayable tick history.
- Dash viewer with trajectory overlays, phase/mode metrics, scrubber, branch/tick controls.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Initialize and edit config:

```bash
gguf-dream-lab init-config
```

Diagnostics:

```bash
gguf-dream-lab diagnostics --config sample_configs/default.json
```

UI:

```bash
gguf-dream-lab app --config sample_configs/default.json --host 127.0.0.1 --port 8050
```

## Docs

- `docs/architecture.md`
- `docs/backend_modes.md`
- `docs/limitations.md`
