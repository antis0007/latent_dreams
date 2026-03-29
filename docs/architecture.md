# Architecture

## Main components

- `backend/runtime/*`: GGUF runtime + capability matrix + latent capture/reinjection abstraction.
- `backend/dream/state.py`: canonical `LatentState` object, dream modes, dream phases.
- `backend/dream/controller.py`: latent-state-first dream engine, branch candidates, commit gating.
- `backend/atlas/atlas.py`: navigation atlas with nodes, transitions, density, neighbors, attractors.
- `storage/session_store.py`: tick persistence and replay metadata.
- `ui/app.py`: latent-state viewer with scrubbing, phase/mode visibility, preview/commit lanes.

## Dream update mechanics

Each tick computes a candidate latent state from:

1. atlas neighborhood pull,
2. prior latent retention,
3. stochastic excitation (annealed),
4. optional branch agreement (metadata signal).

Then it:

- decodes provisional preview from latent state,
- computes coherence,
- commits only after stability-window + threshold,
- writes node + transition into atlas.
