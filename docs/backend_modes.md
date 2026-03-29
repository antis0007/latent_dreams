# Backend Modes

## Baseline GGUF mode (works now)

Uses standard runtime-accessible signals:
- sampled token continuation
- top-token/logprob information
- entropy estimates
- embeddings when available

Latent states are **approximate** and built from these observable summaries.

## Enhanced latent mode (experimental scaffold)

Designed for custom/instrumented llama.cpp forks:
- selected hidden activations
- selected layer outputs
- optional eval callback tensor captures

Not required for baseline functionality.

## Why baseline GGUF dreaming is approximate, and how deeper latent instrumentation can improve it

Stock GGUF runtime APIs generally prioritize fast inference, not stable layer-wise latent extraction. Baseline mode therefore approximates dream state with embeddings/logits/state summaries. This supports practical local dreaming workflows, but cannot fully represent internal per-layer dynamics.

Deeper instrumentation can improve:
- true latent basin seeding from real hidden states
- layer-specific state trajectories
- richer coherence measures from internal geometry
- better branch agreement and manifold-preserving drift
