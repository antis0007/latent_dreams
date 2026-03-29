# Backend Capability Modes

## `baseline_approximate`
- No latent capture sites.
- Uses token/logit/embedding summaries as proxies.
- Supported everywhere as fallback.

## `enhanced_latent`
- Latent-state-first dream loop and atlas controls.
- Uses embedding-derived latent vectors when instrumentation is unavailable.
- Preview/commit decoding still derived from latent state object.

## `true_latent_instrumented`
- Requires instrumented backend adapter support.
- Captures internal tensors into `LatentState` and can reinject conditioning vectors.
- Reports capture sites in diagnostics and UI.

> Important: baseline is useful but **not equivalent** to true latent dreaming.
