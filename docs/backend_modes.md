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
- Is hard-gated by instrumentation verification evidence; mode is downgraded if evidence is missing.

### Mandatory verification evidence for `true_latent_instrumented`
The runtime only exposes `true_latent_instrumented` when **all** of the following are present:
- Non-empty captured latent tensor with shape metadata (`ndim`, `size`).
- Verification metadata fields:
  - `backend_variant`
  - `instrumentation_commit`
  - `capture_api`
  - `tensor_dtype`
- Verification result is not marked as stub/synthetic.

If any requirement fails, diagnostics and UI must emit explicit `verification_downgrade_reasons`.

> Important: baseline is useful but **not equivalent** to true latent dreaming.
