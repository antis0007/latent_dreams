# Limitations

- `true_latent_instrumented` currently depends on a custom instrumentation adapter/fork.
- The provided instrumented adapter is a scaffold/stub, not a production tensor hook.
- Projection plots are for inspection only; control remains in high-dimensional latent vectors.
- Synthetic fallback mode is deterministic-ish for testing and UI demo, not model-faithful dreaming.
- Consumer hardware constraints may require lowering `tick_hz`, branch count, and decode cadence.
