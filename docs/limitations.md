# Limitations

- Baseline latent-state geometry is approximate, not full hidden-state introspection.
- `llama-cpp-python` capabilities vary across versions and model metadata.
- Without GPU offload, performance may degrade on consumer CPUs.
- Coherence score is heuristic and should be tuned per model family.
- Current projection is PCA-first; UMAP can be added optionally.
- Branch dreaming is scaffolded via score inputs but not fully parallelized in MVP.
