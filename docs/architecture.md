# Architecture

## System layers

1. **Runtime (`backend/runtime`)**
   - Baseline GGUF model loading and inference via `llama-cpp-python`
   - Capability detection and benchmark diagnostics
   - Synthetic fallback mode for unsupported environments

2. **Dream controller (`backend/dream`)**
   - Tick-based dream loop (default 1.5 Hz)
   - Basin-conditioned seed prompting + stochastic perturbation
   - Provisional preview lane updated each tick
   - Coherence scorer gating committed transcript

3. **Latent atlas (`backend/atlas`)**
   - State-point accumulation
   - PCA projection, nearest neighbors, local density
   - Clustering and append-mode growth
   - Parquet + joblib persistence

4. **Instrumentation (`backend/instrumentation`)**
   - Baseline no-op adapter
   - Experimental scaffold for custom llama.cpp latent hooks
   - Strict fallback to baseline mode when unavailable

5. **UI (`ui`)**
   - Dash dark-theme dashboard
   - Runtime/dream controls, preview lane, committed lane, metrics panel
   - Latent-state scatter view focused on state points + trajectory

6. **Storage (`storage`)**
   - Atlas cache, logs, sessions

## Dream loop summary

1. Build initial context from prompt + basin + noise schedule.
2. Sample next token/state summary from runtime.
3. Update provisional preview string.
4. Compute coherence score from entropy/density/stability/smoothness.
5. Commit only when coherence threshold passes.
6. Append state to latent atlas and refresh visual geometry.

## Why this is not chatbot generation

The controller treats text as a **readout** from a latent drift process:
- preview text = unstable readout
- committed text = coherence-gated crystallization
