# Autonomous Latent Dreaming with Frozen Local LLMs

This guide describes a realistic implementation strategy for autonomous "dreaming" behavior using a frozen local LLM (no post-download training).

## What is feasible

- **Feasible now:** continuous internal rollouts, latent/state drift, memory consolidation, self-model updates, and periodic text snapshots.
- **Not directly feasible with standard LLM APIs:** free-running neuron-level activation loops without tokenization. Most local runtimes expose token generation and optional embeddings, not arbitrary hidden-state control.

## Practical digital analogue to dreaming

Use a closed runtime loop with no user prompt requirement:

1. **Autonomous seed generation**
   - Produce internal "dream seeds" from scheduler prompts, memory residues, and low-amplitude random perturbations.
2. **Hidden workspace rollouts**
   - Generate multiple candidate trajectories from each seed.
3. **Scoring + arbitration**
   - Keep trajectories balancing novelty, coherence, relevance to persistent goals, grounding, and safety.
4. **Memory writeback**
   - Store high-salience episodes and semantic preference updates.
5. **Periodic decode snapshots**
   - Decode only selected latent summaries into text logs for interpretability.

## Persistent self-model (bounded)

Maintain a continuously updated internal state:

- intent
- confidence calibration
- uncertainty type
- initiative budget
- creative drive
- risk level

This gives a stable behavioral identity over time without claiming literal consciousness.

## "No prompting" mode

In autonomous mode, prompts become internal system-generated thought seeds instead of user input:

- "What unresolved pattern matters most now?"
- "Which memory fragment should be consolidated?"
- "Generate one novel but coherent scene variant."

This satisfies operational "no user prompt" while preserving deterministic control and safety.

## Control sliders for UI

Useful operator controls:

- novelty pressure
- coherence pressure
- grounding strictness
- initiative rate
- idle-thought cadence
- safety strictness

## Safety and policy boundaries

- Filter anthropomorphic overclaims (e.g., "I am conscious").
- Label unsupported factual content as speculative.
- Keep risky autonomous outputs internal unless explicitly approved for display.

## Success criteria

A good autonomous dream runtime should:

- run continuously without user input,
- produce varied but coherent internal trajectories,
- retain long-horizon consistency via memory,
- expose interpretable snapshots,
- and remain policy-aligned.
