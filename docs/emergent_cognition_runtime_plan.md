# Emergent Cognition Runtime Plan (Frozen Weights)

This document merges the two planning drafts into one implementation plan that is now partially scaffolded under `src/gguf_dream_lab/backend/cognition/`.

## Non-negotiable constraints

- No post-download training or weight updates.
- Adaptation only through runtime control (memory, prompts, decoding, retrieval, tool outputs).
- Internal thought traces remain hidden; user-facing responses are curated outputs.

## Runtime flow

1. **Perception + retrieval**
   - Parse user message.
   - Retrieve short-term and long-term memory hits.
2. **Self-model update**
   - Track intent, confidence, uncertainty, tone, initiative budget, and creative drive.
3. **Hidden workspace rollouts**
   - Generate multiple candidate thought trajectories.
4. **Multi-objective scoring**
   - Relevance, coherence, novelty, groundedness, safety margin.
5. **Arbitration**
   - Decide whether to respond, ask a clarifying question, or reflect internally.
6. **Grounding + safety gates**
   - Enforce claim discipline and policy bounds.
7. **Visible decode**
   - Use mode-dependent decoding controls.
8. **Memory writeback**
   - Persist episodic outcomes and preference signals.

## Current scaffold

Implemented now:

- `cognition/orchestrator.py`: end-to-end turn loop wired to short-term retrieval, self-model update, hidden candidate generation, scoring, arbitration, grounding/safety gates, decode-control selection, and episodic writeback.
- `cognition/workspace.py`: hidden multi-path candidate generator with seed intent families and lightweight mutation steps.
- `cognition/self_model.py`: explicit metacognitive state update (intent, confidence, uncertainty, initiative, creative drive, risk).
- `cognition/grounding.py` + `cognition/safety_gate.py`: factual-vs-speculative labeling and unsafe-pattern filtering/rewrite.
- `cognition/scheduler.py`: opt-in idle thought scheduler with salience and surfacing flags.
- `memory/`: short-term scratchpad store, long-term episodic/semantic store, and retrieval ranker (`short_term.py`, `long_term.py`, `retrieval.py`, `schemas.py`).
- `eval/`: turn metric structures and scenario harness (`metrics.py`, `harness.py`).

Still planned next:

- richer candidate recombination (A/B/C merge) and anti-loop controls in workspace rollouts.
- stronger grounding checks tied to runtime tools/evidence references.
- contradiction detection and semantic profile conflict resolution.
- dashboard/reporting integration for evaluation metrics over long sessions.

## Suggested next milestones

### Milestone 1 — Wire into existing runtime manager

- Instantiate `CognitiveOrchestrator` from runtime manager.
- Provide adapters for current backend decode path and session storage.
- Log score breakdowns and arbitration decisions for observability.

### Milestone 2 — Memory implementation

- Add short-term scratchpad fields: objective, constraints, unresolved threads.
- Add long-term episodic schema: timestamp, summary, embedding, salience, confidence tags.
- Add retrieval and salience decay strategy.

### Milestone 3 — Creative exploration upgrade

- Multi-seed intent families (`analyze`, `create`, `ask`, `reflect`).
- Candidate recombination and anti-repetition regularization.
- Relevance floor enforcement to prevent creativity drift.

### Milestone 4 — Evaluation and hardening

- Track novelty, coherence, relevance, consistency, groundedness, and initiative quality.
- Add anti-loop caps (iterations/time/tokens).
- Add hallucination-risk thresholding and uncertainty labeling.

## Success criteria

With frozen weights, the runtime should:

- Show productive initiative without derailing user intent.
- Increase response diversity without sacrificing coherence.
- Maintain grounding and policy compliance.
- Preserve cross-turn consistency through memory.
