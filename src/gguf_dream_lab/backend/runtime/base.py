from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from gguf_dream_lab.backend.dream.state import DreamMode, LatentState


@dataclass
class RuntimeCapabilities:
    supports_embeddings: bool
    supports_logits_all: bool
    supports_streaming: bool
    supports_instrumented_latents: bool
    supports_true_latent_readout: bool
    backend_name: str
    warnings: list[str]
    capture_sites: list[str] = field(default_factory=list)
    active_mode: DreamMode = DreamMode.BASELINE_APPROXIMATE
    supports_capture: bool = False
    supports_reinject: bool = False
    supports_decode_provenance: bool = False
    supports_control_authority: bool = False


@dataclass
class TokenStep:
    token: str
    logprob: float
    entropy: float
    top_tokens: list[tuple[str, float]]
    embedding: np.ndarray | None = None


class RuntimeBackend(Protocol):
    def load(self) -> None: ...

    def capabilities(self) -> RuntimeCapabilities: ...

    def sample_step(self, prompt: str, max_tokens: int = 16) -> TokenStep: ...

    def embed_text(self, text: str) -> np.ndarray | None: ...

    def benchmark(self, prompt: str, steps: int = 16) -> dict[str, Any]: ...

    def capture_latent_state(self, run_id: str, basin: str, prompt: str) -> LatentState: ...

    def evolve_latent_state(
        self,
        state: LatentState,
        target_vector: np.ndarray,
        noise_scale: float,
        *,
        noise_seed: int | None = None,
    ) -> LatentState: ...

    def decode_prompt_conditioned_preview_from_latent(self, state: LatentState, max_tokens: int = 16) -> str: ...

    def decode_true_latent_readout_preview(self, state: LatentState, max_tokens: int = 16) -> str: ...

    def decode_commit_from_latent(self, state: LatentState, max_tokens: int = 24) -> str: ...
