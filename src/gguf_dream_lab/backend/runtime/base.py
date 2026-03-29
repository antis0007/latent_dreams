from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class RuntimeCapabilities:
    supports_embeddings: bool
    supports_logits_all: bool
    supports_streaming: bool
    supports_instrumented_latents: bool
    backend_name: str
    warnings: list[str]


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
