from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class LatentCapture:
    layer: int
    vector: np.ndarray
    metadata: dict


class InstrumentationAdapter(Protocol):
    def available(self) -> bool: ...

    def capture_sites(self) -> list[str]: ...

    def capture(self, layer: int | None = None) -> LatentCapture | None: ...

    def reinject(self, capture: LatentCapture) -> bool: ...


class BaselineNoopInstrumentation:
    def available(self) -> bool:
        return False

    def capture_sites(self) -> list[str]:
        return []

    def capture(self, layer: int | None = None) -> None:
        return None

    def reinject(self, capture: LatentCapture) -> bool:
        return False


class ExperimentalLlamaForkAdapter:
    """Scaffold for a custom llama.cpp fork with latent capture/reinjection hooks."""

    def __init__(self, enabled: bool = False, sites: list[str] | None = None):
        self.enabled = enabled
        self._sites = sites or ["post_attn_l16", "post_ffn_l24"]

    def available(self) -> bool:
        return self.enabled

    def capture_sites(self) -> list[str]:
        return list(self._sites) if self.enabled else []

    def capture(self, layer: int | None = None) -> LatentCapture | None:
        if not self.enabled:
            return None
        chosen = int(layer) if layer is not None else 16
        rng = np.random.default_rng(chosen)
        vec = rng.normal(size=256).astype(np.float32)
        return LatentCapture(layer=chosen, vector=vec, metadata={"source": "instrumented_stub"})

    def reinject(self, capture: LatentCapture) -> bool:
        return self.enabled and capture.vector.size > 0
