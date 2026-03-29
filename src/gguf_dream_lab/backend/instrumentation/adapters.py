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

    def capture(self, layer: int | None = None) -> LatentCapture | None: ...


class BaselineNoopInstrumentation:
    def available(self) -> bool:
        return False

    def capture(self, layer: int | None = None) -> None:
        return None


class ExperimentalLlamaForkAdapter:
    """Scaffold for future instrumented llama.cpp backend integration."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def available(self) -> bool:
        return self.enabled

    def capture(self, layer: int | None = None) -> LatentCapture | None:
        # TODO: Hook custom eval callback / internal tensor capture.
        return None
