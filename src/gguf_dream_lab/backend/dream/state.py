from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class DreamMode(str, Enum):
    BASELINE_APPROXIMATE = "baseline_approximate"
    ENHANCED_LATENT = "enhanced_latent"
    TRUE_LATENT_INSTRUMENTED = "true_latent_instrumented"


class DreamPhase(str, Enum):
    HYPNAGOGIC = "HYPNAGOGIC"
    SCENE_FORMATION = "SCENE_FORMATION"
    CONSOLIDATION = "CONSOLIDATION"
    DRIFT_RESET = "DRIFT_RESET"


@dataclass
class LatentState:
    run_id: str
    basin: str
    mode: DreamMode
    latent_vector: np.ndarray
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    capture_site: str = "embedding_proxy"
    layer_id: int | None = None
    auxiliary_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    topk_logits: list[tuple[str, float]] = field(default_factory=list)
    entropy: float = 0.0
    density: float = 0.0
    coherence: float = 0.0
    preview_text: str = ""
    committed_prefix: str = ""
    phase: DreamPhase = DreamPhase.HYPNAGOGIC
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone_with_vector(self, vector: np.ndarray, *, phase: DreamPhase | None = None) -> "LatentState":
        return LatentState(
            run_id=self.run_id,
            basin=self.basin,
            mode=self.mode,
            latent_vector=np.asarray(vector, dtype=np.float32),
            capture_site=self.capture_site,
            layer_id=self.layer_id,
            auxiliary_vectors={k: np.asarray(v, dtype=np.float32) for k, v in self.auxiliary_vectors.items()},
            topk_logits=list(self.topk_logits),
            entropy=self.entropy,
            density=self.density,
            coherence=self.coherence,
            preview_text=self.preview_text,
            committed_prefix=self.committed_prefix,
            phase=phase or self.phase,
            metadata=dict(self.metadata),
        )
