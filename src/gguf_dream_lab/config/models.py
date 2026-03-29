from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Basin(str, Enum):
    NULL_PRIOR = "null_prior"
    INTROSPECTIVE = "introspective"
    NARRATIVE = "narrative"
    MEMORY = "memory"
    AFFECTIVE = "affective"
    PROMPT_CONDITIONED = "prompt_conditioned"


class RuntimeConfig(BaseModel):
    model_path: Optional[Path] = None
    n_ctx: int = 4096
    n_gpu_layers: int = 0
    n_batch: int = 512
    n_ubatch: int = 512
    logits_all: bool = True
    embedding: bool = True
    offload_kqv: bool = True
    flash_attn: bool = False
    temperature: float = 0.9
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.05
    seed: int = 42
    prefer_true_latent: bool = True
    instrumented_backend: bool = False
    load_timeout_sec: float = 20.0


class CoherenceWeights(BaseModel):
    entropy: float = 0.20
    density: float = 0.20
    token_stability: float = 0.20
    smoothness: float = 0.20
    branch_agreement: float = 0.10
    known_state_similarity: float = 0.10


class DreamConfig(BaseModel):
    prompt: str = ""
    basin: Basin = Basin.NULL_PRIOR
    tick_hz: float = 1.5
    noise_amplitude: float = 0.15
    coherence_threshold: float = 0.62
    anneal_rate: float = 0.98
    randomness: float = 0.9
    max_preview_len: int = 220
    max_committed_len: int = 2000
    branch_count: int = 1
    preview_decode_cadence: int = 1
    stability_window: int = 4
    run_label: str = "default"
    weights: CoherenceWeights = Field(default_factory=CoherenceWeights)


class AppConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    dream: DreamConfig = Field(default_factory=DreamConfig)
    storage_dir: Path = Path("storage")
    log_level: str = "INFO"
