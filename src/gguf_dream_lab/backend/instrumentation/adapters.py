from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class LatentCapture:
    layer: int
    vector: np.ndarray
    metadata: dict


@dataclass
class InstrumentationVerification:
    verified: bool
    source: str
    capture_site_ids: list[str]
    tensor_shape_metadata: dict[str, int]
    verification_metadata: dict[str, str]
    downgrade_reasons: list[str]
    warning: str | None = None


class InstrumentationAdapter(Protocol):
    def available(self) -> bool: ...

    def verify_backend_evidence(self) -> InstrumentationVerification: ...

    def capture_sites(self) -> list[str]: ...

    def capture(self, layer: int | None = None) -> LatentCapture | None: ...

    def reinject(self, capture: LatentCapture) -> bool: ...


class BaselineNoopInstrumentation:
    def available(self) -> bool:
        return False

    def verify_backend_evidence(self) -> InstrumentationVerification:
        return InstrumentationVerification(
            verified=False,
            source="baseline_noop",
            capture_site_ids=[],
            tensor_shape_metadata={},
            verification_metadata={},
            downgrade_reasons=["instrumentation_unavailable"],
            warning="Instrumentation unavailable; backend evidence could not be verified.",
        )

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

    def verify_backend_evidence(self) -> InstrumentationVerification:
        if not self.enabled:
            return InstrumentationVerification(
                verified=False,
                source="instrumented_disabled",
                capture_site_ids=[],
                tensor_shape_metadata={},
                verification_metadata={},
                downgrade_reasons=["instrumented_adapter_disabled"],
                warning="Instrumented adapter disabled; backend evidence could not be verified.",
            )
        capture = self.capture()
        if capture is None:
            return InstrumentationVerification(
                verified=False,
                source="instrumented_capture_missing",
                capture_site_ids=[],
                tensor_shape_metadata={},
                verification_metadata={},
                downgrade_reasons=["instrumented_capture_missing"],
                warning="Instrumented adapter returned no capture during verification.",
            )
        capture_site_id = f"layer_{capture.layer}"
        tensor_shape_metadata = {
            "ndim": int(capture.vector.ndim),
            "size": int(capture.vector.size),
        }
        metadata = {
            "backend_variant": str(capture.metadata.get("backend_variant", "")),
            "instrumentation_commit": str(capture.metadata.get("instrumentation_commit", "")),
            "capture_api": str(capture.metadata.get("capture_api", "")),
            "tensor_dtype": str(capture.vector.dtype),
        }
        missing_metadata = [key for key, value in metadata.items() if not value]
        downgrade_reasons = [f"missing_verification_metadata:{key}" for key in missing_metadata]
        is_stub = str(capture.metadata.get("source", "")) == "instrumented_stub"
        if is_stub:
            return InstrumentationVerification(
                verified=False,
                source="instrumented_stub",
                capture_site_ids=[capture_site_id],
                tensor_shape_metadata=tensor_shape_metadata,
                verification_metadata=metadata,
                downgrade_reasons=["instrumented_stub_capture", *downgrade_reasons],
                warning=(
                    "Instrumentation verification failed: source=instrumented_stub; "
                    "backend evidence is synthetic and true mode promotion is disabled."
                ),
            )
        vector_is_concrete = capture.vector.ndim > 0 and capture.vector.size > 0
        verified = vector_is_concrete and not missing_metadata
        return InstrumentationVerification(
            verified=verified,
            source=str(capture.metadata.get("source", "instrumented_real")),
            capture_site_ids=[capture_site_id],
            tensor_shape_metadata=tensor_shape_metadata,
            verification_metadata=metadata,
            downgrade_reasons=downgrade_reasons + ([] if vector_is_concrete else ["empty_capture_vector"]),
            warning=None if verified else "Instrumented capture evidence incomplete; true mode promotion is disabled.",
        )

    def capture_sites(self) -> list[str]:
        return list(self._sites) if self.enabled else []

    def capture(self, layer: int | None = None) -> LatentCapture | None:
        if not self.enabled:
            return None
        chosen = int(layer) if layer is not None else 16
        rng = np.random.default_rng(chosen)
        vec = rng.normal(size=256).astype(np.float32)
        return LatentCapture(
            layer=chosen,
            vector=vec,
            metadata={
                "source": "instrumented_stub",
                "backend_variant": "llama.cpp.experimental.stub",
                "instrumentation_commit": "",
                "capture_api": "latent_capture_v0",
            },
        )

    def reinject(self, capture: LatentCapture) -> bool:
        return self.enabled and capture.vector.size > 0
