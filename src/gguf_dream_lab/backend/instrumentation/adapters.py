from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

DOCUMENTED_CAPTURE_SITES = ["post_attn_l16", "post_ffn_l24"]


@dataclass
class LatentCapture:
    layer: int
    site_id: str
    vector: np.ndarray
    tensors: dict[str, np.ndarray]
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

    def bind_backend(self, backend: object | None) -> None: ...

    def verify_backend_evidence(self) -> InstrumentationVerification: ...

    def capture_sites(self) -> list[str]: ...

    def capture(self, layer: int | None = None) -> LatentCapture | None: ...

    def reinject(self, capture: LatentCapture) -> bool: ...

    def supports_true_readout(self) -> bool: ...

    def decode_conditioned_preview(self, capture: LatentCapture, max_tokens: int = 16) -> str | None: ...


class BaselineNoopInstrumentation:
    def available(self) -> bool:
        return False

    def bind_backend(self, backend: object | None) -> None:
        del backend

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

    def supports_true_readout(self) -> bool:
        return False

    def decode_conditioned_preview(self, capture: LatentCapture, max_tokens: int = 16) -> None:
        del capture, max_tokens
        return None


class ExperimentalLlamaForkAdapter:
    """Scaffold for a custom llama.cpp fork with latent capture/reinjection hooks."""

    def __init__(self, enabled: bool = False, sites: list[str] | None = None):
        self.enabled = enabled
        self._sites = sites or list(DOCUMENTED_CAPTURE_SITES)
        self._backend: object | None = None

    def available(self) -> bool:
        return self.enabled

    def bind_backend(self, backend: object | None) -> None:
        self._backend = backend

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
        capture_site_id = capture.site_id
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
        vector_is_concrete = capture.vector.ndim > 0 and capture.vector.size > 0
        verified = vector_is_concrete
        return InstrumentationVerification(
            verified=verified,
            source=str(capture.metadata.get("source", "instrumented_real")),
            capture_site_ids=[capture_site_id],
            tensor_shape_metadata=tensor_shape_metadata,
            verification_metadata=metadata,
            downgrade_reasons=downgrade_reasons + ([] if vector_is_concrete else ["empty_capture_vector"]),
            warning=None if verified else "Instrumented capture evidence incomplete; latent mode promotion is disabled.",
        )

    def capture_sites(self) -> list[str]:
        return list(self._sites) if self.enabled else []

    def capture(self, layer: int | None = None) -> LatentCapture | None:
        if not self.enabled:
            return None
        backend_capture = self._capture_from_bound_backend(layer)
        if backend_capture is not None:
            return backend_capture
        chosen = int(layer) if layer is not None else 16
        site_id = self._sites[0] if self._sites else f"layer_{chosen}"
        rng = np.random.default_rng(chosen)
        vec = rng.normal(size=256).astype(np.float32)
        tensors = {
            "residual_stream": vec.copy(),
            "ffn_gate": rng.normal(size=256).astype(np.float32),
        }
        return LatentCapture(
            layer=chosen,
            site_id=site_id,
            vector=vec,
            tensors=tensors,
            metadata={
                "source": "instrumented_stub",
                "backend_variant": "llama.cpp.experimental.stub",
                "instrumentation_commit": "",
                "capture_api": "latent_capture_v0",
            },
        )

    def reinject(self, capture: LatentCapture) -> bool:
        return self.enabled and capture.vector.size > 0

    def supports_true_readout(self) -> bool:
        if not self.enabled:
            return False
        return self._backend is not None and any(
            hasattr(self._backend, attr) for attr in ("decode_from_latent", "decode_conditioned_latent")
        )

    def decode_conditioned_preview(self, capture: LatentCapture, max_tokens: int = 16) -> str | None:
        if not self.enabled:
            return None
        if self._backend is not None:
            if hasattr(self._backend, "decode_from_latent"):
                return str(self._backend.decode_from_latent(capture.vector, capture.tensors, max_tokens=max_tokens))
            if hasattr(self._backend, "decode_conditioned_latent"):
                return str(
                    self._backend.decode_conditioned_latent(
                        vector=capture.vector,
                        tensors=capture.tensors,
                        max_tokens=max_tokens,
                    )
                )
        return None

    def _capture_from_bound_backend(self, layer: int | None = None) -> LatentCapture | None:
        if self._backend is None:
            return None
        raw = None
        if hasattr(self._backend, "capture_latent"):
            raw = self._backend.capture_latent(layer=layer, sites=self._sites)
        elif hasattr(self._backend, "latent_capture"):
            raw = self._backend.latent_capture(layer=layer, sites=self._sites)
        if not isinstance(raw, dict):
            return None

        vector = raw.get("latent_vector", raw.get("vector"))
        if vector is None:
            return None
        tensor_map: dict[str, np.ndarray] = {}
        for key, value in dict(raw.get("tensors", {})).items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.size:
                tensor_map[str(key)] = arr
        layer_id = int(raw.get("layer", layer if layer is not None else 16))
        site_id = str(raw.get("site_id", self._sites[0] if self._sites else f"layer_{layer_id}"))
        metadata = dict(raw.get("metadata", {}))
        return LatentCapture(
            layer=layer_id,
            site_id=site_id,
            vector=np.asarray(vector, dtype=np.float32),
            tensors=tensor_map,
            metadata=metadata,
        )
