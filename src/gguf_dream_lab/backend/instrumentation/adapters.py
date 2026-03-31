from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

DOCUMENTED_CAPTURE_SITES = ["post_attn_l16", "post_ffn_l24"]
LATENT_CAPTURE_API = "latent_capture_v1"
LATENT_CAPTURE_API_VERSION = 1
MANDATORY_METADATA_KEYS = ("backend_variant", "instrumentation_commit", "capture_api", "tensor_dtype")


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
    contract_version: int | None = None
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

    def __init__(self, enabled: bool = False, sites: list[str] | None = None, *, strict: bool = False):
        self.enabled = enabled
        self._sites = sites or list(DOCUMENTED_CAPTURE_SITES)
        self._backend: object | None = None
        self._strict = strict

    def available(self) -> bool:
        return self.enabled

    def bind_backend(self, backend: object | None) -> None:
        self._backend = backend
        self._install_stable_api_aliases()

    def verify_backend_evidence(self) -> InstrumentationVerification:
        if not self.enabled:
            return InstrumentationVerification(
                verified=False,
                source="instrumented_disabled",
                capture_site_ids=[],
                tensor_shape_metadata={},
                verification_metadata={},
                downgrade_reasons=["instrumented_adapter_disabled"],
                contract_version=None,
                warning="Instrumented adapter disabled; backend evidence could not be verified.",
            )
        contract = self._read_backend_contract()
        contract_reasons = self._contract_downgrade_reasons(contract)
        capture = self.capture()
        if capture is None:
            reasons = ["instrumented_capture_missing", *contract_reasons]
            return InstrumentationVerification(
                verified=False,
                source="instrumented_capture_missing",
                capture_site_ids=[],
                tensor_shape_metadata={},
                verification_metadata={},
                downgrade_reasons=reasons,
                contract_version=contract.get("api_version") if isinstance(contract, dict) else None,
                warning="Instrumented adapter returned no capture during verification.",
            )
        capture_site_id = capture.site_id
        metadata = self._normalized_metadata(capture.metadata, capture.vector)
        tensor_shape_metadata = {
            "ndim": int(capture.vector.ndim),
            "size": int(capture.vector.size),
        }
        missing_metadata = [key for key in MANDATORY_METADATA_KEYS if not metadata.get(key)]
        downgrade_reasons = [f"missing_verification_metadata:{key}" for key in missing_metadata]
        vector_is_concrete = capture.vector.ndim > 0 and capture.vector.size > 0
        synthetic_source = str(metadata.get("source", "")).endswith("stub")
        if synthetic_source:
            downgrade_reasons.append("instrumented_stub_capture")
        downgrade_reasons.extend(contract_reasons)
        verified = vector_is_concrete and not synthetic_source and not contract_reasons
        return InstrumentationVerification(
            verified=verified,
            source=str(metadata.get("source", "instrumented_real")),
            capture_site_ids=[capture_site_id],
            tensor_shape_metadata=tensor_shape_metadata,
            verification_metadata=metadata,
            downgrade_reasons=downgrade_reasons + ([] if vector_is_concrete else ["empty_capture_vector"]),
            contract_version=contract.get("api_version") if isinstance(contract, dict) else None,
            warning=None if verified else "Instrumented capture evidence incomplete; latent mode promotion is disabled.",
        )

    def capture_sites(self) -> list[str]:
        return list(self._sites) if self.enabled else []

    def capture(self, layer: int | None = None) -> LatentCapture | None:
        if not self.enabled:
            return None
        if self._strict and self._backend is None:
            raise RuntimeError("Instrumented backend is enabled in production mode, but no backend was bound for capture.")
        backend_capture = self._capture_from_bound_backend(layer)
        if backend_capture is not None:
            return backend_capture
        if self._strict:
            raise RuntimeError(
                "Instrumented backend capture hooks are missing (expected capture_latent/latent_capture). "
                "Disable production_mode for synthetic testing."
            )
        return None

    def reinject(self, capture: LatentCapture) -> bool:
        if not self.enabled or capture.vector.size <= 0:
            return False
        if self._backend is None:
            return False
        payload = {
            "layer": int(capture.layer),
            "site_id": str(capture.site_id),
            "latent_vector": np.asarray(capture.vector, dtype=np.float32),
            "tensors": {k: np.asarray(v, dtype=np.float32) for k, v in capture.tensors.items()},
            "metadata": dict(capture.metadata),
        }
        fn = None
        if hasattr(self._backend, "reinject_latent"):
            fn = getattr(self._backend, "reinject_latent")
        elif hasattr(self._backend, "latent_reinject"):
            fn = getattr(self._backend, "latent_reinject")
        if callable(fn):
            return bool(fn(payload=payload))
        return False

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

    def _install_stable_api_aliases(self) -> None:
        """Expose stable latent API symbols on the bound python backend object.

        This is the python-layer bridge expected by the runtime:
        - capture_latent / latent_capture
        - decode_from_latent
        - reinject_latent
        - get_latent_api_contract
        """
        if self._backend is None:
            return
        backend = self._backend
        capture_fn = self._resolve_callable(("capture_latent", "latent_capture", "capture_layer_latent"))
        decode_fn = self._resolve_callable(("decode_from_latent", "decode_conditioned_latent", "latent_decode"))
        reinject_fn = self._resolve_callable(("reinject_latent", "latent_reinject", "inject_latent"))

        if capture_fn is not None:
            if not callable(getattr(backend, "capture_latent", None)):
                setattr(backend, "capture_latent", lambda layer=None, sites=None: self._call_with_fallback(capture_fn, layer=layer, sites=sites))
            if not callable(getattr(backend, "latent_capture", None)):
                setattr(backend, "latent_capture", lambda layer=None, sites=None: self._call_with_fallback(capture_fn, layer=layer, sites=sites))
        if decode_fn is not None and not callable(getattr(backend, "decode_from_latent", None)):
            setattr(
                backend,
                "decode_from_latent",
                lambda vector, tensors, max_tokens=16: self._call_with_fallback(
                    decode_fn,
                    vector=vector,
                    tensors=tensors,
                    max_tokens=max_tokens,
                ),
            )
        if reinject_fn is not None and not callable(getattr(backend, "reinject_latent", None)):
            setattr(backend, "reinject_latent", lambda payload: self._call_with_fallback(reinject_fn, payload=payload))

        if not callable(getattr(backend, "get_latent_api_contract", None)):
            setattr(backend, "get_latent_api_contract", self._synthesized_contract)

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
        metadata = self._normalized_metadata(raw.get("metadata", {}), np.asarray(vector, dtype=np.float32))
        return LatentCapture(
            layer=layer_id,
            site_id=site_id,
            vector=np.asarray(vector, dtype=np.float32),
            tensors=tensor_map,
            metadata=metadata,
        )

    @staticmethod
    def _normalized_metadata(metadata: dict, vector: np.ndarray) -> dict[str, str]:
        normalized = {str(k): str(v) for k, v in dict(metadata).items() if v is not None}
        normalized.setdefault("capture_api", LATENT_CAPTURE_API)
        normalized.setdefault("tensor_dtype", str(np.asarray(vector, dtype=np.float32).dtype))
        normalized.setdefault("backend_variant", "llama.cpp.instrumented")
        normalized.setdefault("instrumentation_commit", "unknown")
        normalized.setdefault("source", "instrumented_real")
        return normalized

    def _read_backend_contract(self) -> dict:
        if self._backend is None:
            return {}
        if hasattr(self._backend, "latent_api_contract"):
            contract = getattr(self._backend, "latent_api_contract")
            if isinstance(contract, dict):
                return contract
        if hasattr(self._backend, "get_latent_api_contract"):
            fn = getattr(self._backend, "get_latent_api_contract")
            if callable(fn):
                candidate = fn()
                if isinstance(candidate, dict):
                    return candidate
        return self._synthesized_contract()

    def _synthesized_contract(self) -> dict[str, str | int]:
        return {
            "capture_api": LATENT_CAPTURE_API,
            "api_version": LATENT_CAPTURE_API_VERSION,
            "capture_fn": "capture_latent" if callable(self._resolve_callable(("capture_latent", "latent_capture"))) else "",
            "decode_fn": "decode_from_latent"
            if callable(self._resolve_callable(("decode_from_latent", "decode_conditioned_latent")))
            else "",
            "reinject_fn": "reinject_latent"
            if callable(self._resolve_callable(("reinject_latent", "latent_reinject")))
            else "",
        }

    def _resolve_callable(self, names: tuple[str, ...]) -> object | None:
        if self._backend is None:
            return None
        for name in names:
            candidate = getattr(self._backend, name, None)
            if callable(candidate):
                return candidate
        return None

    @staticmethod
    def _call_with_fallback(fn: object, **kwargs):
        if not callable(fn):
            return None
        try:
            return fn(**kwargs)
        except TypeError:
            ordered = [kwargs[key] for key in ("vector", "tensors", "max_tokens") if key in kwargs]
            if ordered:
                return fn(*ordered)
            if "payload" in kwargs:
                return fn(kwargs["payload"])
            if "layer" in kwargs or "sites" in kwargs:
                layer = kwargs.get("layer")
                sites = kwargs.get("sites")
                if sites is not None:
                    return fn(layer, sites)
                return fn(layer)
            raise

    def _contract_downgrade_reasons(self, contract: dict) -> list[str]:
        if not contract:
            return ["latent_contract_missing"]
        reasons: list[str] = []
        api_name = str(contract.get("capture_api", ""))
        if api_name != LATENT_CAPTURE_API:
            reasons.append(f"latent_contract_api_mismatch:{api_name or 'missing'}")
        api_version = contract.get("api_version")
        if not isinstance(api_version, int):
            reasons.append("latent_contract_version_missing")
        elif api_version != LATENT_CAPTURE_API_VERSION:
            reasons.append(f"latent_contract_version_mismatch:{api_version}")
        for fn_key in ("capture_fn", "decode_fn", "reinject_fn"):
            if not str(contract.get(fn_key, "")):
                reasons.append(f"latent_contract_missing:{fn_key}")
        return reasons
