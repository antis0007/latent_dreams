from __future__ import annotations

import logging
import math
import random
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from gguf_dream_lab.backend.dream.state import DreamMode, LatentSource, LatentState
from gguf_dream_lab.backend.instrumentation.adapters import (
    ExperimentalLlamaForkAdapter,
    InstrumentationAdapter,
    InstrumentationVerification,
    LatentCapture,
)
from gguf_dream_lab.config.models import RuntimeConfig

from .base import RuntimeBackend, RuntimeCapabilities, TokenStep
from .capability_contracts import RuntimeBehaviorSnapshot, validate_mode_contract

logger = logging.getLogger(__name__)


class LlamaCppBackend(RuntimeBackend):
    def __init__(self, config: RuntimeConfig, instrumentation: InstrumentationAdapter | None = None):
        self.config = config
        self._llm = None
        self._loaded = False
        self._llama_error: str | None = None
        self.instrumentation = instrumentation or ExperimentalLlamaForkAdapter(enabled=config.instrumented_backend)

    def load(self) -> None:
        if self._loaded:
            return
        model_path = self.config.model_path
        if model_path is None:
            self._llama_error = "No model path configured; using synthetic backend behavior."
            self._loaded = True
            return
        if not Path(model_path).exists():
            self._llama_error = f"Model path does not exist: {model_path}"
            self._loaded = True
            return
        try:
            from llama_cpp import Llama

            holder: dict[str, Any] = {}
            error_holder: list[Exception] = []

            def _construct() -> None:
                try:
                    holder["llm"] = Llama(
                        model_path=str(model_path),
                        n_ctx=self.config.n_ctx,
                        n_gpu_layers=self.config.n_gpu_layers,
                        n_batch=self.config.n_batch,
                        n_ubatch=self.config.n_ubatch,
                        logits_all=self.config.logits_all,
                        embedding=self.config.embedding,
                        offload_kqv=self.config.offload_kqv,
                        flash_attn=self.config.flash_attn,
                        seed=self.config.seed,
                        verbose=False,
                    )
                except Exception as inner_exc:
                    error_holder.append(inner_exc)

            loader = threading.Thread(target=_construct, daemon=True)
            loader.start()
            loader.join(timeout=max(float(self.config.load_timeout_sec), 0.1))

            if loader.is_alive():
                self._llama_error = (
                    f"Model load timed out after {self.config.load_timeout_sec:.1f}s; "
                    "using synthetic backend behavior."
                )
                logger.warning(self._llama_error)
            elif error_holder:
                raise error_holder[0]
            else:
                self._llm = holder.get("llm")
                logger.info("Loaded GGUF model: %s", model_path)
        except Exception as exc:  # graceful degradation path
            self._llama_error = f"Failed to initialize llama-cpp-python: {exc}"
            logger.warning(self._llama_error)
        self._loaded = True

    def capabilities(self) -> RuntimeCapabilities:
        warnings = []
        if self._llama_error:
            warnings.append(self._llama_error)
        verification: InstrumentationVerification = self.instrumentation.verify_backend_evidence()
        verification_reasons = list(verification.downgrade_reasons)
        supports_true = self.instrumentation.available() and verification.verified
        if self.instrumentation.available() and not verification.verified and not verification_reasons:
            verification_reasons.append("instrumentation_verification_failed")
        if verification.warning:
            warnings.append(verification.warning)
        if verification_reasons:
            warnings.append(
                "True latent mode downgraded: "
                f"source={verification.source}; reasons={verification_reasons}"
            )
        claimed_mode = DreamMode.TRUE_LATENT_INSTRUMENTED if supports_true else DreamMode.ENHANCED_LATENT
        if not self.config.embedding:
            claimed_mode = DreamMode.BASELINE_APPROXIMATE
            warnings.append("Embeddings disabled; falling back to baseline approximate mode.")
        capture_sites = list(dict.fromkeys([*self.instrumentation.capture_sites(), *verification.capture_site_ids]))
        behaviors = RuntimeBehaviorSnapshot(
            capture=bool(self.config.embedding or supports_true),
            reinject=bool(supports_true),
            decode_provenance=True,
            control_authority=bool(self.config.embedding or supports_true),
        )
        contract = validate_mode_contract(claimed_mode, behaviors)
        if contract.downgraded:
            missing = ", ".join(contract.missing_behaviors)
            warnings.append(
                "Capability contract downgrade: "
                f"claimed_mode={contract.claimed_mode.value} -> effective_mode={contract.effective_mode.value}; "
                f"missing=[{missing}]"
            )
        return RuntimeCapabilities(
            supports_embeddings=self.config.embedding,
            supports_logits_all=self.config.logits_all,
            supports_streaming=True,
            supports_instrumented_latents=supports_true,
            supports_true_latent_readout=False,
            backend_name="llama.cpp (via llama-cpp-python)" if self._llm else "synthetic-fallback",
            warnings=warnings,
            capture_sites=capture_sites,
            active_mode=contract.effective_mode,
            supports_capture=behaviors.capture,
            supports_reinject=behaviors.reinject,
            supports_decode_provenance=behaviors.decode_provenance,
            supports_control_authority=behaviors.control_authority,
            instrumentation_verification_source=verification.source,
            instrumentation_verification_metadata=dict(verification.verification_metadata),
            instrumentation_downgrade_reasons=verification_reasons,
        )

    def sample_step(self, prompt: str, max_tokens: int = 16) -> TokenStep:
        self.load()
        if self._llm is None:
            return self._synthetic_step(prompt)

        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
            logprobs=5,
            echo=False,
            stream=False,
        )
        text = out["choices"][0]["text"]
        token = text.strip().split(" ")[0] if text.strip() else "..."
        lp = out["choices"][0].get("logprobs", {})
        top = lp.get("top_logprobs", [{}])
        top_items = list(top[0].items())[:5] if top else []
        probs = np.array([math.exp(v) for _, v in top_items], dtype=float)
        probs = probs / probs.sum() if probs.size else np.array([1.0])
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        embedding = self.embed_text(prompt + " " + text)
        return TokenStep(
            token=token,
            logprob=float(lp.get("token_logprobs", [-1.0])[0] if lp else -1.0),
            entropy=entropy,
            top_tokens=[(k, float(v)) for k, v in top_items],
            embedding=embedding,
        )

    def embed_text(self, text: str) -> np.ndarray | None:
        self.load()
        if self._llm is None:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return rng.normal(size=256).astype(np.float32)
        try:
            vec = self._llm.embed(text)
            return np.array(vec, dtype=np.float32)
        except Exception:
            return None

    def capture_latent_state(self, run_id: str, basin: str, prompt: str) -> LatentState:
        caps = self.capabilities()
        mode = caps.active_mode
        if caps.supports_instrumented_latents:
            cap = self.instrumentation.capture()
            if cap is not None:
                return self._state_from_capture(run_id, basin, mode, cap)
            return LatentState(
                run_id=run_id,
                basin=basin,
                mode=mode,
                latent_vector=np.zeros(256, dtype=np.float32),
                latent_source=LatentSource.STUB_CAPTURE,
                capture_site="instrumentation_stub",
            )
        vec = self.embed_text(prompt)
        if vec is None:
            vec = np.zeros(256, dtype=np.float32)
        return LatentState(
            run_id=run_id,
            basin=basin,
            mode=mode,
            latent_vector=np.asarray(vec, dtype=np.float32),
            latent_source=LatentSource.EMBEDDING_PROXY,
            capture_site=LatentSource.EMBEDDING_PROXY.value,
        )

    def evolve_latent_state(
        self,
        state: LatentState,
        target_vector: np.ndarray,
        noise_scale: float,
        *,
        noise_seed: int | None = None,
    ) -> LatentState:
        target = np.asarray(target_vector, dtype=np.float32)
        if target.shape != state.latent_vector.shape:
            target = np.resize(target, state.latent_vector.shape)
        proposal = 0.7 * state.latent_vector + 0.3 * target
        rng = np.random.default_rng(noise_seed)
        proposal = proposal + rng.normal(scale=noise_scale, size=proposal.shape).astype(np.float32)
        if state.mode == DreamMode.TRUE_LATENT_INSTRUMENTED:
            injected = self.instrumentation.reinject(LatentCapture(layer=state.layer_id or 0, vector=proposal, metadata={}))
            state.metadata["reinject_ok"] = injected
        return state.clone_with_vector(proposal)

    def decode_prompt_conditioned_preview_from_latent(self, state: LatentState, max_tokens: int = 16) -> str:
        self.load()
        if self._llm is not None:
            seed = state.metadata.get("prompt_seed", "")
            context = state.committed_prefix.strip()
            phase = str(state.phase.value).replace("_", " ").lower()
            prompt = (
                f"Dream prompt: {seed}\n"
                f"Dream phase: {phase}\n"
                f"Committed memory: {context or '[none]'}\n"
                "Continue the dream with vivid concrete imagery:"
            )
            try:
                out = self._llm(
                    prompt,
                    max_tokens=max(8, int(max_tokens)),
                    temperature=min(1.35, max(0.2, self.config.temperature + 0.1)),
                    top_k=max(20, int(self.config.top_k)),
                    top_p=min(0.99, max(0.6, self.config.top_p)),
                    repeat_penalty=max(1.0, self.config.repeat_penalty),
                    echo=False,
                    stream=False,
                )
                text = out["choices"][0]["text"].strip()
                if text:
                    return text
            except Exception:
                pass

        hash_seed = abs(hash(state.latent_vector.tobytes()[:64])) % (2**32)
        rng = np.random.default_rng(hash_seed)
        lex = [
            "echo",
            "drift",
            "velvet",
            "signal",
            "memory",
            "city",
            "glass",
            "night",
            "corridor",
            "lantern",
            "rain",
            "threshold",
            "hushed",
            "mirror",
            "orbit",
            "paper",
            "shimmer",
            "voice",
            "shore",
            "clock",
        ]
        count = max(8, min(max_tokens, 28))
        return " ".join(rng.choice(lex, size=count, replace=True).tolist())

    def decode_true_latent_readout_preview(self, state: LatentState, max_tokens: int = 16) -> str:
        # Reserved for instrumented latent-to-token readout once backend hooks expose
        # a true decode path. Until then we deliberately route to prompt-conditioned synthesis.
        return self.decode_prompt_conditioned_preview_from_latent(state, max_tokens=max_tokens)

    def decode_commit_from_latent(self, state: LatentState, max_tokens: int = 24) -> str:
        caps = self.capabilities()
        if caps.supports_instrumented_latents and caps.supports_true_latent_readout:
            state.metadata["commit_source"] = "true_latent_decode"
            return self.decode_true_latent_readout_preview(state, max_tokens=max_tokens)

        state.metadata["commit_source"] = "approx_preview"
        self.load()
        if self._llm is not None:
            seed = state.metadata.get("prompt_seed", "")
            context = state.committed_prefix.strip()
            phase = str(state.phase.value).replace("_", " ").lower()
            prompt = (
                "Synthesize one concise committed memory fragment for a dream journal.\n"
                f"Prompt seed: {seed or '[none]'}\n"
                f"Dream phase: {phase}\n"
                f"Already committed context: {context or '[none]'}\n"
                "Return only the fragment text:"
            )
            try:
                out = self._llm(
                    prompt,
                    max_tokens=max(8, int(max_tokens)),
                    temperature=min(1.15, max(0.15, self.config.temperature)),
                    top_k=max(20, int(self.config.top_k)),
                    top_p=min(0.98, max(0.55, self.config.top_p)),
                    repeat_penalty=max(1.0, self.config.repeat_penalty),
                    echo=False,
                    stream=False,
                )
                text = out["choices"][0]["text"].strip()
                if text:
                    return text
            except Exception:
                pass

        # Fallback synthesis path for synthetic mode: deterministic but independent
        # from preview-token truncation.
        hash_seed = abs(hash(state.latent_vector.tobytes()[64:128])) % (2**32)
        rng = np.random.default_rng(hash_seed)
        lex = [
            "a lantern hums under rain",
            "footsteps fold into velvet static",
            "glass corridors breathe moonlit dust",
            "a paper clock forgets the hour",
            "the shoreline mirrors a distant voice",
            "hushed signals drift through midnight rooms",
            "memory bends around an open threshold",
            "the city exhales in silver echoes",
        ]
        span = max(1, min(3, max_tokens // 8))
        return ". ".join(rng.choice(lex, size=span, replace=False).tolist())

    def benchmark(self, prompt: str, steps: int = 16) -> dict[str, Any]:
        self.load()
        started = time.perf_counter()
        for _ in range(steps):
            self.sample_step(prompt=prompt, max_tokens=8)
        elapsed = time.perf_counter() - started
        return {
            "steps": steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": steps / max(elapsed, 1e-9),
            "backend": self.capabilities().backend_name,
            "active_mode": self.capabilities().active_mode.value,
        }

    def _synthetic_step(self, prompt: str) -> TokenStep:
        lex = ["drift", "echo", "horizon", "memory", "soft", "night", "pulse", "glass", "city", "signal"]
        token = random.choice(lex)
        probs = np.random.default_rng().dirichlet(np.ones(5))
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        top = [(random.choice(lex), float(np.log(max(p, 1e-9)))) for p in probs]
        emb = self.embed_text(f"{prompt} {token}")
        return TokenStep(token=token, logprob=float(np.log(float(max(probs)))), entropy=entropy, top_tokens=top, embedding=emb)

    @staticmethod
    def _state_from_capture(run_id: str, basin: str, mode: DreamMode, cap: LatentCapture) -> LatentState:
        return LatentState(
            run_id=run_id,
            basin=basin,
            mode=mode,
            latent_vector=np.asarray(cap.vector, dtype=np.float32),
            capture_site=f"layer_{cap.layer}",
            latent_source=LatentSource.TRUE_TENSOR_CAPTURE,
            layer_id=cap.layer,
            metadata=dict(cap.metadata),
        )
