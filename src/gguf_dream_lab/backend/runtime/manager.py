from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from gguf_dream_lab.config.models import RuntimeConfig

from .llama_backend import LlamaCppBackend


@dataclass(frozen=True)
class RuntimeManagerDiagnostics:
    active_instances: int
    model_load_count: int
    reuse_hits: int
    reuse_hit_rate: float


class RuntimeManager:
    """Thread-safe runtime factory with singleton-per-config semantics."""

    def __init__(self, runtime_factory: Callable[[RuntimeConfig], LlamaCppBackend] | None = None):
        self._runtime_factory = runtime_factory or LlamaCppBackend
        self._instances: dict[str, LlamaCppBackend] = {}
        self._lock = threading.Lock()
        self._model_load_count = 0
        self._reuse_hits = 0

    @staticmethod
    def _config_key(config: RuntimeConfig) -> str:
        return config.model_dump_json(exclude_none=False, by_alias=True)

    def get_runtime(self, config: RuntimeConfig) -> LlamaCppBackend:
        key = self._config_key(config)
        with self._lock:
            existing = self._instances.get(key)
            if existing is not None:
                self._reuse_hits += 1
                return existing
            runtime = self._runtime_factory(config)
            self._instances[key] = runtime
            self._model_load_count += 1
            return runtime

    def teardown(self, config: RuntimeConfig | None = None) -> None:
        with self._lock:
            if config is None:
                runtimes = list(self._instances.values())
                self._instances.clear()
            else:
                runtime = self._instances.pop(self._config_key(config), None)
                runtimes = [runtime] if runtime is not None else []

        for runtime in runtimes:
            close = getattr(runtime, "teardown", None)
            if callable(close):
                close()

    def diagnostics(self) -> RuntimeManagerDiagnostics:
        with self._lock:
            total_requests = self._reuse_hits + self._model_load_count
            hit_rate = (self._reuse_hits / total_requests) if total_requests else 0.0
            return RuntimeManagerDiagnostics(
                active_instances=len(self._instances),
                model_load_count=self._model_load_count,
                reuse_hits=self._reuse_hits,
                reuse_hit_rate=hit_rate,
            )


_runtime_manager_singleton = RuntimeManager()


def get_runtime_manager() -> RuntimeManager:
    return _runtime_manager_singleton
