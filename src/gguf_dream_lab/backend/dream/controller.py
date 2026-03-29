from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from gguf_dream_lab.backend.atlas.atlas import LatentAtlas, StatePoint
from gguf_dream_lab.backend.dream.coherence import score_coherence
from gguf_dream_lab.backend.runtime.base import RuntimeBackend
from gguf_dream_lab.config.models import Basin, DreamConfig

SEED_BASIN_PREFIX = {
    Basin.NULL_PRIOR: "",
    Basin.INTROSPECTIVE: "I am inside a half-remembered thought where",
    Basin.NARRATIVE: "In a dimly lit scene,",
    Basin.MEMORY: "I remember something that almost happened:",
    Basin.AFFECTIVE_STYLE: "A feeling arrives first, then",
    Basin.CUSTOM: "",
}


@dataclass
class DreamTick:
    run_id: str
    step_idx: int
    preview_text: str
    committed_text: str
    coherence: float
    entropy: float
    local_density: float
    token_stability: float
    smoothness: float
    state_id: str
    status: str


@dataclass
class DreamSessionState:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    active: bool = False
    paused: bool = False
    step_idx: int = 0
    preview_text: str = ""
    committed_text: str = ""
    status: str = "idle"
    status_detail: str = ""
    latest_tick: DreamTick | None = None


class DreamController:
    def __init__(self, runtime: RuntimeBackend, atlas: LatentAtlas):
        self.runtime = runtime
        self.atlas = atlas
        self.state = DreamSessionState()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def start(self, cfg: DreamConfig) -> DreamSessionState:
        if self.state.active:
            return self.state
        self.state = DreamSessionState(active=True, paused=False, status="loading_model", status_detail="Loading model...")
        self._stop_event.clear()
        self._pause_event.clear()
        self._thread = threading.Thread(target=self._loop, args=(cfg,), daemon=True)
        self._thread.start()
        return self.state

    def pause(self) -> None:
        if self.state.active:
            self.state.paused = True
            self._pause_event.set()
            self.state.status = "paused"

    def resume(self) -> None:
        if self.state.active:
            self.state.paused = False
            self._pause_event.clear()
            self.state.status = "running"

    def stop(self) -> None:
        self._stop_event.set()
        self.state.active = False
        self.state.status = "stopped"

    def _loop(self, cfg: DreamConfig) -> None:
        try:
            self.runtime.load()
        except Exception as exc:
            self.state.active = False
            self.state.status = "error"
            self.state.status_detail = f"Model load failed: {exc}"
            return

        self.state.status = "running"
        self.state.status_detail = ""
        period = 1.0 / max(cfg.tick_hz, 0.1)
        preview_history: deque[str] = deque(maxlen=4)
        prev_vec = None
        basin_prefix = SEED_BASIN_PREFIX[cfg.basin]
        prompt = (basin_prefix + " " + cfg.prompt).strip()
        anneal = 1.0

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue
            tick_start = time.perf_counter()
            try:
                step = self.runtime.sample_step(prompt=prompt, max_tokens=12)
            except Exception as exc:
                self.state.status = "error"
                self.state.status_detail = f"Runtime step failed: {exc}"
                self._stop_event.set()
                break
            token = step.token.strip()
            if token:
                self.state.preview_text = self._merge_preview_token(self.state.preview_text, token, cfg.max_preview_len)
            preview_history.append(self.state.preview_text)
            token_stability = self._token_stability(preview_history)
            vec = self._normalize_embedding(step.embedding, fallback=prev_vec)
            if prev_vec is None:
                smoothness = 0.5
            else:
                cos = float(np.dot(vec, prev_vec) / (np.linalg.norm(vec) * np.linalg.norm(prev_vec) + 1e-9))
                smoothness = float((cos + 1.0) / 2.0)
            prev_vec = vec

            local_density = self._estimate_local_density(vec)
            coherence = score_coherence(
                entropy=step.entropy,
                local_density=local_density,
                token_stability=token_stability,
                smoothness=smoothness,
                branch_agreement=1.0,
                known_state_similarity=min(local_density / 5.0, 1.0),
                weights=cfg.weights,
            )

            if coherence >= cfg.coherence_threshold:
                if self.state.preview_text:
                    committed_addition = self.state.preview_text.split(" ")[-1]
                    self.state.committed_text = self._merge_committed_token(
                        self.state.committed_text,
                        committed_addition,
                        cfg.max_committed_len,
                    )
            prompt = self._evolve_prompt(base_prompt=(basin_prefix + " " + cfg.prompt).strip(), cfg=cfg)

            state_id = str(uuid.uuid4())
            self.atlas.append_points(
                [
                    StatePoint(
                        state_id=state_id,
                        run_id=self.state.run_id,
                        run_label=cfg.run_label,
                        basin=cfg.basin.value,
                        step_idx=self.state.step_idx,
                        preview=self.state.preview_text,
                        committed=self.state.committed_text,
                        coherence=coherence,
                        entropy=step.entropy,
                        embedding=(vec + np.random.normal(scale=cfg.noise_amplitude * anneal, size=vec.shape)).astype(np.float32),
                    )
                ]
            )

            self.state.latest_tick = DreamTick(
                run_id=self.state.run_id,
                step_idx=self.state.step_idx,
                preview_text=self.state.preview_text,
                committed_text=self.state.committed_text,
                coherence=coherence,
                entropy=step.entropy,
                local_density=local_density,
                token_stability=token_stability,
                smoothness=smoothness,
                state_id=state_id,
                status=self.state.status,
            )
            self.state.step_idx += 1
            anneal *= cfg.anneal_rate
            elapsed = time.perf_counter() - tick_start
            time.sleep(max(period - elapsed, 0.0))

        self.state.active = False

    def _estimate_local_density(self, vec: np.ndarray) -> float:
        if not self.atlas.points:
            return 0.0
        mat = np.stack([p.embedding for p in self.atlas.points], axis=0)
        if mat.shape[1] != vec.shape[0]:
            return 0.0
        dists = np.linalg.norm(mat - vec.reshape(1, -1), axis=1)
        top = np.sort(dists)[: min(8, len(dists))]
        return float(1.0 / (np.mean(top) + 1e-6))

    def _normalize_embedding(self, embedding: np.ndarray | None, fallback: np.ndarray | None = None) -> np.ndarray:
        if embedding is not None:
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.ndim > 1:
                vec = vec.mean(axis=0)
            if vec.ndim == 1 and vec.size:
                return vec

        if fallback is not None:
            return np.zeros_like(fallback, dtype=np.float32)
        if self.atlas.points:
            return np.zeros_like(self.atlas.points[-1].embedding, dtype=np.float32)
        return np.zeros(256, dtype=np.float32)

    @staticmethod
    def _token_stability(history: deque[str]) -> float:
        if len(history) < 2:
            return 0.0
        tail_words = [h.split(" ")[-1] if h else "" for h in history]
        counts = {w: tail_words.count(w) for w in set(tail_words)}
        return max(counts.values()) / len(tail_words)

    @staticmethod
    def _merge_preview_token(current_preview: str, token: str, max_len: int) -> str:
        """Mutate the preview by replacing repeated tails instead of endlessly appending."""
        words = current_preview.split()
        if words and words[-1] == token:
            return current_preview[-max_len:]
        if len(words) >= 2 and words[-1] == words[-2] and words[-1] != token:
            words[-1] = token
            updated = " ".join(words)
        else:
            updated = " ".join([*words, token]) if words else token
        return updated[-max_len:]

    @staticmethod
    def _merge_committed_token(current_committed: str, token: str, max_len: int) -> str:
        words = current_committed.split()
        if words and words[-1] == token:
            return current_committed[:max_len]
        updated = " ".join([*words, token]) if words else token
        return updated[:max_len]

    def _evolve_prompt(self, base_prompt: str, cfg: DreamConfig) -> str:
        """Use recent committed/preview context to avoid sampling from a static prompt."""
        committed_tail = " ".join(self.state.committed_text.split()[-24:])
        preview_tail = " ".join(self.state.preview_text.split()[-12:])
        parts = [part for part in (base_prompt, committed_tail, preview_tail) if part]
        return " ".join(parts)[-max(cfg.max_preview_len * 2, 256) :]
