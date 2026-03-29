from __future__ import annotations

from dataclasses import dataclass

from gguf_dream_lab.backend.dream.state import DreamMode


@dataclass(frozen=True)
class RuntimeBehaviorSnapshot:
    capture: bool
    reinject: bool
    decode_provenance: bool
    control_authority: bool


@dataclass(frozen=True)
class CapabilityContractValidation:
    claimed_mode: DreamMode
    effective_mode: DreamMode
    missing_behaviors: list[str]

    @property
    def downgraded(self) -> bool:
        return self.claimed_mode != self.effective_mode


MODE_CONTRACTS: dict[DreamMode, tuple[str, ...]] = {
    DreamMode.BASELINE_APPROXIMATE: ("capture",),
    DreamMode.ENHANCED_LATENT: ("capture", "decode_provenance", "control_authority"),
    DreamMode.TRUE_LATENT_INSTRUMENTED: ("capture", "reinject", "decode_provenance", "control_authority"),
}

MODE_FALLBACK_ORDER: tuple[DreamMode, ...] = (
    DreamMode.TRUE_LATENT_INSTRUMENTED,
    DreamMode.ENHANCED_LATENT,
    DreamMode.BASELINE_APPROXIMATE,
)


def validate_mode_contract(claimed_mode: DreamMode, behaviors: RuntimeBehaviorSnapshot) -> CapabilityContractValidation:
    missing_for_claim = _missing_behaviors(claimed_mode, behaviors)
    if not missing_for_claim:
        return CapabilityContractValidation(
            claimed_mode=claimed_mode,
            effective_mode=claimed_mode,
            missing_behaviors=[],
        )

    for candidate in MODE_FALLBACK_ORDER:
        if not _missing_behaviors(candidate, behaviors):
            return CapabilityContractValidation(
                claimed_mode=claimed_mode,
                effective_mode=candidate,
                missing_behaviors=missing_for_claim,
            )
    return CapabilityContractValidation(
        claimed_mode=claimed_mode,
        effective_mode=DreamMode.BASELINE_APPROXIMATE,
        missing_behaviors=missing_for_claim,
    )


def _missing_behaviors(mode: DreamMode, behaviors: RuntimeBehaviorSnapshot) -> list[str]:
    required = MODE_CONTRACTS[mode]
    missing: list[str] = []
    for behavior_name in required:
        if not getattr(behaviors, behavior_name):
            missing.append(behavior_name)
    return missing
