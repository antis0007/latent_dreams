from gguf_dream_lab.backend.dream.state import DreamMode
from gguf_dream_lab.backend.runtime.capability_contracts import RuntimeBehaviorSnapshot, validate_mode_contract


def test_true_mode_claim_downgrades_when_reinject_missing():
    validation = validate_mode_contract(
        DreamMode.TRUE_LATENT_INSTRUMENTED,
        RuntimeBehaviorSnapshot(
            capture=True,
            reinject=False,
            decode_provenance=True,
            control_authority=True,
        ),
    )

    assert validation.claimed_mode == DreamMode.TRUE_LATENT_INSTRUMENTED
    assert validation.effective_mode == DreamMode.ENHANCED_LATENT
    assert validation.downgraded is True
    assert validation.missing_behaviors == ["reinject"]


def test_enhanced_mode_claim_downgrades_to_baseline_when_control_authority_missing():
    validation = validate_mode_contract(
        DreamMode.ENHANCED_LATENT,
        RuntimeBehaviorSnapshot(
            capture=True,
            reinject=False,
            decode_provenance=True,
            control_authority=False,
        ),
    )

    assert validation.claimed_mode == DreamMode.ENHANCED_LATENT
    assert validation.effective_mode == DreamMode.BASELINE_APPROXIMATE
    assert validation.downgraded is True
    assert validation.missing_behaviors == ["control_authority"]
