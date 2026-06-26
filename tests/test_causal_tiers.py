"""
Tests for the confidence-tier taxonomy and the fabric/power cause classes added
to the causal reasoning engine.

The headline guarantees these pin:
  - The passive engine NEVER emits a CONFIRMED_* tier (the "no fake 100%" rule).
  - A second, physically-independent instrument is what earns HIGH.
  - A non-thermal subsystem failing while thermals are clean gets PROMOTED to the
    primary hypothesis instead of headlining NOMINAL.
  - Pre-existing thermal callers are unaffected (backward compatibility).
"""

from theta.agent.causal import (
    reason, ConfidenceTier, _assess_tier,
)
from theta.agent.fault_classifier import FaultCause
from theta.agent.metrics import GPUState


def _base(**over):
    """A minimal healthy reason() call; override fields per test."""
    kw = dict(
        gpu_index=0,
        smoothed_state=GPUState.CLEAN_IDLE,
        state_confidence=0.9,
        alternative_states=[],
        fault_cause=FaultCause.NOMINAL,
        fault_confidence=0.5,
        rtheta_current=0.72,
        rtheta_baseline=0.72,
        rtheta_k_sigma=0.0,
    )
    kw.update(over)
    return reason(**kw)


# ── the cardinal rule: passive never confirms ────────────────────────────────

def test_passive_engine_never_emits_confirmed_tier():
    # Even a maximal-evidence passive diagnosis tops out at HIGH.
    exp = _base(
        fault_cause=FaultCause.TIM_DEGRADATION, fault_confidence=0.9,
        rtheta_k_sigma=6.0, rtheta_trend_per_min=0.01,
        correlated_gpus=(1, 2), ecc_dbit_any=True, micro_throttle=True,
    )
    assert exp.tier in (ConfidenceTier.UNCONFIRMED, ConfidenceTier.PROBABLE, ConfidenceTier.HIGH)
    assert exp.tier.rank < ConfidenceTier.CONFIRMED_SUBSYSTEM.rank


def test_independent_signal_earns_high():
    # ECC double-bit is a physically distinct instrument from R_θ → HIGH.
    exp = _base(fault_cause=FaultCause.HBM_THERMAL, rtheta_k_sigma=3.5, ecc_dbit_any=True)
    assert exp.tier is ConfidenceTier.HIGH


def test_passive_multisignal_is_probable_not_high():
    # Strong σ + peer correlation, but no independent instrument → PROBABLE.
    exp = _base(
        fault_cause=FaultCause.DUST_ACCUMULATION,
        rtheta_k_sigma=4.0, correlated_gpus=(1, 2),
    )
    assert exp.tier is ConfidenceTier.PROBABLE


def test_single_weak_passive_signal_is_unconfirmed():
    exp = _base(fault_cause=FaultCause.DUST_ACCUMULATION, rtheta_k_sigma=1.2)
    assert exp.tier is ConfidenceTier.UNCONFIRMED


def test_nominal_is_unconfirmed():
    assert _base().tier is ConfidenceTier.UNCONFIRMED


def test_assess_tier_unit():
    # Direct unit coverage of the pure assessor.
    assert _assess_tier(
        fault_cause=FaultCause.NOMINAL, rtheta_k_sigma=0.0,
        rtheta_trend_per_min=0.0, correlated_gpus=(), independent_signal=False,
    ) is ConfidenceTier.UNCONFIRMED
    assert _assess_tier(
        fault_cause=FaultCause.TIM_DEGRADATION, rtheta_k_sigma=0.0,
        rtheta_trend_per_min=0.0, correlated_gpus=(), independent_signal=True,
    ) is ConfidenceTier.HIGH


# ── fabric / power cause classes ─────────────────────────────────────────────

def test_fabric_link_promoted_when_thermals_clean():
    # Thermal classifier says NOMINAL, but NVLink errors are climbing → the
    # primary hypothesis must become FABRIC_LINK, not stay NOMINAL.
    exp = _base(rtheta_k_sigma=0.3, nvlink_error_rate=5.0)
    assert exp.hypothesis.cause is FaultCause.FABRIC_LINK
    assert exp.tier is ConfidenceTier.HIGH  # fabric is an independent instrument
    assert any("nvlink" in e.name.lower() or "fabric" in e.name.lower() for e in exp.evidence)
    assert any("NVLink" in a.title or "PCIe" in a.title for a in exp.actions)


def test_power_delivery_detected_when_clocks_capped_thermals_normal():
    exp = _base(rtheta_k_sigma=0.5, clock_efficiency=0.80, power_violation_rate=0.30)
    assert exp.hypothesis.cause is FaultCause.POWER_DELIVERY
    assert any("power" in a.title.lower() for a in exp.actions)


def test_power_delivery_not_flagged_when_rtheta_hot():
    # If R_θ is elevated, low clocks are a THERMAL throttle story, not power
    # delivery — power_limited must stay off so we don't mislabel.
    exp = _base(
        fault_cause=FaultCause.TIM_DEGRADATION,
        rtheta_k_sigma=4.0, clock_efficiency=0.80, power_violation_rate=0.30,
    )
    assert exp.hypothesis.cause is FaultCause.TIM_DEGRADATION


def test_fabric_does_not_override_a_real_thermal_diagnosis():
    # When the thermal path already has a verdict, fabric stays an ALTERNATIVE.
    exp = _base(
        fault_cause=FaultCause.DUST_ACCUMULATION, fault_confidence=0.8,
        rtheta_k_sigma=4.0, nvlink_error_rate=5.0,
    )
    assert exp.hypothesis.cause is FaultCause.DUST_ACCUMULATION
    assert any(h.cause is FaultCause.FABRIC_LINK for h in exp.alternatives)


# ── backward compatibility ───────────────────────────────────────────────────

def test_thermal_only_path_unchanged():
    # A pure thermal call with no new signals behaves exactly as before:
    # default tier wiring, no fabric/power artifacts.
    exp = _base(fault_cause=FaultCause.TIM_DEGRADATION, fault_confidence=0.7, rtheta_k_sigma=3.0)
    assert exp.hypothesis.cause is FaultCause.TIM_DEGRADATION
    assert not any(h.cause in (FaultCause.FABRIC_LINK, FaultCause.POWER_DELIVERY)
                   for h in exp.alternatives)
    # as_dict still serializes and now carries the tier.
    d = exp.as_dict()
    assert d["tier"] == exp.tier.value
    assert "hypothesis" in d and "actions" in d
