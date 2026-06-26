"""
Tests for the signature-matrix classifier.

Two claims under test:
  1. Distinct degradation modes occupy unique coordinates — given a clean
     fingerprint, the right cause is pinned (`identifiable=True`).
  2. When the axis that *would* separate two modes is UNKNOWN, the classifier
     does NOT guess — it reports the degeneracy and names the missing axis
     (power range, fan RPM, cold-plate sensor) and how to get it.
"""

from theta.agent.signature import (
    FeatureVector, classify, FaultCause, Tristate, SIGNATURES,
    SCORE_FLOOR, STRONG_SCORE,
)


def _quiet(**over) -> FeatureVector:
    """A healthy, fully-observed GPU; override fields to inject a fault."""
    fv = FeatureVector(
        rtheta_overall_z=0.0, power_range_observed=True, alpha_z=0.0, beta_z=0.0,
        drift_rate_z=0.0, step_detected=False, near_service_event=False,
        locality="single", fan_rpm_residual=0.0, inlet_delta_z=0.0,
        mem_core_delta_z=0.0, dram_active=0.1, ecc_sbe_rate=0.0,
        nvlink_error_rate=0.0, pcie_replay_rate=0.0, power_violation_rate=0.0,
        clock_efficiency=1.0, recovery_tau_z=0.0, perf_per_watt_z=0.0,
    )
    for k, v in over.items():
        setattr(fv, k, v)
    return fv


# ── nominal ──────────────────────────────────────────────────────────────────

def test_quiet_gpu_is_nominal():
    v = classify(_quiet())
    assert v.headline_cause is FaultCause.NOMINAL
    assert not v.identifiable
    assert v.top.score < SCORE_FLOOR


# ── unique fingerprints pin the exact cause ─────────────────────────────────

def test_tim_pinned_by_slope_with_flat_intercept():
    # β steep, α flat, slow ramp, power range exercised → exact TIM.
    v = classify(_quiet(rtheta_overall_z=4.0, beta_z=4.0, alpha_z=0.5, drift_rate_z=3.0))
    assert v.headline_cause is FaultCause.TIM_DEGRADATION
    assert v.identifiable
    assert v.top.score >= STRONG_SCORE


def test_dust_separated_from_tim_by_elevated_intercept():
    # α AND β up uniformly → dust, not TIM (TIM has flat α).
    v = classify(_quiet(rtheta_overall_z=4.0, alpha_z=4.0, beta_z=2.5, drift_rate_z=4.0))
    assert v.headline_cause is FaultCause.DUST_ACCUMULATION
    assert FaultCause.DUST_ACCUMULATION not in v.degenerate_with
    assert v.identifiable


def test_fan_bearing_pinned_by_rpm_deficit():
    v = classify(_quiet(rtheta_overall_z=3.0, alpha_z=2.5, fan_rpm_residual=-0.30,
                        recovery_tau_z=3.0))
    assert v.headline_cause is FaultCause.FAN_BEARING_WEAR
    assert v.identifiable


def test_fabric_pinned_with_normal_thermals():
    v = classify(_quiet(nvlink_error_rate=5.0, rtheta_overall_z=0.3))
    assert v.headline_cause is FaultCause.FABRIC_LINK
    assert v.identifiable


def test_power_delivery_pinned_clock_capped_thermals_in_band():
    v = classify(_quiet(power_violation_rate=0.3, clock_efficiency=0.80,
                        rtheta_overall_z=0.4))
    assert v.headline_cause is FaultCause.POWER_DELIVERY
    assert v.identifiable


def test_hbm_pinned_by_memcore_delta_under_load():
    v = classify(_quiet(mem_core_delta_z=4.0, dram_active=0.8, ecc_sbe_rate=2.0,
                        rtheta_overall_z=0.2))
    assert v.headline_cause is FaultCause.HBM_THERMAL
    assert v.identifiable


# ── degeneracy: name the missing axis instead of guessing ───────────────────

def test_dust_vs_tim_degenerate_without_power_range():
    # No power range → α/β unobservable. R_θ is clearly rising and slowly, which
    # fits BOTH dust and TIM. The engine must refuse to pick and ask for the
    # decomposition.
    v = classify(_quiet(
        rtheta_overall_z=4.0, power_range_observed=False, alpha_z=None, beta_z=None,
        drift_rate_z=4.0, step_detected=False,
    ))
    assert not v.identifiable
    causes = {v.headline_cause, *v.degenerate_with}
    assert FaultCause.DUST_ACCUMULATION in causes
    assert FaultCause.TIM_DEGRADATION in causes
    needs = " ".join(a.needs for a in v.missing_axes).lower()
    assert "slope" in needs or "power" in needs
    assert any(a.via in ("workload", "probe") for a in v.missing_axes)


def test_fan_vs_blockage_degenerate_without_fan_telemetry():
    # α up on one GPU with a step — fits both fan-bearing and blockage. The
    # splitter (RPM-vs-duty) is unobserved → must surface "need fan RPM".
    v = classify(_quiet(
        rtheta_overall_z=3.0, alpha_z=3.0, step_detected=True,
        fan_rpm_residual=None,   # no fan telemetry
    ))
    assert not v.identifiable
    needs = " ".join(a.needs for a in v.missing_axes).lower()
    assert "fan" in needs and "rpm" in needs


def test_tim_identifiable_but_coldplate_refinement_still_surfaced():
    # A clean TIM fingerprint pins the cause and the remediation (it's a
    # conduction-path fault — repaste/reseat). The cold-plate-vs-TIM-dryout
    # distinction is a *sub-class refinement*: same subsystem, same action, so
    # it must NOT block the exact-cause call — but it IS surfaced as the finer
    # probe that would name the exact failure mode. This is the line between
    # "I can't tell which fault" (blocks) and "I know the fault; a deeper probe
    # would name the precise mode within it" (doesn't block).
    v = classify(_quiet(rtheta_overall_z=4.0, beta_z=4.0, alpha_z=0.5, drift_rate_z=3.0))
    assert v.identifiable
    assert any("cold-plate" in a.needs.lower() or "coolant" in a.needs.lower()
               for a in v.missing_axes)


# ── matrix integrity ─────────────────────────────────────────────────────────

def test_every_signature_predicate_is_three_valued_on_quiet_input():
    # No predicate should raise; every test returns a Tristate.
    fv = _quiet()
    for preds in SIGNATURES.values():
        for p in preds:
            assert p.test(fv) in (Tristate.SUPPORTS, Tristate.CONTRADICTS, Tristate.UNKNOWN)


def test_subsystem_level_when_no_discriminating_axis_observed():
    # Only a magnitude z is observed — no α/β, no fan, no recovery. Theta may
    # confirm the channel (thermal) but must NOT name a specific mode as if it
    # discriminated it. This is the real-data steady-load case (Princeton).
    v = classify(_quiet(
        rtheta_overall_z=4.0, power_range_observed=False, alpha_z=None, beta_z=None,
        drift_rate_z=None, step_detected=None, recovery_tau_z=None,
        fan_rpm_residual=None,
    ))
    assert not v.identifiable
    assert not v.discriminated      # nothing discriminating was exercised
    # A fully-observed fault, by contrast, IS discriminated.
    v2 = classify(_quiet(rtheta_overall_z=4.0, beta_z=4.0, alpha_z=0.5, drift_rate_z=3.0))
    assert v2.discriminated


def test_verdict_serializes():
    v = classify(_quiet(rtheta_overall_z=4.0, beta_z=4.0, alpha_z=0.5, drift_rate_z=3.0))
    d = v.as_dict()
    assert d["headline_cause"] == "tim_degradation"
    assert "missing_axes" in d and "ranked" in d and isinstance(d["identifiable"], bool)
