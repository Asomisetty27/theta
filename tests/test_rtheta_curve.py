"""
Tests for the R_θ(P) residual-curve accumulator — the α/β decomposition that
sharpens cause attribution as a workload spans power.

The decisive property: feeding REAL fingerprints (offset vs conduction) through
the accumulator → classifier yields the right exact cause, where steady-load data
could only reach subsystem-level. This is the mechanism that resolves the
identifiability gap the Princeton analysis exposed.
"""

from theta.agent import h100_reference as h100
from theta.agent.rtheta_curve import (
    RThetaResidualCurve, MIN_BIN_SAMPLES,
)
from theta.agent.signature_adapter import build_feature_vector
from theta.agent.signature import classify
from theta.agent.fault_classifier import FaultCause


def _feed(curve, power_w, rtheta, n=MIN_BIN_SAMPLES):
    for _ in range(n):
        curve.update(power_w, rtheta)


def test_steady_load_yields_no_decomposition():
    # All samples at one power → only one bin → α/β unobservable (None).
    c = RThetaResidualCurve()
    _feed(c, 650, 0.085, n=100)
    assert c.decompose() is None


def test_insufficient_power_span_yields_none():
    # Two bins but too close together → not enough leverage to decompose.
    c = RThetaResidualCurve()
    _feed(c, 560, 0.06)
    _feed(c, 590, 0.06)
    assert c.decompose() is None


def test_healthy_unit_decomposes_to_near_zero():
    # A unit tracking the healthy curve at every power → α≈0, β≈0.
    c = RThetaResidualCurve()
    for p in (200, 400, 650):
        _feed(c, p, h100.expected_rtheta(p))
    d = c.decompose()
    assert d is not None
    assert abs(d.alpha_z) < 0.5 and abs(d.beta_z) < 0.5


def test_offset_fault_is_high_alpha_low_beta():
    # Uniform +0.02 offset at every power (dust/airflow): residual flat across
    # power → α elevated, β ~0.
    c = RThetaResidualCurve()
    for p in (200, 650):
        _feed(c, p, h100.expected_rtheta(p) + 0.02)
    d = c.decompose()
    assert d.alpha_z > 2.0          # offset present at low power
    assert abs(d.beta_z) < 1.0      # does not grow with power


def test_conduction_fault_is_low_alpha_high_beta():
    # Residual ~0 at low power, large at high power (TIM): β elevated, α ~0.
    c = RThetaResidualCurve()
    _feed(c, 200, h100.expected_rtheta(200) + 0.001)   # fine at low power
    _feed(c, 650, h100.expected_rtheta(650) + 0.03)    # bad at high power
    d = c.decompose()
    assert abs(d.alpha_z) < 1.0
    assert d.beta_z > 3.0


def test_counter_reset_safe_and_span_reported():
    c = RThetaResidualCurve()
    _feed(c, 200, h100.expected_rtheta(200) + 0.02)
    _feed(c, 650, h100.expected_rtheta(650) + 0.02)
    d = c.decompose()
    assert d.power_span_w >= 150.0
    assert d.low_power_w < d.high_power_w


# ── the payoff: power range resolves the steady-load degeneracy ──────────────

def test_conduction_curve_resolves_to_exact_tim_through_classifier():
    # Build the conduction fingerprint via the accumulator, feed the whole chain.
    c = RThetaResidualCurve()
    _feed(c, 200, h100.expected_rtheta(200) + 0.001)
    _feed(c, 650, h100.expected_rtheta(650) + 0.03)
    fv = build_feature_vector(
        rtheta=h100.expected_rtheta(650) + 0.03, power_w=650,
        fault_cause=FaultCause.TIM_DEGRADATION, curve=c.decompose(),
    )
    v = classify(fv)
    # β observed and high, α observed and low → TIM, and now DISCRIMINATED
    # (a real key axis was exercised), unlike the steady-load subsystem-level case.
    assert v.headline_cause is FaultCause.TIM_DEGRADATION
    assert v.discriminated


def test_offset_curve_resolves_toward_dust_through_classifier():
    c = RThetaResidualCurve()
    for p in (200, 650):
        _feed(c, p, h100.expected_rtheta(p) + 0.02)
    fv = build_feature_vector(
        rtheta=h100.expected_rtheta(650) + 0.02, power_w=650,
        fault_cause=FaultCause.DUST_ACCUMULATION, curve=c.decompose(),
        fault_drift_rate=None,
    )
    v = classify(fv)
    assert v.discriminated
    assert v.headline_cause in (FaultCause.DUST_ACCUMULATION, FaultCause.AIRFLOW_BLOCKAGE)
