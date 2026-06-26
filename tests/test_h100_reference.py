"""
Tests for the Princeton-calibrated H100 reference and the signature adapter.

These pin the calibration's intent: power-aware deviation (not flat magnitude),
honest None-propagation through the adapter, and that the reference reproduces
the healthy-fleet anchor points it was derived from.
"""

import pytest

from theta.agent import h100_reference as h100
from theta.agent.signature_adapter import build_feature_vector
from theta.agent.signature import classify
from theta.agent.fault_classifier import FaultCause


# ── reference curve ──────────────────────────────────────────────────────────

def test_curve_reproduces_anchor_points():
    # At the bin centers, expected R_θ matches the measured healthy values.
    assert h100.expected_rtheta(625) == pytest.approx(0.0585, abs=1e-4)
    assert h100.expected_rtheta(185) == pytest.approx(0.1198, abs=1e-4)


def test_curve_is_monotone_decreasing_and_clamped():
    # Healthy R_θ falls with power; clamps outside the measured range.
    assert h100.expected_rtheta(100) == h100.expected_rtheta(185)   # clamp low
    assert h100.expected_rtheta(900) == h100.expected_rtheta(625)   # clamp high
    assert h100.expected_rtheta(250) > h100.expected_rtheta(550)


def test_overall_z_is_power_aware():
    # The SAME R_θ is healthy at low power but anomalous at high power, because
    # the healthy curve is much lower at high power. Flat-magnitude scoring can't
    # see this; power-aware scoring is the accuracy gain.
    rtheta = 0.085
    z_lowP = h100.overall_z(rtheta, 200)
    z_highP = h100.overall_z(rtheta, 650)
    assert z_highP > z_lowP
    assert z_highP > 2.0          # clearly elevated vs the 0.0585 high-power baseline
    assert abs(z_lowP) < 2.0      # within band vs the 0.12 low-power baseline


def test_overall_z_flat_fallback_without_power():
    # Healthy median reads ~0σ; a degraded value reads high.
    assert abs(h100.overall_z(h100.HEALTHY_RTHETA_MEDIAN)) < 0.1
    assert h100.overall_z(0.085) > 3.0


def test_positional_residual_surfaces_cool_slot_unit():
    # Ordinal 2 is the coolest slot (0.0557). A unit there at 0.075 is a big
    # residual even though its raw R_θ is unremarkable fleet-wide — the j13g2:2
    # "hidden in a cool slot" case.
    r = h100.positional_residual(0.075, 2)
    assert r is not None and r > 2.0
    assert h100.positional_residual(0.075, 99) is None  # unknown ordinal


# ── adapter ──────────────────────────────────────────────────────────────────

def test_adapter_steady_load_leaves_alpha_beta_unknown():
    fv = build_feature_vector(
        rtheta=0.085, power_w=650, fault_cause=FaultCause.INSUFFICIENT_DATA,
    )
    assert fv.power_range_observed is False
    assert fv.alpha_z is None and fv.beta_z is None
    assert fv.rtheta_overall_z is not None and fv.rtheta_overall_z > 2.0


def test_adapter_power_range_enables_alpha_beta():
    from theta.agent.rtheta_curve import CurveDecomp
    decomp = CurveDecomp(alpha_z=0.3, beta_z=3.5, power_span_w=400,
                         low_power_w=200, high_power_w=600)
    fv = build_feature_vector(
        rtheta=0.085, power_w=650, fault_cause=FaultCause.TIM_DEGRADATION,
        curve=decomp,
    )
    assert fv.power_range_observed is True
    assert fv.alpha_z == 0.3 and fv.beta_z == 3.5


def test_adapter_propagates_fabric_and_locality():
    fv = build_feature_vector(
        rtheta=0.06, power_w=650, fault_cause=FaultCause.NOMINAL,
        nvlink_error_rate=5.0, correlated_gpus=(1, 2),
    )
    assert fv.nvlink_error_rate == 5.0
    assert fv.locality == "node"
    # fed through the classifier, the live fabric signal surfaces FABRIC_LINK
    v = classify(fv)
    assert v.headline_cause is FaultCause.FABRIC_LINK


def test_adapter_unknowns_stay_none():
    fv = build_feature_vector(rtheta=0.06, power_w=650, fault_cause=FaultCause.NOMINAL)
    # Signals the base collector lacks must remain UNKNOWN, never fabricated.
    assert fv.fan_rpm_residual is None
    assert fv.mem_core_delta_z is None
    assert fv.recovery_tau_z is None
