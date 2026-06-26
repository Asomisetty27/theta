"""
Adapter: daemon signals → signature `FeatureVector`.

The signature classifier is pure and signal-agnostic; this module is the single
place that maps Theta's live daemon state onto its axes, using the H100 empirical
reference (Princeton-calibrated) to z-score against real silicon rather than
T4-derived guesses. Kept separate and unit-tested so the daemon hook is one call.

Honesty is structural: any signal the daemon doesn't actually have maps to None
(→ UNKNOWN on that axis), which the classifier reports as a missing axis instead
of guessing. Nothing here fabricates a value to fill a gap.
"""

from __future__ import annotations

from typing import Optional

from . import h100_reference as h100
from .fault_classifier import FaultCause
from .rtheta_curve import CurveDecomp
from .signature import FeatureVector


def build_feature_vector(
    *,
    rtheta: Optional[float],
    power_w: float,
    fault_cause: FaultCause,
    peer_robust_z: Optional[float] = None,
    curve: Optional[CurveDecomp] = None,
    fault_drift_rate: Optional[float] = None,
    fault_session_delta: Optional[float] = None,
    gpu_ordinal: Optional[int] = None,
    correlated_gpus: tuple[int, ...] = (),
    nvlink_error_rate: float = 0.0,
    pcie_replay_rate: float = 0.0,
    power_violation_rate: float = 0.0,
    clock_efficiency: float = 1.0,
) -> FeatureVector:
    """
    Compose a FeatureVector from what the daemon computed this cycle. Power-aware
    where possible (the H100 curve), None where the signal isn't available.
    """
    # ── Magnitude: the STRONGEST available "elevated vs healthy" signal. ──
    #    Two complementary detectors: the power-aware deviation from the H100
    #    R_θ(P) curve, and the position-conditioned peer robust-z (which catches
    #    a unit hidden in a structurally-cool HGX slot — the j13g2:2 case the
    #    curve alone misses). Take the max so whichever fires drives attribution.
    overall_z = None
    if rtheta is not None and rtheta > 0:
        overall_z = h100.overall_z(rtheta, power_w)
    if peer_robust_z is not None:
        overall_z = peer_robust_z if overall_z is None else max(overall_z, peer_robust_z)

    # ── α/β: from the R_θ(P) residual curve, the moment a workload spans power. ──
    #    `curve` is the decomposition of the unit's deviation from the healthy
    #    H100 curve — α = offset (present at all power), β = how much the deviation
    #    grows with power (conduction). None until the power span is sufficient,
    #    which is the honest steady-load gap the Princeton analysis surfaced.
    power_range_observed = curve is not None
    alpha_z = curve.alpha_z if curve is not None else None
    beta_z = curve.beta_z if curve is not None else None

    # ── Time-shape: drift rate vs the fleet's fast-drifter p95; step vs noise. ──
    drift_z = None
    if fault_drift_rate is not None:
        # Map the fleet's p95 fast-drifter line onto the ELEVATED threshold (2σ):
        # a unit drifting at p95 reads as a clearly-elevated ramp.
        drift_z = (fault_drift_rate / h100.DRIFT_SLOPE_P95) * 2.0
    step = None
    if fault_session_delta is not None:
        step = abs(fault_session_delta) > 2.0 * h100.WITHIN_GPU_NOISE_STD

    # ── Locality: correlated node-mates → node/rack common mode, else single. ──
    locality = "node" if correlated_gpus else "single"

    return FeatureVector(
        rtheta_overall_z=overall_z,
        power_range_observed=power_range_observed,
        alpha_z=alpha_z,
        beta_z=beta_z,
        drift_rate_z=drift_z,
        step_detected=step,
        near_service_event=None,        # no maintenance-log join in the live agent yet
        locality=locality,
        fan_rpm_residual=None,          # duty is known, true RPM is not → UNKNOWN
        inlet_delta_z=None,
        mem_core_delta_z=None,          # no separate HBM temperature in the base collector
        dram_active=None,
        ecc_sbe_rate=None,
        nvlink_error_rate=nvlink_error_rate,
        pcie_replay_rate=pcie_replay_rate,
        power_violation_rate=power_violation_rate,
        clock_efficiency=clock_efficiency,
        recovery_tau_z=None,            # cooldown τ not tracked live yet
        perf_per_watt_z=None,
    )
