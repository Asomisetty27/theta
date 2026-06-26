"""
H100 empirical reference — calibrated from the real Princeton Della fleet.

Theta's defaults were derived from Stage-1 Tesla T4 data; on H100/B200 the R_θ
operating range is different, which is why `theta calibrate` exists. This module
is the *measured* H100 reference, extracted from 64 production H100s in the
immutable Princeton Della export (princeton_della_2026_06_11). It replaces
first-principles guesses for the axes that sit *upstream* of cause attribution:
what a healthy R_θ is, how much it varies, how it bends with power, and how fast
"normal" drifts.

Honest scope: this calibrates DETECTION and the OBSERVABLE-AXIS thresholds
against real silicon. It does NOT calibrate cause attribution — the Princeton
export carries no repair/inspection labels, so which-fault accuracy still needs
ground-truth from the Noyce AI Factory fleet or the E-LT testbed.

The single most important calibration: the healthy R_θ(P) curve is NONLINEAR and
falls with power (more power → more efficient cooling regime). So degradation is
measured as **deviation from the expected curve at the unit's own power**, not as
raw R_θ magnitude — which is what lets the α/β (intercept-vs-slope) axis be
computed on any varied-power fleet instead of collapsing on steady load.

Provenance: ~80k steady-state samples, 64 H100 SXM, 8 nodes, inlet ~25 °C.
"""

from __future__ import annotations

from bisect import bisect_left

# ── Healthy population (56 non-flagged units) ──────────────────────────────
HEALTHY_RTHETA_MEDIAN       = 0.05979   # C/W
HEALTHY_RTHETA_ROBUST_SIGMA = 0.00718   # 1.4826 · MAD — robust spread of the healthy fleet
HEALTHY_RTHETA_MIN          = 0.04867
HEALTHY_RTHETA_MAX          = 0.07155

# ── Noise / drift floors ───────────────────────────────────────────────────
WITHIN_GPU_NOISE_STD = 0.00626          # C/W — within-GPU sample noise (steady window)
DRIFT_SLOPE_MEDIAN   = 6.54e-05         # C/W per unit-time — typical fleet drift
DRIFT_SLOPE_P95      = 1.19e-03         # C/W per unit-time — a "fast drifter" lives above this

# ── R_θ(P) reference curve — (power_w, healthy R_θ, healthy std) per bin ────
# Nonlinear and decreasing: this is the shape degradation is measured against.
RTHETA_P_CURVE: list[tuple[float, float, float]] = [
    (185.0, 0.1198, 0.0351),
    (325.0, 0.0775, 0.0200),
    (475.0, 0.0697, 0.0149),
    (625.0, 0.0585, 0.0074),
]

# ── Positional (HGX slot) structure ────────────────────────────────────────
# Per-ordinal healthy R_θ — slot position imposes ~21% structure (spearman 0.68),
# which the position-conditioned detector removes before scoring a unit.
POSITIONAL_RTHETA: dict[int, float] = {
    0: 0.06856, 1: 0.05765, 2: 0.05572, 3: 0.06615,
    4: 0.06908, 5: 0.05616, 6: 0.06820, 7: 0.05978,
}
POSITIONAL_SPEARMAN = 0.677


def expected_rtheta(power_w: float) -> float:
    """
    Healthy-fleet R_θ expected at a given power, linearly interpolated between
    the measured bins (clamped at the ends). This is the baseline a unit's R_θ
    is compared against — accounting for power, not just a flat median.
    """
    pts = RTHETA_P_CURVE
    if power_w <= pts[0][0]:
        return pts[0][1]
    if power_w >= pts[-1][0]:
        return pts[-1][1]
    powers = [p for p, _, _ in pts]
    i = bisect_left(powers, power_w)
    p0, r0, _ = pts[i - 1]
    p1, r1, _ = pts[i]
    frac = (power_w - p0) / (p1 - p0)
    return r0 + frac * (r1 - r0)


def expected_std(power_w: float) -> float:
    """Healthy R_θ spread at a given power (nearest-bin); the local noise scale."""
    return min(RTHETA_P_CURVE, key=lambda b: abs(b[0] - power_w))[2]


def overall_z(rtheta: float, power_w: float | None = None) -> float:
    """
    Deviation of an observed R_θ from healthy, in robust-σ. When power is known,
    measure against the power-conditioned curve and the local spread (the correct,
    power-aware anomaly signal); otherwise fall back to the flat healthy median.
    Sign is positive when R_θ is HIGHER than healthy (the degradation direction).
    """
    if power_w is not None:
        ref = expected_rtheta(power_w)
        scale = max(expected_std(power_w), WITHIN_GPU_NOISE_STD)
        return (rtheta - ref) / scale
    return (rtheta - HEALTHY_RTHETA_MEDIAN) / HEALTHY_RTHETA_ROBUST_SIGMA


def positional_residual(rtheta: float, ordinal: int) -> float | None:
    """
    R_θ minus the healthy value for this HGX slot, in robust-σ. Removes the slot
    structure so a hot unit in a structurally-cool slot still surfaces (the
    j13g2:2 case). Returns None for an unknown ordinal.
    """
    ref = POSITIONAL_RTHETA.get(ordinal)
    if ref is None:
        return None
    return (rtheta - ref) / HEALTHY_RTHETA_ROBUST_SIGMA
