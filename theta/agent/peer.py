"""
Peer-relative anomaly detector — the E009 method, made live.

The per-GPU :class:`DriftDetector` is *temporal*: it compares a GPU to its OWN
rolling baseline. That has two structural blind spots:

  1. It needs a warm-up — it cannot say anything until the GPU has logged
     ~20 healthy stable windows of its own.
  2. It cannot see a GPU that has been degraded *the entire time you have been
     watching it* — there is no healthy baseline to diverge from.

E009 (Princeton Della, 72 production H100s) closed both gaps with a
*cross-sectional* method: at a single moment, compare each GPU's R_θ to its
node-mates **at matched power**, and flag robust-z outliers. It blind-flagged
3 degraded units — one at robust z = +15.6 — with no temporal history at all,
two of them invisible to temperature thresholds. That is the project's
strongest real-hardware validation, so for any fleet it should be a *default*
signal, not a special case.

This detector is the complement to :class:`DriftDetector`, not a replacement:

    DriftDetector   temporal,        per-GPU,      needs warm-up,  any node size
    PeerDetector    cross-sectional, peer-relative, no warm-up,    needs a fleet

Method (robust, distribution-free — the E009 recipe):
  * Group GPUs by matched power (R_θ is a curve in P, so peers must be compared
    at comparable load — peers are GPUs within ±`power_tol` fractional power).
  * Within a group of at least `min_group` GPUs, take the median and MAD of
    R_θ. robust_z = (rtheta − median) / (1.4826 · MAD).  1.4826 makes MAD a
    consistent estimator of σ for normal data; the median/MAD pair is unmoved
    by a minority of outliers, so one degraded GPU cannot mask itself.
  * The scale (1.4826·MAD) is floored so a near-uniform fleet with one slightly
    high GPU does not produce a screaming z from numerical noise. The floor is
    *relative* to the peer median (a fraction of it), NOT a fixed C/W constant:
    H100/B200 fleets run R_θ ≈ 0.05–0.12, so a T4-scale absolute floor (~0.01)
    would swamp a real degradation signal — the same compressed-range problem
    DriftDetector documents for liquid-cooled hardware. A relative floor scales
    with the cohort, so the detector works across vendors without retuning.
  * Alert only on a robust-z sustained over several evaluation cycles.

Guard rails (first-run trust > cleverness):
  * No group of `min_group` matched-power peers → no evaluation, no alert. This
    makes peer detection inherently a *fleet* feature: single / dual-GPU hosts
    never peer-alarm.
  * Stateless across snapshots except the sustained-cycle counter, so it needs
    zero warm-up and reflects the fleet as it is right now.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional

# Robust-z thresholds. Deliberately conservative — E009's real degraded unit
# sat at +15.6, far above these, so we can keep the false-positive budget tight.
Z_WARN      = 4.0
Z_CRITICAL  = 8.0

POWER_TOL       = 0.15   # peers are GPUs within ±15% power (matched-load comparison)
MIN_GROUP       = 4      # candidate + ≥3 peers before a group is trustworthy
REL_FLOOR       = 0.04   # min scale as a fraction of the peer median (hardware-agnostic)
SUSTAINED       = 3      # consecutive cycles above Z_WARN before alerting
MAD_K           = 1.4826 # MAD → σ consistency constant for normal data


@dataclass
class PeerResult:
    gpu_index:     int
    timestamp:     float
    rtheta:        float
    power_w:       float
    robust_z:      Optional[float]    # None when the GPU had no valid peer group
    peer_median:   Optional[float]
    peer_scale:    Optional[float]    # 1.4826·MAD, floored
    n_peers:       int                # peers compared against (group size − 1)
    is_anomaly:    bool               # sustained robust-z above Z_WARN
    is_critical:   bool               # sustained robust-z above Z_CRITICAL
    confidence:    float              # sustained-count / SUSTAINED, capped at 1


class PeerRelativeDetector:
    """
    Cross-sectional, power-matched, no-warm-up fleet anomaly detector.

    Call :meth:`evaluate` once per polling cycle with the whole fleet's current
    (power, R_θ) snapshot. Returns one :class:`PeerResult` per GPU.
    """

    def __init__(
        self,
        z_warn:     float = Z_WARN,
        z_critical: float = Z_CRITICAL,
        power_tol:  float = POWER_TOL,
        min_group:  int   = MIN_GROUP,
        rel_floor:  float = REL_FLOOR,
        sustained:  int   = SUSTAINED,
    ):
        self._z_warn      = z_warn
        self._z_critical  = z_critical
        self._power_tol   = power_tol
        self._min_group   = min_group
        self._rel_floor   = rel_floor
        self._sustained   = sustained
        self._anomaly_counts: dict[int, int] = {}

    def evaluate(
        self,
        snapshot: dict[int, tuple[float, float]],
        timestamp: float,
    ) -> dict[int, PeerResult]:
        """
        snapshot: {gpu_index: (power_w, rtheta)}. Only GPUs with a finite,
        positive R_θ should be passed — callers skip invalid/low-power reads.
        """
        results: dict[int, PeerResult] = {}

        for gpu, (power, rtheta) in snapshot.items():
            # Peer group: other GPUs at comparable power (matched-load), plus self.
            group = [
                rt for g2, (p2, rt) in snapshot.items()
                if _power_matched(power, p2, self._power_tol)
            ]

            if len(group) < self._min_group:
                # Not enough matched-power peers — say nothing. A single high
                # GPU on a small/heterogeneous host must never peer-alarm.
                self._anomaly_counts.pop(gpu, None)
                results[gpu] = PeerResult(
                    gpu_index=gpu, timestamp=timestamp, rtheta=rtheta, power_w=power,
                    robust_z=None, peer_median=None, peer_scale=None,
                    n_peers=len(group) - 1, is_anomaly=False, is_critical=False,
                    confidence=0.0,
                )
                continue

            median = statistics.median(group)
            mad    = statistics.median([abs(v - median) for v in group])
            # Relative floor: scales with the cohort's own R_θ level so the
            # detector is hardware-agnostic (see module docstring).
            scale  = max(MAD_K * mad, self._rel_floor * median)
            robust_z = (rtheta - median) / scale

            above_warn = robust_z > self._z_warn
            count = self._anomaly_counts.get(gpu, 0)
            count = count + 1 if above_warn else max(0, count - 1)
            self._anomaly_counts[gpu] = count

            is_anomaly  = above_warn and count >= self._sustained
            is_critical = robust_z > self._z_critical and count >= self._sustained
            confidence  = min(1.0, count / self._sustained) if above_warn else 0.0

            results[gpu] = PeerResult(
                gpu_index=gpu, timestamp=timestamp, rtheta=rtheta, power_w=power,
                robust_z=round(robust_z, 2), peer_median=round(median, 4),
                peer_scale=round(scale, 4), n_peers=len(group) - 1,
                is_anomaly=is_anomaly, is_critical=is_critical,
                confidence=round(confidence, 2),
            )

        return results

    def reset(self, gpu_index: int) -> None:
        self._anomaly_counts.pop(gpu_index, None)


def _power_matched(p1: float, p2: float, tol: float) -> bool:
    """True if p2 is within ±tol fractional power of p1 (matched-load peers)."""
    if p1 <= 0 or p2 <= 0:
        return False
    return abs(p1 - p2) <= tol * p1


# ── Position-conditioned (fleet) scoring — the full E009 median-polish ────────
#
# The PeerRelativeDetector above compares a GPU to its node-mates directly. That
# catches an unambiguous outlier (E009's j13g2:7, +58%) but MISSES units whose
# anomaly is masked by HGX baseboard-position structure: on Della, position
# effects span ±11% of μ (hot ordinals {0,3,4,6}, cool {1,2,5,7}) — bigger than
# the 3.7% residual noise floor. A hot GPU sitting in a structurally-cool slot
# can read at or below its node median yet be genuinely degraded (E009's
# j13g2:2 was −2.5% within-node but +12% after position correction).
#
# Removing that structure needs a *fleet*: several nodes that share the ordinal
# layout, so the per-ordinal effect can be estimated by pooling the same slot
# across nodes. This is therefore a fleet-service capability, not something a
# single-node agent can do alone. Two-way (node × ordinal) median polish is the
# E009 method; it reproduces all 3 Della flags at zero false positives.

def median_polish_z(
    fleet: dict[int, tuple[str, int, float]],
    iterations: int = 20,
    rel_floor: float = REL_FLOOR,
) -> dict[int, float]:
    """
    Position-conditioned robust-z for a multi-node fleet.

    fleet: {gpu_id: (node, ordinal, rtheta)} — callers should pass only
    matched-power, steady-load GPUs (R_θ is a curve in P). Decomposes
    R(node, ordinal) = μ + node_effect + ordinal_effect + residual via Tukey
    two-way median polish, then scores each residual against the robust σ
    (1.4826·MAD of residuals, floored relative to μ). Returns {gpu_id: robust_z}.
    GPUs whose (node, ordinal) cell is unique enough to be unpolishable are
    still returned with their residual z.
    """
    cells = {(node, ordn): (gid, rt) for gid, (node, ordn, rt) in fleet.items()}
    nodes = sorted({n for n, _ in cells})
    ords  = sorted({o for _, o in cells})

    res = {k: cells[k][1] for k in cells}            # residuals, polished in place
    for _ in range(iterations):
        for n in nodes:                              # sweep node (row) medians
            row = [res[(n, o)] for o in ords if (n, o) in res]
            if row:
                m = statistics.median(row)
                for o in ords:
                    if (n, o) in res:
                        res[(n, o)] -= m
        for o in ords:                               # sweep ordinal (col) medians
            col = [res[(n, o)] for n in nodes if (n, o) in res]
            if col:
                m = statistics.median(col)
                for n in nodes:
                    if (n, o) in res:
                        res[(n, o)] -= m

    resid = list(res.values())
    rmed  = statistics.median(resid)
    mad   = statistics.median([abs(x - rmed) for x in resid])
    grand = statistics.median([cells[k][1] for k in cells])
    sigma = max(MAD_K * mad, rel_floor * grand)

    return {cells[k][0]: round(res[k] / sigma, 2) for k in cells}
