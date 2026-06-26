"""
Per-GPU R_θ(P) residual curve — the α/β decomposition that sharpens attribution
automatically as a workload exercises a range of power.

The signature classifier's most important thermal discriminator is intercept (α)
vs slope (β): a uniform offset (dust, airflow) raises R_θ at every power, while a
conduction fault (TIM, contact loss) makes R_θ worse *specifically at high power*.
On steady-load data (every sample at one power) this axis is UNKNOWN — the gap the
Princeton analysis surfaced. This module fills it the moment the workload spans
power, with no probe required.

The trick that makes it hardware-agnostic: don't fit the raw R_θ(P) curve (whose
shape is hardware-specific) — fit the **residual** from the Princeton-calibrated
healthy H100 curve. A healthy unit has ~zero residual at every power. Then:

    α deviation  =  residual at low power            (offset: present everywhere)
    β deviation  =  residual_high − residual_low     (conduction: grows with power)

So dust → α high, β ~0; TIM → α ~0, β high — exactly the signature axes, computed
against real silicon. Pure state + arithmetic; trivially testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from . import h100_reference as h100

# Power bins (H100-scale watts). A unit must populate at least two, far enough
# apart, before α/β are trustworthy.
_BIN_EDGES = [0.0, 250.0, 400.0, 550.0, 1e9]
MIN_BIN_SAMPLES = 20       # samples in a bin before it counts
MIN_POWER_SPAN  = 150.0    # W between the low and high populated bin centers
_BIN_MAXLEN = 200          # rolling residuals retained per bin


@dataclass
class CurveDecomp:
    alpha_z: float          # offset deviation in noise units (low-power residual)
    beta_z:  float          # slope deviation in noise units (high − low residual)
    power_span_w: float     # observed low→high power span backing this decomposition
    low_power_w: float
    high_power_w: float


def _bin_index(power_w: float) -> int:
    for i in range(len(_BIN_EDGES) - 1):
        if _BIN_EDGES[i] <= power_w < _BIN_EDGES[i + 1]:
            return i
    return len(_BIN_EDGES) - 2


def _bin_center(idx: int) -> float:
    lo, hi = _BIN_EDGES[idx], _BIN_EDGES[idx + 1]
    return (lo + hi) / 2.0 if hi < 1e9 else lo + 100.0


class RThetaResidualCurve:
    """Per-GPU accumulator of (power → residual-from-healthy-curve)."""

    def __init__(self) -> None:
        self._bins: dict[int, deque[float]] = {}
        self._bin_power_sum: dict[int, float] = {}
        self._bin_n: dict[int, int] = {}

    def update(self, power_w: float, rtheta: Optional[float]) -> None:
        if rtheta is None or rtheta <= 0 or power_w <= 0:
            return
        resid = rtheta - h100.expected_rtheta(power_w)
        idx = _bin_index(power_w)
        self._bins.setdefault(idx, deque(maxlen=_BIN_MAXLEN)).append(resid)
        # Track the true mean power in the bin (for an accurate span/center).
        self._bin_power_sum[idx] = self._bin_power_sum.get(idx, 0.0) + power_w
        self._bin_n[idx] = self._bin_n.get(idx, 0) + 1

    def _populated(self) -> dict[int, tuple[float, float]]:
        """bin_idx → (mean_residual, mean_power) for bins with enough samples."""
        out = {}
        for idx, d in self._bins.items():
            if len(d) >= MIN_BIN_SAMPLES:
                out[idx] = (sum(d) / len(d), self._bin_power_sum[idx] / self._bin_n[idx])
        return out

    def decompose(self) -> Optional[CurveDecomp]:
        """
        α/β decomposition, or None until the workload has spanned enough power.
        Uses the lowest- and highest-power populated bins.
        """
        pop = self._populated()
        if len(pop) < 2:
            return None
        lo_idx = min(pop, key=lambda i: pop[i][1])
        hi_idx = max(pop, key=lambda i: pop[i][1])
        resid_lo, p_lo = pop[lo_idx]
        resid_hi, p_hi = pop[hi_idx]
        if p_hi - p_lo < MIN_POWER_SPAN:
            return None
        noise = h100.WITHIN_GPU_NOISE_STD
        return CurveDecomp(
            alpha_z=resid_lo / noise,
            beta_z=(resid_hi - resid_lo) / noise,
            power_span_w=p_hi - p_lo,
            low_power_w=p_lo,
            high_power_w=p_hi,
        )
