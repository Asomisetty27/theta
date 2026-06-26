"""
Rate tracker — turns the collector's monotonic counters into the per-second and
fractional rates the causal engine actually reasons about.

The collector exposes *cumulative* counters (total NVLink CRC+recovery errors,
accumulated power-throttle microseconds). Feeding those raw into the causal
engine would be wrong twice over: a GPU that logged a handful of NVLink errors
hours ago would look permanently "degraded," and a single accumulated number
carries no sense of "right now." What matters for diagnosis is the *rate* —
errors per second, fraction of the interval spent throttled — so a problem that
is actively happening separates cleanly from old, settled history.

This is the missing piece that activates FABRIC_LINK / POWER_DELIVERY in the
live daemon. It is pure and per-GPU: first sample yields zeros (no interval to
divide by yet), and a counter reset (GPU reboot drops the counter back to 0) is
clamped to zero rather than producing a negative spike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Microseconds per second — power-violation counters are accumulated µs.
_US_PER_S = 1_000_000.0


@dataclass
class FabricPowerRates:
    """Instantaneous rates for one GPU over the last sampling interval."""
    nvlink_error_rate:    float = 0.0   # NVLink CRC+recovery errors per second
    pcie_replay_rate:     float = 0.0   # PCIe replays per second (0 until collected)
    power_violation_rate: float = 0.0   # fraction of the interval power-throttled, [0, 1]


class RateTracker:
    """
    Per-GPU previous-sample memory. Call `update()` once per tick with the
    current cumulative counters; get back the rates since the last tick.
    """

    def __init__(self) -> None:
        # gpu -> (timestamp, {counter_name: value})
        self._prev: dict[int, tuple[float, dict[str, float]]] = {}

    def update(
        self,
        gpu: int,
        ts: float,
        *,
        nvlink_errors: float,
        power_violation_us: float,
        pcie_replays: Optional[float] = None,
    ) -> FabricPowerRates:
        counters: dict[str, float] = {
            "nvlink_errors": float(nvlink_errors),
            "power_violation_us": float(power_violation_us),
        }
        if pcie_replays is not None:
            counters["pcie_replays"] = float(pcie_replays)

        prev = self._prev.get(gpu)
        self._prev[gpu] = (ts, counters)

        # No interval to divide by on the first sample.
        if prev is None:
            return FabricPowerRates()

        prev_ts, prev_c = prev
        dt = ts - prev_ts
        if dt <= 0:
            return FabricPowerRates()

        def delta(key: str) -> float:
            dv = counters[key] - prev_c.get(key, counters[key])
            return dv if dv >= 0.0 else 0.0  # counter reset (reboot) → 0, not negative

        nvlink_rate = delta("nvlink_errors") / dt
        # Fraction of wall-clock time the GPU spent throttled this interval.
        power_rate = min(1.0, delta("power_violation_us") / (dt * _US_PER_S))
        pcie_rate = (delta("pcie_replays") / dt) if "pcie_replays" in counters else 0.0

        return FabricPowerRates(
            nvlink_error_rate=nvlink_rate,
            pcie_replay_rate=pcie_rate,
            power_violation_rate=power_rate,
        )

    def reset(self, gpu: int) -> None:
        """Forget a GPU's history (e.g. after it leaves the fleet)."""
        self._prev.pop(gpu, None)


def clock_efficiency(clock_sm_mhz: float, sm_clock_max_mhz: float) -> float:
    """
    SM clock as a fraction of its boost ceiling. 1.0 = at boost. Instantaneous,
    no history needed. Meaningful as a *suppression* signal only alongside a
    power-violation rate — at idle the clock is legitimately low, which is why
    the causal engine requires both before it concludes POWER_DELIVERY.
    """
    if sm_clock_max_mhz and sm_clock_max_mhz > 0:
        return clock_sm_mhz / sm_clock_max_mhz
    return 1.0
