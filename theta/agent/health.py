"""
Health-as-conditions — the scheduler-facing health surface.

Alerts are EDGE events ("GPU 3 just started drifting"). A scheduler or operator
deciding whether to cordon/drain a node needs the orthogonal thing: the current
LEVEL state — "is GPU 3 fit to run work right now, what's wrong, and since when?"
That's the node-problem-detector pattern (NodeConditions), applied per GPU.

This tracker maintains, per GPU, a set of named **conditions** (problem present or
not) with transition timestamps, and derives an overall health status + a single
`schedulable` boolean. It is fed each cycle from signals the daemon already
computes (classified state, drift result, governor warming, silicon faults), so it
adds a consumable surface without new detection.

Conditions are level-state, not latched-forever: each reflects current truth and
records when it last transitioned (NPD semantics). Upstream signals already carry
hysteresis (sustained-window drift, governor warm-up), so the conditions don't flap.

Distinct from:
  - GPUState (metrics.py): the thermal *classification* (clean_idle/under_load/…).
  - AlertEvent: the edge *event* when something changes.
  - this: the current *condition* a scheduler reads to make a placement decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .metrics import GPUState


class HealthStatus(str, Enum):
    UNKNOWN  = "unknown"    # no observation yet
    WARMING  = "warming"    # monitor not yet confident; GPU itself presumed fine
    HEALTHY  = "healthy"    # confident, no problem conditions
    DEGRADED = "degraded"   # a warning-level problem is present
    CRITICAL = "critical"   # a critical problem is present


# Condition name → severity tier. CRITICAL conditions make a GPU non-schedulable
# and set status=CRITICAL; WARNING conditions set status=DEGRADED.
CRITICAL_CONDITIONS = {"CoolingCritical", "EccErrors", "ZombieContext"}
WARNING_CONDITIONS  = {"CoolingDegraded", "Throttling", "TelemetryStale"}
# Info conditions surface a fact but do NOT degrade status or schedulability.
# TelemetryUnavailable (vGPU guest with no temp/power) means "can't assess" — the
# GPU may be perfectly fine; don't drain a fleet because we can't read it.
INFO_CONDITIONS     = {"TelemetryUnavailable"}
ALL_CONDITIONS      = sorted(CRITICAL_CONDITIONS | WARNING_CONDITIONS | INFO_CONDITIONS)


@dataclass
class Condition:
    name:   str
    active: bool = False
    since:  Optional[float] = None     # when it last became active
    reason: str = ""
    message: str = ""


@dataclass
class GpuHealth:
    gpu_index:   int
    status:      HealthStatus
    schedulable: bool
    since:       Optional[float]              # when `status` was entered
    conditions:  list[Condition] = field(default_factory=list)  # ACTIVE only
    message:     str = ""

    def as_dict(self) -> dict:
        return {
            "gpu_index":   self.gpu_index,
            "status":      self.status.value,
            "schedulable": self.schedulable,
            "since":       self.since,
            "message":     self.message,
            "conditions":  [
                {"name": c.name, "since": c.since, "reason": c.reason, "message": c.message}
                for c in self.conditions
            ],
        }


@dataclass
class _GpuConditions:
    conds:        dict[str, Condition]
    status:       HealthStatus = HealthStatus.UNKNOWN
    status_since: Optional[float] = None
    observed:     bool = False


class HealthConditionTracker:
    """Per-GPU NPD-style health conditions, fed once per cycle by the daemon."""

    def __init__(self):
        self._gpus: dict[int, _GpuConditions] = {}

    def _g(self, gpu: int) -> _GpuConditions:
        if gpu not in self._gpus:
            self._gpus[gpu] = _GpuConditions(
                conds={n: Condition(n) for n in ALL_CONDITIONS})
        return self._gpus[gpu]

    def _set(self, g: _GpuConditions, name: str, active: bool,
             ts: float, reason: str = "", message: str = "") -> None:
        c = g.conds[name]
        if active and not c.active:
            c.since = ts                       # transition false→true: stamp it
        if not active:
            c.since = None
        c.active = active
        c.reason = reason if active else ""
        c.message = message if active else ""

    def observe(
        self,
        gpu: int,
        *,
        ts: float,
        warming: bool,
        state: GPUState,
        drift_warning: bool = False,
        drift_critical: bool = False,
        peer_flagged: bool = False,
        peer_critical: bool = False,
        throttling: bool = False,
        ecc_dbit: int = 0,
        telemetry_stale: bool = False,
        telemetry_unavailable: bool = False,
    ) -> None:
        g = self._g(gpu)
        g.observed = True

        # Telemetry unavailable (vGPU guest): we cannot assess this GPU. Surface
        # the fact, hold status at UNKNOWN, keep it schedulable — and skip the
        # R_θ-derived conditions, which would be meaningless without temp/power.
        if telemetry_unavailable:
            for n in ALL_CONDITIONS:
                self._set(g, n, n == "TelemetryUnavailable", ts,
                          "vgpu_no_telemetry", "vGPU guest — temperature/power not exposed; cannot assess")
            if g.status is not HealthStatus.UNKNOWN:
                g.status = HealthStatus.UNKNOWN
                g.status_since = ts
            return

        self._set(g, "TelemetryUnavailable", False, ts)
        cooling_critical = drift_critical or peer_critical or state == GPUState.CRITICAL
        cooling_degraded = (drift_warning or peer_flagged or state == GPUState.DRIFTING) \
            and not cooling_critical

        self._set(g, "CoolingCritical", cooling_critical, ts,
                  "rtheta_critical", "R_θ critically elevated vs baseline/peers")
        self._set(g, "CoolingDegraded", cooling_degraded, ts,
                  "rtheta_drift", "R_θ elevated vs baseline/peers at steady power")
        self._set(g, "ZombieContext", state == GPUState.ZOMBIE_RECOVERY, ts,
                  "cuda_context_retained", "GPU pinned at P0 with retained CUDA context")
        self._set(g, "Throttling", throttling, ts,
                  "clock_suppressed", "SM clock suppressed under load (micro-throttle)")
        self._set(g, "EccErrors", ecc_dbit > 0, ts,
                  "ecc_double_bit", f"{ecc_dbit} uncorrectable (double-bit) ECC error(s)")
        self._set(g, "TelemetryStale", telemetry_stale, ts,
                  "poll_latency", "NVML poll latency elevated — telemetry may be stale")

        # Derive overall status from active conditions.
        active_crit = [c for c in g.conds.values() if c.active and c.name in CRITICAL_CONDITIONS]
        active_warn = [c for c in g.conds.values() if c.active and c.name in WARNING_CONDITIONS]
        if active_crit:
            new_status = HealthStatus.CRITICAL
        elif active_warn:
            new_status = HealthStatus.DEGRADED
        elif warming:
            new_status = HealthStatus.WARMING
        else:
            new_status = HealthStatus.HEALTHY

        if new_status != g.status:
            g.status = new_status
            g.status_since = ts

    def health(self, gpu: int) -> GpuHealth:
        g = self._gpus.get(gpu)
        if g is None or not g.observed:
            return GpuHealth(gpu, HealthStatus.UNKNOWN, schedulable=True, since=None,
                             message="no observation yet")
        active = [c for c in g.conds.values() if c.active]
        # CRITICAL first, then warnings — for a stable, readable order.
        active.sort(key=lambda c: (c.name not in CRITICAL_CONDITIONS, c.name))
        schedulable = g.status not in (HealthStatus.DEGRADED, HealthStatus.CRITICAL)
        if g.status == HealthStatus.HEALTHY:
            msg = "healthy — no active problem conditions"
        elif g.status == HealthStatus.WARMING:
            msg = "warming up — establishing baseline, not yet confident"
        else:
            msg = "; ".join(c.message for c in active) or g.status.value
        return GpuHealth(gpu, g.status, schedulable, g.status_since, active, msg)

    def all(self) -> dict[int, GpuHealth]:
        return {gpu: self.health(gpu) for gpu in self._gpus}

    def fleet_summary(self) -> dict:
        """Roll-up for the /conditions endpoint and the readiness gauge."""
        hs = self.all()
        by_status: dict[str, int] = {}
        for h in hs.values():
            by_status[h.status.value] = by_status.get(h.status.value, 0) + 1
        return {
            "gpus": {str(g): h.as_dict() for g, h in hs.items()},
            "summary": {
                "total": len(hs),
                "schedulable": sum(1 for h in hs.values() if h.schedulable),
                "by_status": by_status,
            },
        }
