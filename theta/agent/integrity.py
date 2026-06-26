"""
Telemetry-integrity gate (Stage 1 of the fault-identification ladder).

Before Theta diagnoses a GPU, it has to trust the data. A hung GPU, a GPU that
fell off the bus, a broken DCGM scrape, or an implausible sensor reading must
produce "diagnosis blocked: telemetry unreliable" — NOT a confident R_θ story
about a number that is itself garbage. Diagnosing bad telemetry is how a
monitoring tool earns a reputation for crying wolf.

This module is pure: given a snapshot of integrity signals the daemon already
tracks (poll latency, staleness, whether R_θ is even computable, basic sensor
plausibility), it returns a verdict. The daemon blocks diagnosis when the
verdict says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Conservative defaults. A blocked diagnosis is cheap; a wrong one is expensive,
# but a gate that fires on every P-state transition is worse than no gate. Tune
# against real fleet poll-latency distributions.
MAX_POLL_LATENCY_S = 2.0     # NVML/DCGM call taking longer ⇒ GPU/driver not responding
MAX_POWER_W        = 2000.0  # above any real single-GPU board ⇒ sensor fault
MIN_POWER_W        = 0.0     # negative power ⇒ sensor fault


# Machine-readable block reasons (also the operator-facing "likely issue").
BLOCK_COLLECTOR   = "collector_or_driver_failure"
BLOCK_UNRESPONSIVE = "gpu_unresponsive"
BLOCK_OFF_BUS     = "gpu_off_bus_or_reenumerated"
BLOCK_STALE       = "telemetry_stale"
BLOCK_SENSOR      = "sensor_implausible"


@dataclass
class IntegritySignals:
    """Everything the gate needs — all of it already tracked by the daemon."""
    collector_ok:   bool  = True   # NVML/DCGM call returned this cycle
    uuid_stable:    bool  = True   # GPU UUID unchanged since last cycle
    stale:          bool  = False  # daemon flagged telemetry as stale for this GPU
    poll_latency_s: float = 0.0    # rolling poll latency for this GPU
    power_w:        Optional[float] = None  # for plausibility; None = unknown (skip check)
    rtheta_computable: bool = True  # False when R_θ denominator is unusable


@dataclass
class IntegrityVerdict:
    trustworthy: bool
    score:       float            # 0..1, 1 = pristine
    reason:      str              # human-readable
    blocked_cause: Optional[str]  # one of BLOCK_*, or None when trustworthy

    @property
    def blocked(self) -> bool:
        return not self.trustworthy


def assess_integrity(sig: IntegritySignals) -> IntegrityVerdict:
    """
    Return a trust verdict for one GPU's telemetry this cycle. Hard failures
    block; soft signals erode the score but still pass.

    Note `rtheta_computable=False` alone does NOT block — a GPU at clean idle
    legitimately has too little power to compute R_θ. It only blocks when paired
    with another fault (handled by the daemon, which knows the context).
    """
    if not sig.collector_ok:
        return IntegrityVerdict(False, 0.0,
            "Collector/driver returned no data this cycle.", BLOCK_COLLECTOR)

    if not sig.uuid_stable:
        return IntegrityVerdict(False, 0.0,
            "GPU UUID changed — the device fell off the bus or re-enumerated.", BLOCK_OFF_BUS)

    if sig.poll_latency_s > MAX_POLL_LATENCY_S:
        return IntegrityVerdict(False, 0.1,
            f"Poll latency {sig.poll_latency_s:.1f}s exceeds {MAX_POLL_LATENCY_S:.0f}s — "
            f"the GPU is not responding promptly.", BLOCK_UNRESPONSIVE)

    if sig.power_w is not None and not (MIN_POWER_W <= sig.power_w <= MAX_POWER_W):
        return IntegrityVerdict(False, 0.1,
            f"Power reading {sig.power_w:.0f}W is outside the plausible range — sensor fault.",
            BLOCK_SENSOR)

    if sig.stale:
        return IntegrityVerdict(False, 0.2,
            "Telemetry is stale — samples are not advancing.", BLOCK_STALE)

    # Trustworthy, but degrade the score as latency approaches the limit so the
    # health API can show "trusted but slowing".
    score = 1.0 - min(0.5, sig.poll_latency_s / (2.0 * MAX_POLL_LATENCY_S))
    return IntegrityVerdict(True, round(score, 3), "Telemetry within trusted bounds.", None)


def blocked_explanation(gpu_index: int, verdict: IntegrityVerdict) -> dict:
    """
    A CausalExplanation-shaped dict for a blocked diagnosis. Same keys the site
    and alert layer already consume from `CausalExplanation.as_dict()`, plus a
    `telemetry_blocked` flag and `block_cause`, so downstream never confuses a
    blocked GPU with a healthy one.
    """
    return {
        "headline": f"GPU {gpu_index}: diagnosis blocked — telemetry unreliable ({verdict.reason})",
        "urgency": "watch",
        "tier": "unconfirmed",
        "telemetry_blocked": True,
        "block_cause": verdict.blocked_cause,
        "hypothesis": {
            "cause": "insufficient_data",
            "confidence": 0.0,
            "one_line": "Cannot diagnose the GPU until its telemetry is trustworthy.",
        },
        "alternatives": [],
        "evidence": [
            {"name": "telemetry_integrity", "value": verdict.reason, "weight": 1.0},
        ],
        "actions": [
            {
                "title": f"Restore telemetry for GPU {gpu_index}",
                "detail": (
                    "Check the collector/driver (nvidia-smi responds?), DCGM scrape health, "
                    "and whether the GPU is still enumerated on the PCIe bus. Resume diagnosis "
                    "only once samples advance with plausible values."
                ),
                "effort": "varies",
                "expected_impact": "Unblocks diagnosis; rules out a phantom thermal alert.",
                "blocks_workload": False,
                "integration": None,
            }
        ],
        "when_started": None,
        "eta_to_threshold": None,
        "eta_to_recovery": None,
    }
