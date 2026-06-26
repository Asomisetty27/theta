"""
AlertGovernor — the trust layer between detection and the operator's inbox.

Theta's detectors are sophisticated but most are validated on one GPU class. The
thing that actually decides whether an operator keeps the agent installed is NOT
how clever the detectors are — it is whether the FIRST hour on a stranger's fleet
produces a true signal and ZERO false alarms. One "Theta cried wolf" thread kills
OSS adoption. This module is the discipline that earns first-run trust.

It composes three best-in-class patterns:

1. First-run WARMING (Netdata-watchdog / node-problem-detector style).
   On a GPU whose baseline is not yet established, INFERENTIAL alerts (anything
   derived from R_θ statistics — drift, peer, fault-curve, the unsupervised critic)
   are HELD, not fired. The agent's honest posture is "learning your baseline, not
   yet confident," not a guess dressed as an alarm. Ground-truth hardware events
   (ECC, Xid, throttle, poll-latency) bypass warming — they are facts, not inferences.

2. Severity INHIBITION (Prometheus Alertmanager style).
   While a CRITICAL is active for a GPU, concurrent lower-severity alerts for the
   same GPU are suppressed — the operator should see "GPU 3 critical," not also
   five warnings about the same GPU. (Dedup, the orthogonal Alertmanager pattern,
   already lives in alerter.py's _AlertDeduper; the governor sits in front of it.)

3. False-positive BUDGET / circuit breaker (the novel safety net).
   A rolling per-GPU alert-rate budget. If INFERENTIAL alerts on one GPU exceed the
   budget, that is itself evidence the agent is mis-calibrated for this hardware
   (T4-derived thresholds on an unprofiled GPU, say). The governor TRIPS a breaker:
   further inferential alerts on that GPU are suppressed and ONE meta-alert fires —
   "Theta is firing more than expected on GPU N; likely mis-calibrated, run
   `theta calibrate`." Better to go quiet and say so than to spray wrong alarms.

Ground-truth hardware faults are NEVER held, inhibited, or budget-suppressed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .metrics import AlertEvent, GPUState


# Alert kinds the governor treats as INFERENTIAL (R_θ-statistics-derived). Anything
# not in here is treated as ground-truth and bypasses warming/inhibition/budget.
INFERENTIAL_DETECTORS = {"drift", "peer_relative", "fault_curve", "critic", "median_polish"}


class Action(Enum):
    ROUTE              = auto()   # send it
    HOLD_WARMING       = auto()   # GPU not yet confident — hold inferential alert
    SUPPRESS_INHIBITED = auto()   # a critical is active on this GPU
    SUPPRESS_BUDGET    = auto()   # FP-budget breaker tripped for this GPU
    HOLD_CONSENSUS     = auto()   # only one inferential detector agrees so far — wait for a second


@dataclass
class Decision:
    action:     Action
    reason:     str
    meta_alert: Optional[AlertEvent] = None   # e.g. the one-shot "mis-calibrated" notice


@dataclass
class _GpuPosture:
    seen_since:      Optional[float] = None
    ready:           bool            = False   # baseline established (set by daemon)
    alert_times:     deque           = field(default_factory=deque)  # inferential alert ts
    breaker_tripped: bool            = False
    breaker_since:   Optional[float] = None
    last_critical:   Optional[float] = None
    votes:           dict            = field(default_factory=dict)   # detector name → last vote ts


def _severity(event: AlertEvent) -> str:
    """info < warning < critical. Prefer explicit context, else map from state."""
    ctx = getattr(event, "context", None)
    if isinstance(ctx, dict) and ctx.get("severity"):
        return ctx["severity"]
    if event.state == GPUState.CRITICAL:
        return "critical"
    if event.state in (GPUState.DRIFTING, GPUState.ZOMBIE_RECOVERY):
        return "warning"
    return "info"


def _detector_of(event: AlertEvent) -> Optional[str]:
    ctx = getattr(event, "context", None)
    return ctx.get("detector") if isinstance(ctx, dict) else None


_SEV_RANK = {"info": 0, "warning": 1, "critical": 2}


class AlertGovernor:
    """First-run trust + FP-budget governor. Stateful, single-threaded (one daemon)."""

    def __init__(
        self,
        warmup_sec:        float = 120.0,   # below this AND not ready → hold inferential
        budget_count:      int   = 12,      # inferential alerts per GPU per window
        budget_window_sec: float = 3600.0,  # rolling budget window (1h)
        inhibit_sec:       float = 300.0,   # a critical inhibits lower-sev for this long
        breaker_cooldown:  float = 1800.0,  # breaker stays tripped this long, then re-arms
        consensus_min:     int   = 1,       # distinct inferential detectors that must agree
        consensus_window_sec: float = 300.0,  # window over which agreement counts
    ):
        self._warmup_sec        = warmup_sec
        self._budget_count      = budget_count
        self._budget_window_sec = budget_window_sec
        self._inhibit_sec       = inhibit_sec
        self._breaker_cooldown  = breaker_cooldown
        # Consensus gate (Netdata-style unanimous-bit FP control, generalised to
        # K-of-N). consensus_min=1 is pass-through (default — preserves the
        # single-detector contract). The daemon raises it to 2 so a sub-critical
        # degradation warning must be corroborated by a SECOND independent
        # inferential detector (e.g. temporal drift AND cross-sectional peer, or
        # AND median-polish / fault-curve / critic) before it routes. CRITICAL
        # severities bypass the gate, so a worsening degradation still escalates
        # immediately even from a single detector — the gate trades a little
        # warning-latency for far fewer one-off false positives, never recall on
        # severe events.
        self._consensus_min        = max(1, consensus_min)
        self._consensus_window_sec = consensus_window_sec
        self._gpus: dict[int, _GpuPosture] = {}
        # observability counters (exported by the daemon)
        self.counts = {a.name: 0 for a in Action}

    def _posture(self, gpu: int) -> _GpuPosture:
        return self._gpus.setdefault(gpu, _GpuPosture())

    def note_cycle(self, gpu: int, *, ready: bool, ts: float) -> None:
        """Called once per GPU per polling cycle to track first-seen + readiness."""
        p = self._posture(gpu)
        if p.seen_since is None:
            p.seen_since = ts
        p.ready = ready
        # Re-arm the breaker after its cooldown so a transiently-noisy GPU recovers.
        if p.breaker_tripped and p.breaker_since is not None \
                and ts - p.breaker_since >= self._breaker_cooldown:
            p.breaker_tripped = False
            p.breaker_since = None
            p.alert_times.clear()

    def is_warming(self, gpu: int, ts: float) -> bool:
        p = self._posture(gpu)
        if p.ready:
            return False
        if p.seen_since is None:
            return True
        return (ts - p.seen_since) < self._warmup_sec

    def readiness(self, gpu: int) -> float:
        """1.0 = confident, 0.0 = warming/tripped — for the readiness gauge."""
        p = self._posture(gpu)
        if p.breaker_tripped:
            return 0.0
        return 1.0 if p.ready else 0.0

    def evaluate(self, event: AlertEvent, ts: float) -> Decision:
        """Gate one alert. Ground-truth hardware faults always ROUTE."""
        gpu = event.gpu_index
        detector = _detector_of(event)
        inferential = detector in INFERENTIAL_DETECTORS
        sev = _severity(event)
        p = self._posture(gpu)

        # Track active critical for inhibition (any source, incl. ground-truth).
        if sev == "critical":
            p.last_critical = ts

        # Ground-truth hardware faults bypass the trust gates entirely.
        if not inferential:
            return self._done(Action.ROUTE, "ground_truth fault — bypasses governor")

        # 1. Warming: hold inferential alerts until the GPU is confident.
        if self.is_warming(gpu, ts):
            return self._done(Action.HOLD_WARMING,
                              f"GPU {gpu} still warming (baseline not established)")

        # 2. Budget breaker already tripped → stay quiet.
        if p.breaker_tripped:
            return self._done(Action.SUPPRESS_BUDGET,
                              f"GPU {gpu} FP-budget breaker tripped — suppressing")

        # 3. Inhibition: suppress sub-critical while a critical is active on this GPU.
        if sev != "critical" and p.last_critical is not None \
                and ts - p.last_critical < self._inhibit_sec:
            return self._done(Action.SUPPRESS_INHIBITED,
                              f"GPU {gpu} has an active critical — inhibiting {sev}")

        # 3b. Consensus gate. Record this detector's vote, expire stale ones, and
        # count how many DISTINCT inferential detectors agree within the window.
        # A sub-critical degradation routes only once a second independent
        # detector corroborates; CRITICAL bypasses (worsening events escalate
        # immediately). consensus_min=1 makes this a no-op (default).
        if detector:
            p.votes[detector] = ts
        for d, vts in list(p.votes.items()):
            if ts - vts > self._consensus_window_sec:
                del p.votes[d]
        if self._consensus_min > 1 and sev != "critical" \
                and len(p.votes) < self._consensus_min:
            return self._done(
                Action.HOLD_CONSENSUS,
                f"GPU {gpu} {sev} from '{detector}' alone "
                f"({len(p.votes)}/{self._consensus_min} detectors) — holding for corroboration")

        # 4. Budget accounting: record this inferential alert, prune the window.
        p.alert_times.append(ts)
        while p.alert_times and ts - p.alert_times[0] > self._budget_window_sec:
            p.alert_times.popleft()

        if len(p.alert_times) > self._budget_count:
            # Trip the breaker and emit ONE meta-alert.
            p.breaker_tripped = True
            p.breaker_since = ts
            meta = AlertEvent(
                gpu_index=gpu, timestamp=ts,
                state=GPUState.UNKNOWN, prev_state=GPUState.UNKNOWN,
                rtheta=None, rtheta_baseline=None, drift_sigma=None,
                confidence=1.0,
                message=(
                    f"Theta tripped its false-positive breaker on GPU {gpu}: "
                    f"{len(p.alert_times)} inferential alerts in "
                    f"{int(self._budget_window_sec/60)} min exceeds the budget of "
                    f"{self._budget_count}. This usually means the R_θ thresholds are "
                    f"mis-calibrated for this hardware. Further inferential alerts on "
                    f"GPU {gpu} are suppressed for {int(self._breaker_cooldown/60)} min. "
                    f"Recommended: run `theta calibrate --gpu {gpu}`."
                ),
                context={"severity": "warning", "detector": "governor",
                         "breaker": "tripped", "budget": self._budget_count,
                         "observed": len(p.alert_times)},
            )
            return self._done(Action.SUPPRESS_BUDGET,
                              f"GPU {gpu} exceeded FP budget — breaker tripped",
                              meta_alert=meta)

        return self._done(Action.ROUTE, "confident — routed")

    def _done(self, action: Action, reason: str,
              meta_alert: Optional[AlertEvent] = None) -> Decision:
        self.counts[action.name] += 1
        return Decision(action=action, reason=reason, meta_alert=meta_alert)
