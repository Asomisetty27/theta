"""
Incident store — the flywheel substrate.

Theta's confidence numbers are physics priors until they are measured against
outcomes. This module is what makes the measurement possible: every time a GPU
crosses from healthy → diagnosed and later recovers, Theta records an
**Incident** — the feature vector *before*, the diagnosis + confidence tier, the
action taken (if known), the feature vector *after*, and, once an operator
confirms, the ground-truth **label**. That record is the difference between
"R_θ inference says TIM" and "we repasted GPU 6 and the slope normalized — TIM
confirmed."

Two properties make this the real moat rather than just logging:

  1. It captures `features_before/after` automatically around recoveries Theta
     already detects — so the labeled-pattern library starts filling before a
     single operator types anything.
  2. The confirmed label is the only thing allowed to elevate an incident to
     CONFIRMED_CAUSE (see `causal.ConfidenceTier`). Passive telemetry never
     reaches that tier on its own. This is the "no fake 100%" rule, enforced in
     storage rather than vibes.

Storage is JSONL at a configurable path (default ~/.theta/incidents.jsonl), one
JSON object per line, last-write-wins by id. All time and ids are injected so
the store is deterministically testable with no clock or randomness.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Incident lifecycle stages, for the operator-facing summary.
STAGE_OPEN      = "open"        # diagnosed, no action recorded yet
STAGE_ACTIONED  = "actioned"    # a remediation was applied, awaiting recovery
STAGE_RESOLVED  = "resolved"    # metrics recovered, no confirmed label yet
STAGE_CONFIRMED = "confirmed"   # operator attached a ground-truth label


def _default_path() -> Path:
    root = os.environ.get("THETA_HOME", str(Path.home() / ".theta"))
    return Path(root) / "incidents.jsonl"


@dataclass
class Incident:
    """One detected-and-tracked GPU health episode, open through confirmed."""
    id:              str
    gpu_uuid:        str
    gpu_index:       int
    node:            str
    opened_at:       float
    cause:           str            # FaultCause.value at open time
    confidence:      float
    tier:            str            # ConfidenceTier.value (passive: ≤ "high")
    headline:        str
    features_before: dict = field(default_factory=dict)

    # Filled as the episode progresses.
    action_taken:    Optional[str]   = None
    action_at:       Optional[float] = None
    features_after:  Optional[dict]  = None
    resolved:        bool            = False
    resolved_at:     Optional[float] = None
    confirmed_label: Optional[str]   = None   # operator ground truth (cause string)
    labeled_at:      Optional[float] = None
    notes:           Optional[str]   = None

    @property
    def stage(self) -> str:
        if self.confirmed_label:
            return STAGE_CONFIRMED
        if self.resolved:
            return STAGE_RESOLVED
        if self.action_taken:
            return STAGE_ACTIONED
        return STAGE_OPEN

    @property
    def effective_tier(self) -> str:
        """
        A confirmed label is the *only* thing that elevates to confirmed_cause —
        passive telemetry tops out at whatever tier the reasoning engine assigned.
        """
        if self.confirmed_label and self.resolved:
            return "confirmed_cause"
        return self.tier

    @property
    def prediction_was_correct(self) -> Optional[bool]:
        """
        True/False once a label exists, else None. This is the column that turns
        the store into a precision measurement: count correct / total labeled.
        """
        if not self.confirmed_label:
            return None
        return self.confirmed_label == self.cause

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Incident":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class IncidentStore:
    """
    JSONL-backed incident log. Small by design — rewrites the whole file on each
    mutation (atomic temp-file + rename), which is fine for the hundreds-of-rows
    scale a fleet generates and keeps the format trivially greppable.
    """

    def __init__(
        self,
        path: Optional[os.PathLike | str] = None,
        *,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex[:12],
    ) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self._id_factory = id_factory
        self._incidents: dict[str, Incident] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    inc = Incident.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue  # skip a corrupt line rather than crash the agent
                self._incidents[inc.id] = inc

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for inc in self._incidents.values():
                    fh.write(json.dumps(inc.to_dict(), separators=(",", ":")) + "\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ── lifecycle ────────────────────────────────────────────────────────
    def open_incident(
        self,
        *,
        gpu_uuid: str,
        gpu_index: int,
        node: str,
        cause: str,
        confidence: float,
        tier: str,
        headline: str,
        features_before: dict,
        now: float,
        incident_id: Optional[str] = None,
    ) -> Incident:
        inc = Incident(
            id=incident_id or self._id_factory(),
            gpu_uuid=gpu_uuid,
            gpu_index=gpu_index,
            node=node,
            opened_at=now,
            cause=cause,
            confidence=confidence,
            tier=tier,
            headline=headline,
            features_before=dict(features_before),
        )
        self._incidents[inc.id] = inc
        self._flush()
        return inc

    def record_action(self, incident_id: str, action: str, now: float) -> Optional[Incident]:
        inc = self._incidents.get(incident_id)
        if inc is None:
            return None
        inc.action_taken = action
        inc.action_at = now
        self._flush()
        return inc

    def record_recovery(self, incident_id: str, features_after: dict, now: float) -> Optional[Incident]:
        inc = self._incidents.get(incident_id)
        if inc is None:
            return None
        inc.features_after = dict(features_after)
        inc.resolved = True
        inc.resolved_at = now
        self._flush()
        return inc

    def label(
        self,
        incident_id: str,
        confirmed_label: str,
        now: float,
        notes: Optional[str] = None,
    ) -> Optional[Incident]:
        """Attach operator ground truth. This is the CONFIRMED_CAUSE gate."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return None
        inc.confirmed_label = confirmed_label
        inc.labeled_at = now
        if notes:
            inc.notes = notes
        self._flush()
        return inc

    # ── queries ──────────────────────────────────────────────────────────
    def get(self, incident_id: str) -> Optional[Incident]:
        return self._incidents.get(incident_id)

    def all(self) -> list[Incident]:
        return sorted(self._incidents.values(), key=lambda i: i.opened_at, reverse=True)

    def open_for_gpu(self, gpu_uuid: str) -> Optional[Incident]:
        """The most recent unresolved incident for a GPU, if any."""
        for inc in self.all():
            if inc.gpu_uuid == gpu_uuid and not inc.resolved:
                return inc
        return None

    def unlabeled_resolved(self) -> list[Incident]:
        """Resolved incidents awaiting an operator label — the work queue."""
        return [i for i in self.all() if i.resolved and not i.confirmed_label]

    def accuracy(self) -> dict:
        """
        Measured cause accuracy over labeled incidents. Returns counts and a
        rate, or {'labeled': 0} when there is nothing to measure yet — Theta
        must never report an accuracy figure it has not earned.
        """
        labeled = [i for i in self.all() if i.confirmed_label]
        if not labeled:
            return {"labeled": 0, "correct": 0, "rate": None}
        correct = sum(1 for i in labeled if i.prediction_was_correct)
        return {"labeled": len(labeled), "correct": correct, "rate": correct / len(labeled)}


# ──────────────────────────────────────────────────────────────────────────
# Daemon integration — kept here (and unit-tested) so the daemon hook is a
# couple of guarded lines rather than embedded lifecycle logic.
# ──────────────────────────────────────────────────────────────────────────

# Tiers/urgencies at or above which an episode is worth tracking.
_TRACK_TIERS = {"probable", "high", "confirmed_subsystem", "confirmed_cause"}
_TRACK_URGENCY = {"act_soon", "act_now", "emergency"}
_HEALTHY_CAUSES = {"nominal", "insufficient_data"}


def update_from_diagnosis(
    store: IncidentStore,
    *,
    gpu_uuid: str,
    gpu_index: int,
    node: str,
    causal_dict: dict,
    features: dict,
    now: float,
) -> Optional[Incident]:
    """
    Reconcile one fresh diagnosis against the store: open a new incident when a
    GPU starts failing, or close the open one when it returns to health. Safe to
    call every cycle — it no-ops when nothing changed.

    `causal_dict` is `CausalExplanation.as_dict()`. `features` is whatever
    feature snapshot the caller wants preserved as before/after (R_θ, slope,
    intercept, fan, inlet, clock-eff, fabric rates, ...).
    """
    cause = causal_dict.get("hypothesis", {}).get("cause", "nominal")
    tier = causal_dict.get("tier", "unconfirmed")
    urgency = causal_dict.get("urgency", "info")
    confidence = causal_dict.get("hypothesis", {}).get("confidence", 0.0)
    headline = causal_dict.get("headline", "")

    open_inc = store.open_for_gpu(gpu_uuid)
    healthy = cause in _HEALTHY_CAUSES and urgency not in _TRACK_URGENCY
    worth_tracking = tier in _TRACK_TIERS or urgency in _TRACK_URGENCY

    if open_inc is None:
        if worth_tracking and not healthy:
            return store.open_incident(
                gpu_uuid=gpu_uuid, gpu_index=gpu_index, node=node,
                cause=cause, confidence=confidence, tier=tier,
                headline=headline, features_before=features, now=now,
            )
        return None

    # An incident is already open for this GPU.
    if healthy:
        return store.record_recovery(open_inc.id, features, now)
    return None
