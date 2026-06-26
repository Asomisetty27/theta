"""
Signature-matrix classifier — exact-cause identification by multi-axis fingerprint.

The core idea: stop treating R_θ as a scalar ("is it high?") and treat each
degradation mode as a **fingerprint vector** across orthogonal axes that load
differently per fault:

    time-domain shape · power-conditioning (α vs β) · locality · channel
    (thermal/power/memory/fabric/structural) · cross-correlations · recovery τ

Most degradation modes occupy a *unique coordinate* in that space, so they
separate cleanly even though no single metric distinguishes them. Dust and TIM
both raise R_θ slowly — but dust raises the intercept α while TIM steepens the
slope β. Fan-bearing wear and airflow blockage both raise α on one GPU — but the
fan's RPM-vs-duty residual splits them. Mounting and TIM both hit the conduction
path — but one is a *step at a service event* and the other a slow ramp.

The second, equally important idea: **three-valued evidence**. Every axis test
returns SUPPORTS / CONTRADICTS / UNKNOWN. UNKNOWN is not a failure — it is the
engine admitting the data didn't exercise that axis (no power range → β
unobservable; no fan telemetry → RPM residual unobservable). When two modes tie
because the axis that *would* separate them came back UNKNOWN, the classifier
emits the **missing axis** by name, and how to get it (a sensor, a probe, a
workload condition, a service log). That is the honest realization of "identify
the exact cause on any GPU": exact when the fingerprint is unique on observed
axes, and *exact-pending-X* with X specified when it isn't.

CALIBRATION: every numeric threshold below is a first-principles placeholder
marked `# CALIBRATE`. They need fitting against real data — the Stage-1 set, the
Noyce AI Factory fleet, and deliberately-degraded units on the E-LT testbed.
The *structure* (which axes separate which modes) is the engineering claim; the
*thresholds* are the empirical surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .fault_classifier import FaultCause


# ──────────────────────────────────────────────────────────────────────────
# Calibration surface — thresholds to fit against real data.
# ──────────────────────────────────────────────────────────────────────────
ELEVATED_Z      = 2.0     # CALIBRATE: a z-scored signal at/above this is "elevated"
NORMAL_Z        = 1.5     # CALIBRATE: at/below this a signal is "within normal band"
FAN_DEFICIT     = -0.10   # CALIBRATE: RPM running ≥10% under commanded duty = underperforming
FAN_OK          = -0.05   # CALIBRATE: RPM tracking duty within 5% = healthy fan
FABRIC_ERR_PER_S = 1.0    # CALIBRATE: NVLink/PCIe errors per second that count as a live fault
POWER_VIOL_FRAC = 0.10    # CALIBRATE: fraction of interval power-throttled that counts
CLOCK_EFF_LOW   = 0.90    # CALIBRATE: SM clock below this fraction of boost = suppressed
DRAM_HOT        = 0.50    # CALIBRATE: dram-active fraction above which HBM load is "heavy"

# Decision geometry.
SCORE_FLOOR     = 0.50    # below this top score → call it NOMINAL (nothing convincing)
STRONG_SCORE    = 0.75    # at/above this (and discriminators observed) → exact cause
DEGENERACY_EPS  = 0.15    # top modes within this score band are treated as tied
MIN_COVERAGE    = 0.34    # a mode needs at least this fraction of its axes observed to rank


class Axis(Enum):
    TIME      = "time_shape"
    POWER     = "power_conditioning"   # α (intercept) vs β (slope)
    LOCALITY  = "locality"
    CHANNEL   = "channel"
    CORRELATE = "cross_correlation"
    RECOVERY  = "recovery_dynamics"


class Tristate(Enum):
    SUPPORTS    = "supports"
    CONTRADICTS = "contradicts"
    UNKNOWN     = "unknown"       # the data didn't exercise this axis


@dataclass
class FeatureVector:
    """
    One GPU's observed fingerprint. `None` means *unobserved this window* (the
    axis wasn't exercised / the sensor isn't present), which is distinct from a
    measured zero. That distinction is what powers the missing-axis ledger.

    α and β are deliberately `None` unless `power_range_observed` — you cannot
    decompose intercept from slope without the workload spanning power tiers, so
    a flat workload leaves both unobserved and only `rtheta_overall_z` is known.
    """
    # Thermal magnitude / decomposition
    rtheta_overall_z:  Optional[float] = None   # overall R_θ deviation (needs baseline only)
    power_range_observed: bool         = False  # did the workload span power tiers?
    alpha_z:           Optional[float] = None   # intercept deviation (None unless power range)
    beta_z:            Optional[float] = None   # slope deviation     (None unless power range)
    # Time-domain shape
    drift_rate_z:      Optional[float] = None   # slow monotonic ramp magnitude
    step_detected:     Optional[bool]  = None   # discrete intra/inter-session jump
    near_service_event: Optional[bool] = None   # step coincides with a maintenance window
    # Locality / cross-section
    locality:          str             = "unknown"   # single|node|rack|slot|unknown
    # Cross-correlations / channels
    fan_rpm_residual:  Optional[float] = None   # actual RPM vs commanded duty (−ve = under)
    inlet_delta_z:     Optional[float] = None   # inlet-temp rise (Redfish/BMC)
    mem_core_delta_z:  Optional[float] = None   # ΔT(mem − core) deviation
    dram_active:       Optional[float] = None   # memory-interface active fraction 0..1
    ecc_sbe_rate:      Optional[float] = None   # single-bit ECC errors per hour
    nvlink_error_rate: float           = 0.0    # errors/sec (always observed; 0 = none)
    pcie_replay_rate:  float           = 0.0    # replays/sec
    power_violation_rate: float        = 0.0    # fraction of interval power-throttled
    clock_efficiency:  float           = 1.0    # SM clock / boost
    recovery_tau_z:    Optional[float] = None   # cooldown time-constant deviation
    perf_per_watt_z:   Optional[float] = None   # efficiency drift (silicon aging)


# ── predicate plumbing ────────────────────────────────────────────────────

Getter = Callable[[FeatureVector], Optional[float]]
Test   = Callable[[FeatureVector], Tristate]


@dataclass(frozen=True)
class Predicate:
    axis:   Axis
    name:   str                       # human-readable evidence line
    weight: float
    test:   Test
    key:    bool = False              # a discriminating axis (drives the missing-axis ledger)
    needs:  Optional[str] = None      # what observation would resolve it, if UNKNOWN
    via:    Optional[str] = None      # how to get it: sensor | probe | workload | service-log
    splits: Optional[FaultCause] = None  # the *competing cause* this axis rules out. When set and
                                         # UNKNOWN, identifiability is blocked (a real alternative
                                         # cause survives). When None, an UNKNOWN `needs` is only a
                                         # sub-class refinement (more precise repair, same subsystem).


def _hi(get: Getter, thr: float) -> Test:
    def f(fv: FeatureVector) -> Tristate:
        v = get(fv)
        if v is None:
            return Tristate.UNKNOWN
        return Tristate.SUPPORTS if v >= thr else Tristate.CONTRADICTS
    return f


def _lo(get: Getter, thr: float) -> Test:
    def f(fv: FeatureVector) -> Tristate:
        v = get(fv)
        if v is None:
            return Tristate.UNKNOWN
        return Tristate.SUPPORTS if v <= thr else Tristate.CONTRADICTS
    return f


def _flag(get: Callable[[FeatureVector], Optional[bool]], want: bool) -> Test:
    def f(fv: FeatureVector) -> Tristate:
        v = get(fv)
        if v is None:
            return Tristate.UNKNOWN
        return Tristate.SUPPORTS if v is want else Tristate.CONTRADICTS
    return f


def _loc(allowed: set[str]) -> Test:
    def f(fv: FeatureVector) -> Tristate:
        if fv.locality in (None, "unknown"):
            return Tristate.UNKNOWN
        return Tristate.SUPPORTS if fv.locality in allowed else Tristate.CONTRADICTS
    return f


# ──────────────────────────────────────────────────────────────────────────
# The signature matrix — each mode's fingerprint as weighted, axis-tagged
# predicates. `key=True` marks the axis that *separates* this mode from its
# look-alikes; when such a predicate is UNKNOWN, the missing-axis ledger fires.
# ──────────────────────────────────────────────────────────────────────────

# Shared "this is a thermal R_θ rise at all" gate — without it, NOMINAL wins.
def _thermal_rise(weight: float = 1.0) -> Predicate:
    return Predicate(Axis.POWER, "overall R_θ elevated above baseline", weight,
                     _hi(lambda f: f.rtheta_overall_z, ELEVATED_Z))


# The α/β decomposition is the discriminator among thermal modes, and it is
# UNKNOWN without power range — the single most common identifiability gap.
_NEEDS_POWER_RANGE = "R_θ(P) slope — needs a high-power workload or a power-sweep probe"


SIGNATURES: dict[FaultCause, list[Predicate]] = {
    FaultCause.DUST_ACCUMULATION: [
        _thermal_rise(),
        Predicate(Axis.POWER, "intercept α elevated (uniform rise)", 1.0,
                  _hi(lambda f: f.alpha_z, ELEVATED_Z),
                  key=True, needs=_NEEDS_POWER_RANGE, via="workload",
                  splits=FaultCause.TIM_DEGRADATION),
        Predicate(Axis.TIME, "slow monotonic drift over days", 1.0,
                  _hi(lambda f: f.drift_rate_z, ELEVATED_Z)),
    ],
    FaultCause.TIM_DEGRADATION: [
        _thermal_rise(),
        Predicate(Axis.POWER, "slope β steepening (worse at high power)", 1.5,
                  _hi(lambda f: f.beta_z, ELEVATED_Z),
                  key=True, needs=_NEEDS_POWER_RANGE, via="workload",
                  splits=FaultCause.DUST_ACCUMULATION),
        Predicate(Axis.POWER, "intercept α within normal band", 0.6,
                  _lo(lambda f: f.alpha_z, NORMAL_Z),
                  key=True, needs=_NEEDS_POWER_RANGE, via="workload",
                  splits=FaultCause.DUST_ACCUMULATION),
        Predicate(Axis.TIME, "slow ramp, not a step", 0.4,
                  _flag(lambda f: f.step_detected, False)),
        # vs cold-plate/coolant contact: identical at the die without a second
        # sensor between die and coolant. Always surface the gap on TIM.
        Predicate(Axis.CHANNEL, "no cold-plate/coolant sensor to rule out contact loss", 0.0,
                  lambda f: Tristate.UNKNOWN,
                  key=True, needs="cold-plate / coolant-outlet temperature",
                  via="sensor"),
    ],
    FaultCause.MOUNTING_EVENT: [
        Predicate(Axis.TIME, "discrete step in R_θ", 1.5,
                  _flag(lambda f: f.step_detected, True)),
        Predicate(Axis.TIME, "step coincides with a service event", 1.0,
                  _flag(lambda f: f.near_service_event, True),
                  key=True, needs="maintenance/service log correlation", via="service-log"),
        Predicate(Axis.LOCALITY, "single GPU affected", 0.4, _loc({"single"})),
    ],
    FaultCause.FAN_BEARING_WEAR: [
        _thermal_rise(),
        Predicate(Axis.CORRELATE, "RPM running under commanded duty", 1.5,
                  _lo(lambda f: f.fan_rpm_residual, FAN_DEFICIT),
                  key=True, needs="fan RPM vs commanded duty", via="sensor",
                  splits=FaultCause.AIRFLOW_BLOCKAGE),
        Predicate(Axis.RECOVERY, "slow cooldown (airflow-limited)", 0.6,
                  _hi(lambda f: f.recovery_tau_z, ELEVATED_Z),
                  needs="a load→idle transition or cooldown probe", via="probe"),
    ],
    FaultCause.AIRFLOW_BLOCKAGE: [
        _thermal_rise(),
        Predicate(Axis.POWER, "intercept α elevated", 1.0,
                  _hi(lambda f: f.alpha_z, ELEVATED_Z),
                  needs=_NEEDS_POWER_RANGE, via="workload"),
        Predicate(Axis.TIME, "sudden onset (step)", 0.8,
                  _flag(lambda f: f.step_detected, True)),
        Predicate(Axis.CORRELATE, "fan RPM tracking duty normally", 1.0,
                  _hi(lambda f: f.fan_rpm_residual, FAN_OK),
                  key=True, needs="fan RPM vs commanded duty", via="sensor",
                  splits=FaultCause.FAN_BEARING_WEAR),
        Predicate(Axis.LOCALITY, "single GPU or slot pattern", 0.5, _loc({"single", "slot"})),
    ],
    FaultCause.HBM_THERMAL: [
        Predicate(Axis.CORRELATE, "memory-to-core ΔT elevated", 1.5,
                  _hi(lambda f: f.mem_core_delta_z, ELEVATED_Z),
                  key=True, needs="HBM/memory temperature", via="sensor"),
        Predicate(Axis.CORRELATE, "correlated with memory-bandwidth load", 0.8,
                  _hi(lambda f: f.dram_active, DRAM_HOT),
                  needs="DCGM dram-active profiling", via="sensor"),
        Predicate(Axis.CHANNEL, "single-bit ECC rate rising", 0.5,
                  _hi(lambda f: f.ecc_sbe_rate, 1.0)),
    ],
    FaultCause.FABRIC_LINK: [
        # The NVLink error rate is a distinct instrument from R_θ and is always
        # observed (defaults to 0) — it IS the discriminator for this channel.
        Predicate(Axis.CHANNEL, "NVLink CRC/recovery errors live", 1.5,
                  _hi(lambda f: f.nvlink_error_rate, FABRIC_ERR_PER_S), key=True),
        Predicate(Axis.POWER, "thermals within band (not a heat story)", 0.8,
                  _lo(lambda f: f.rtheta_overall_z, NORMAL_Z)),
    ],
    FaultCause.POWER_DELIVERY: [
        Predicate(Axis.CHANNEL, "power-throttle active a meaningful fraction", 1.2,
                  _hi(lambda f: f.power_violation_rate, POWER_VIOL_FRAC), key=True),
        Predicate(Axis.CHANNEL, "SM clock suppressed below boost", 1.0,
                  _lo(lambda f: f.clock_efficiency, CLOCK_EFF_LOW)),
        Predicate(Axis.POWER, "R_θ within band (heat is fine; power is the limit)", 1.0,
                  _lo(lambda f: f.rtheta_overall_z, NORMAL_Z),
                  key=True, needs="thermal in-band confirmation"),
    ],
}


# ── results ────────────────────────────────────────────────────────────────

@dataclass
class ModeScore:
    cause:        FaultCause
    score:        float            # 0..1 fraction of observed evidence that supports
    coverage:     float            # 0..1 fraction of this mode's axes that were observable
    supporting:   list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)


@dataclass
class MissingAxis:
    needs:    str     # the observation that would resolve the ambiguity
    via:      str     # sensor | probe | workload | service-log
    resolves: str     # which causes it would separate


@dataclass
class SignatureVerdict:
    top:            Optional[ModeScore]
    ranked:         list[ModeScore]
    identifiable:   bool                  # exact cause pinned on observed axes
    degenerate_with: list[FaultCause]     # causes tied with the top on observed axes
    missing_axes:   list[MissingAxis]     # what to observe to break the tie
    headline_cause: FaultCause            # top.cause, or NOMINAL/INSUFFICIENT_DATA
    discriminated:  bool = False          # a KEY axis was actually observed for the top mode.
                                          # When False, top.cause is only a *lean* — we know the
                                          # channel/subsystem, not the specific mode — because no
                                          # discriminating axis was exercised. Guards against
                                          # naming a precise cause off non-discriminating evidence
                                          # (e.g. "single-GPU" locality, shared by every local fault).

    def as_dict(self) -> dict:
        return {
            "headline_cause": self.headline_cause.value,
            "identifiable": self.identifiable,
            "discriminated": self.discriminated,
            "top": None if self.top is None else {
                "cause": self.top.cause.value,
                "score": round(self.top.score, 3),
                "coverage": round(self.top.coverage, 3),
                "supporting": self.top.supporting,
                "contradicting": self.top.contradicting,
            },
            "ranked": [
                {"cause": m.cause.value, "score": round(m.score, 3),
                 "coverage": round(m.coverage, 3)}
                for m in self.ranked
            ],
            "degenerate_with": [c.value for c in self.degenerate_with],
            "missing_axes": [
                {"needs": a.needs, "via": a.via, "resolves": a.resolves}
                for a in self.missing_axes
            ],
        }


def _score_mode(fv: FeatureVector, preds: list[Predicate]) -> ModeScore:
    matched_w = 0.0
    observed_w = 0.0
    total_w = sum(p.weight for p in preds) or 1.0
    supporting: list[str] = []
    contradicting: list[str] = []
    for p in preds:
        if p.weight == 0.0:
            continue  # marker-only predicate (e.g. the cold-plate gap) — no score weight
        t = p.test(fv)
        if t is Tristate.UNKNOWN:
            continue
        observed_w += p.weight
        if t is Tristate.SUPPORTS:
            matched_w += p.weight
            supporting.append(p.name)
        else:
            contradicting.append(p.name)
    score = (matched_w / observed_w) if observed_w > 0 else 0.0
    coverage = observed_w / total_w
    return ModeScore(FaultCause.NOMINAL, score, coverage, supporting, contradicting)


def classify(fv: FeatureVector) -> SignatureVerdict:
    """Score every fault mode's fingerprint, rank, and report identifiability."""
    scored: list[ModeScore] = []
    for cause, preds in SIGNATURES.items():
        ms = _score_mode(fv, preds)
        ms.cause = cause
        if ms.coverage >= MIN_COVERAGE:
            scored.append(ms)
    scored.sort(key=lambda m: m.score, reverse=True)

    # Nothing convincing → NOMINAL (quiet GPU) or INSUFFICIENT_DATA (nothing observable).
    if not scored:
        return SignatureVerdict(None, [], False, [], [], FaultCause.INSUFFICIENT_DATA)
    best = scored[0]
    if best.score < SCORE_FLOOR:
        return SignatureVerdict(best, scored, False, [], [], FaultCause.NOMINAL)

    # Did we actually exercise a discriminating axis for the leading mode? If not,
    # `best.cause` is only a lean — we've confirmed the channel/subsystem but the
    # specific mode rode in on non-discriminating evidence (thermal-rise, locality).
    discriminated = any(
        p.key and p.test(fv) in (Tristate.SUPPORTS, Tristate.CONTRADICTS)
        for p in SIGNATURES[best.cause]
    )

    # The tied cluster: modes within EPS of the best score.
    cluster = [m for m in scored if best.score - m.score <= DEGENERACY_EPS]
    degenerate_with = [m.cause for m in cluster if m.cause is not best.cause]

    # Missing axes: the discriminating (key) predicates that came back UNKNOWN on
    # any mode in the tied cluster — i.e. the axes that *would* separate them.
    missing: dict[str, MissingAxis] = {}
    for m in cluster:
        for p in SIGNATURES[m.cause]:
            if p.key and p.needs and p.test(fv) is Tristate.UNKNOWN:
                resolves = m.cause.value
                if p.needs in missing:
                    # merge the set of causes this axis would resolve
                    prev = missing[p.needs].resolves
                    if m.cause.value not in prev:
                        resolves = f"{prev}, {m.cause.value}"
                    else:
                        resolves = prev
                missing[p.needs] = MissingAxis(p.needs, p.via or "sensor", resolves)

    # Identifiability blocks only when a *competing cause* survives — i.e. a
    # `splits` discriminator came back UNKNOWN, so a real alternative mode can't
    # be ruled out. An UNKNOWN axis that only refines a sub-class (cold-plate vs
    # TIM-dryout: same subsystem, same remediation) is reported in missing_axes
    # but does NOT block the exact-cause call — the operator still knows what to
    # fix. This is the line between "I can't tell which fault" and "I know the
    # fault; a finer probe would name the exact failure mode within it."
    competing_unresolved = any(
        p.splits is not None and p.test(fv) is Tristate.UNKNOWN
        for p in SIGNATURES[best.cause]
    )
    identifiable = (
        len(degenerate_with) == 0
        and best.score >= STRONG_SCORE
        and not competing_unresolved
        and discriminated   # can't be "exact" if no discriminating axis was observed
    )

    return SignatureVerdict(
        top=best,
        ranked=scored,
        identifiable=identifiable,
        degenerate_with=degenerate_with,
        missing_axes=list(missing.values()),
        headline_cause=best.cause,
        discriminated=discriminated,
    )
