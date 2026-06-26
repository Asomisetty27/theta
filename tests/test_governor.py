"""
Tests for the AlertGovernor — Theta's first-run-trust + FP-budget layer.

Pins the behaviors that earn first-run trust: inferential alerts are HELD while a
GPU warms up, ground-truth hardware faults are NEVER gated, a critical inhibits
concurrent sub-critical alerts, and an alert storm trips the FP-budget breaker with
exactly one meta-alert recommending calibration.
"""
from theta.agent.governor import AlertGovernor, Action
from theta.agent.metrics import AlertEvent, GPUState


def _alert(gpu, detector, severity, ts=0.0, state=GPUState.DRIFTING):
    return AlertEvent(
        gpu_index=gpu, timestamp=ts, state=state, prev_state=GPUState.UNKNOWN,
        rtheta=0.9, rtheta_baseline=0.6, drift_sigma=3.0, confidence=0.9,
        message="x", context={"detector": detector, "severity": severity},
    )


def _ready(gov, gpu, ts):
    gov.note_cycle(gpu, ready=True, ts=ts)


def test_inferential_alert_held_while_warming():
    gov = AlertGovernor(warmup_sec=120)
    gov.note_cycle(0, ready=False, ts=0.0)          # first seen, not ready
    d = gov.evaluate(_alert(0, "drift", "warning", ts=10), ts=10)
    assert d.action is Action.HOLD_WARMING


def test_inferential_routes_once_ready():
    gov = AlertGovernor(warmup_sec=120)
    _ready(gov, 0, 5.0)                               # baseline established
    d = gov.evaluate(_alert(0, "drift", "warning", ts=10), ts=10)
    assert d.action is Action.ROUTE


def test_inferential_routes_after_warmup_even_if_not_ready():
    gov = AlertGovernor(warmup_sec=120)
    gov.note_cycle(0, ready=False, ts=0.0)
    gov.note_cycle(0, ready=False, ts=200.0)         # past warmup window
    d = gov.evaluate(_alert(0, "drift", "warning", ts=200), ts=200)
    assert d.action is Action.ROUTE


def test_ground_truth_fault_bypasses_warming():
    gov = AlertGovernor(warmup_sec=120)
    gov.note_cycle(0, ready=False, ts=0.0)           # warming
    # An ECC/Xid alert has no inferential detector tag → ground truth.
    ecc = _alert(0, "silicon_ecc", "critical", ts=10, state=GPUState.CRITICAL)
    d = gov.evaluate(ecc, ts=10)
    assert d.action is Action.ROUTE


def test_critical_inhibits_subcritical_same_gpu():
    gov = AlertGovernor(warmup_sec=0)                 # no warming
    _ready(gov, 0, 0.0)
    crit = _alert(0, "drift", "critical", ts=100, state=GPUState.CRITICAL)
    assert gov.evaluate(crit, ts=100).action is Action.ROUTE
    warn = _alert(0, "peer_relative", "warning", ts=120)
    assert gov.evaluate(warn, ts=120).action is Action.SUPPRESS_INHIBITED


def test_inhibition_expires_and_other_gpu_unaffected():
    gov = AlertGovernor(warmup_sec=0, inhibit_sec=300)
    _ready(gov, 0, 0.0); _ready(gov, 1, 0.0)
    gov.evaluate(_alert(0, "drift", "critical", ts=100, state=GPUState.CRITICAL), ts=100)
    # Different GPU is never inhibited.
    assert gov.evaluate(_alert(1, "drift", "warning", ts=110), ts=110).action is Action.ROUTE
    # Same GPU after the window: routes again.
    assert gov.evaluate(_alert(0, "drift", "warning", ts=500), ts=500).action is Action.ROUTE


def test_fp_budget_breaker_trips_with_one_meta_alert():
    gov = AlertGovernor(warmup_sec=0, budget_count=5, budget_window_sec=3600)
    _ready(gov, 0, 0.0)
    routed = trips = 0
    meta_alerts = []
    for i in range(20):
        d = gov.evaluate(_alert(0, "drift", "warning", ts=100 + i), ts=100 + i)
        if d.action is Action.ROUTE:
            routed += 1
        if d.action is Action.SUPPRESS_BUDGET:
            trips += 1
            if d.meta_alert is not None:
                meta_alerts.append(d.meta_alert)
    assert routed <= 6                       # ~budget then suppressed
    assert trips > 0                         # breaker engaged
    assert len(meta_alerts) == 1             # EXACTLY one meta-alert on trip
    assert "calibrate" in meta_alerts[0].message
    assert gov.readiness(0) == 0.0           # readiness reflects the tripped breaker


def test_breaker_rearms_after_cooldown():
    gov = AlertGovernor(warmup_sec=0, budget_count=3,
                        budget_window_sec=3600, breaker_cooldown=600)
    _ready(gov, 0, 0.0)
    for i in range(10):                      # trip it
        gov.evaluate(_alert(0, "drift", "warning", ts=100 + i), ts=100 + i)
    # After cooldown, a fresh cycle re-arms.
    gov.note_cycle(0, ready=True, ts=100 + 700)
    d = gov.evaluate(_alert(0, "drift", "warning", ts=100 + 701), ts=100 + 701)
    assert d.action is Action.ROUTE


def test_counts_accumulate_for_export():
    gov = AlertGovernor(warmup_sec=0)
    _ready(gov, 0, 0.0)
    gov.evaluate(_alert(0, "drift", "warning", ts=1), ts=1)
    assert gov.counts["ROUTE"] == 1


# ── Consensus gate ───────────────────────────────────────────────────────────
# consensus_min=2: a sub-critical degradation warning must be corroborated by a
# SECOND independent inferential detector before it routes. Critical bypasses.

def test_consensus_holds_single_detector_warning():
    gov = AlertGovernor(warmup_sec=0, consensus_min=2)
    _ready(gov, 0, 0.0)
    d = gov.evaluate(_alert(0, "drift", "warning", ts=10), ts=10)
    assert d.action is Action.HOLD_CONSENSUS     # only drift has voted


def test_consensus_routes_when_second_detector_corroborates():
    gov = AlertGovernor(warmup_sec=0, consensus_min=2)
    _ready(gov, 0, 0.0)
    assert gov.evaluate(_alert(0, "drift", "warning", ts=10), ts=10).action is Action.HOLD_CONSENSUS
    # A different inferential detector agrees within the window → routes.
    d = gov.evaluate(_alert(0, "peer_relative", "warning", ts=12), ts=12)
    assert d.action is Action.ROUTE


def test_consensus_distinct_detectors_required_not_repeats():
    gov = AlertGovernor(warmup_sec=0, consensus_min=2)
    _ready(gov, 0, 0.0)
    # Same detector firing twice is still one voter → still held.
    assert gov.evaluate(_alert(0, "drift", "warning", ts=10), ts=10).action is Action.HOLD_CONSENSUS
    assert gov.evaluate(_alert(0, "drift", "warning", ts=11), ts=11).action is Action.HOLD_CONSENSUS


def test_consensus_critical_bypasses_gate():
    gov = AlertGovernor(warmup_sec=0, consensus_min=2)
    _ready(gov, 0, 0.0)
    # A single critical inferential alert must route immediately — recall on
    # severe events is never traded for the consensus gate.
    d = gov.evaluate(_alert(0, "drift", "critical", ts=10, state=GPUState.CRITICAL), ts=10)
    assert d.action is Action.ROUTE


def test_consensus_votes_expire_outside_window():
    gov = AlertGovernor(warmup_sec=0, consensus_min=2, consensus_window_sec=300)
    _ready(gov, 0, 0.0)
    assert gov.evaluate(_alert(0, "drift", "warning", ts=10), ts=10).action is Action.HOLD_CONSENSUS
    # Second detector arrives long after the window → drift's vote expired, so
    # this is again a lone voter and stays held.
    d = gov.evaluate(_alert(0, "peer_relative", "warning", ts=10 + 400), ts=10 + 400)
    assert d.action is Action.HOLD_CONSENSUS


def test_consensus_min_one_is_passthrough():
    gov = AlertGovernor(warmup_sec=0, consensus_min=1)   # default — disabled
    _ready(gov, 0, 0.0)
    d = gov.evaluate(_alert(0, "drift", "warning", ts=10), ts=10)
    assert d.action is Action.ROUTE


def test_consensus_ground_truth_never_gated():
    gov = AlertGovernor(warmup_sec=0, consensus_min=2)
    _ready(gov, 0, 0.0)
    ecc = _alert(0, None, "critical", ts=10, state=GPUState.CRITICAL)
    ecc.context = {"severity": "critical"}               # no detector tag = ground truth
    assert gov.evaluate(ecc, ts=10).action is Action.ROUTE
