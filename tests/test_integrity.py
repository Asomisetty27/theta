"""
Tests for the telemetry-integrity gate (Stage 1).

The gate's whole job is restraint: diagnose the pipeline before diagnosing the
GPU, and refuse to narrate a number that is itself garbage.
"""

from theta.agent.integrity import (
    IntegritySignals, assess_integrity, blocked_explanation,
    BLOCK_COLLECTOR, BLOCK_OFF_BUS, BLOCK_UNRESPONSIVE, BLOCK_STALE, BLOCK_SENSOR,
)


def test_clean_telemetry_is_trustworthy():
    v = assess_integrity(IntegritySignals(power_w=300.0, poll_latency_s=0.05))
    assert v.trustworthy and not v.blocked
    assert v.blocked_cause is None
    assert v.score > 0.9


def test_collector_failure_blocks():
    v = assess_integrity(IntegritySignals(collector_ok=False))
    assert v.blocked and v.blocked_cause == BLOCK_COLLECTOR


def test_uuid_change_blocks_off_bus():
    v = assess_integrity(IntegritySignals(uuid_stable=False))
    assert v.blocked and v.blocked_cause == BLOCK_OFF_BUS


def test_high_poll_latency_blocks_unresponsive():
    v = assess_integrity(IntegritySignals(poll_latency_s=5.0))
    assert v.blocked and v.blocked_cause == BLOCK_UNRESPONSIVE


def test_implausible_power_blocks_sensor():
    assert assess_integrity(IntegritySignals(power_w=-5.0)).blocked_cause == BLOCK_SENSOR
    assert assess_integrity(IntegritySignals(power_w=9999.0)).blocked_cause == BLOCK_SENSOR


def test_stale_blocks():
    v = assess_integrity(IntegritySignals(stale=True, power_w=300.0))
    assert v.blocked and v.blocked_cause == BLOCK_STALE


def test_idle_gpu_without_rtheta_is_not_blocked():
    # Clean idle legitimately can't compute R_θ — that alone must not block.
    v = assess_integrity(IntegritySignals(rtheta_computable=False, power_w=15.0))
    assert v.trustworthy


def test_hard_failures_take_priority_over_soft():
    # collector failure dominates even if other fields also look bad.
    v = assess_integrity(IntegritySignals(collector_ok=False, stale=True, poll_latency_s=10.0))
    assert v.blocked_cause == BLOCK_COLLECTOR


def test_blocked_explanation_shape_matches_causal_dict():
    v = assess_integrity(IntegritySignals(collector_ok=False))
    d = blocked_explanation(7, v)
    # Same keys the site/alert layer reads from CausalExplanation.as_dict()...
    for key in ("headline", "urgency", "tier", "hypothesis", "alternatives", "evidence", "actions"):
        assert key in d
    # ...plus the explicit block markers so nothing mistakes it for healthy.
    assert d["telemetry_blocked"] is True
    assert d["block_cause"] == BLOCK_COLLECTOR
    assert d["tier"] == "unconfirmed"
    assert d["hypothesis"]["confidence"] == 0.0
