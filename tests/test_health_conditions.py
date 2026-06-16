"""
Tests for the health-as-conditions tracker (theta/agent/health.py).

Pins NPD semantics: conditions are level-state with transition timestamps, the
overall status derives correctly (critical > degraded > warming > healthy), and
`schedulable` reflects fitness for placement.
"""
from theta.agent.health import (
    HealthConditionTracker, HealthStatus, CRITICAL_CONDITIONS, WARNING_CONDITIONS,
)
from theta.agent.metrics import GPUState


def test_unknown_before_observation():
    tr = HealthConditionTracker()
    h = tr.health(0)
    assert h.status is HealthStatus.UNKNOWN
    assert h.schedulable is True          # don't block placement on no data
    assert h.conditions == []


def test_healthy_when_confident_and_clean():
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=False, state=GPUState.UNDER_LOAD)
    h = tr.health(0)
    assert h.status is HealthStatus.HEALTHY
    assert h.schedulable is True
    assert h.conditions == []


def test_warming_status_when_not_confident():
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=True, state=GPUState.CLEAN_IDLE)
    h = tr.health(0)
    assert h.status is HealthStatus.WARMING
    assert h.schedulable is True          # the GPU is fine; only the monitor is warming
    assert "warming" in h.message


def test_degraded_on_drift_warning():
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=False, state=GPUState.DRIFTING, drift_warning=True)
    h = tr.health(0)
    assert h.status is HealthStatus.DEGRADED
    assert h.schedulable is False
    assert any(c.name == "CoolingDegraded" for c in h.conditions)


def test_critical_on_ecc_double_bit():
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=False, state=GPUState.UNDER_LOAD, ecc_dbit=2)
    h = tr.health(0)
    assert h.status is HealthStatus.CRITICAL
    assert h.schedulable is False
    names = {c.name for c in h.conditions}
    assert "EccErrors" in names


def test_critical_outranks_warning():
    tr = HealthConditionTracker()
    # Both a warning (throttle) and a critical (critical drift) present.
    tr.observe(0, ts=100, warming=False, state=GPUState.CRITICAL,
               drift_critical=True, throttling=True)
    h = tr.health(0)
    assert h.status is HealthStatus.CRITICAL
    # critical condition sorts first
    assert h.conditions[0].name in CRITICAL_CONDITIONS


def test_condition_since_stamps_on_transition_and_clears():
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=False, state=GPUState.DRIFTING, drift_warning=True)
    h1 = tr.health(0)
    deg = next(c for c in h1.conditions if c.name == "CoolingDegraded")
    assert deg.since == 100
    # persists with the SAME since across cycles
    tr.observe(0, ts=130, warming=False, state=GPUState.DRIFTING, drift_warning=True)
    deg2 = next(c for c in tr.health(0).conditions if c.name == "CoolingDegraded")
    assert deg2.since == 100
    # clears when the signal goes away
    tr.observe(0, ts=160, warming=False, state=GPUState.UNDER_LOAD)
    assert tr.health(0).status is HealthStatus.HEALTHY
    assert all(c.name != "CoolingDegraded" for c in tr.health(0).conditions)


def test_status_since_updates_on_status_change():
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=False, state=GPUState.UNDER_LOAD)
    assert tr.health(0).since == 100
    tr.observe(0, ts=110, warming=False, state=GPUState.UNDER_LOAD)
    assert tr.health(0).since == 100        # unchanged while status stable
    tr.observe(0, ts=150, warming=False, state=GPUState.CRITICAL, drift_critical=True)
    assert tr.health(0).since == 150        # re-stamped on transition


def test_zombie_context_is_critical():
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=False, state=GPUState.ZOMBIE_RECOVERY)
    h = tr.health(0)
    assert h.status is HealthStatus.CRITICAL
    assert any(c.name == "ZombieContext" for c in h.conditions)


def test_peer_flag_degrades_even_without_drift():
    # A peer-relative anomaly (degraded-since-startup) with no temporal drift.
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=False, state=GPUState.UNDER_LOAD, peer_flagged=True)
    assert tr.health(0).status is HealthStatus.DEGRADED


def test_fleet_summary_counts():
    tr = HealthConditionTracker()
    tr.observe(0, ts=1, warming=False, state=GPUState.UNDER_LOAD)            # healthy
    tr.observe(1, ts=1, warming=False, state=GPUState.DRIFTING, drift_warning=True)  # degraded
    tr.observe(2, ts=1, warming=True,  state=GPUState.CLEAN_IDLE)            # warming
    s = tr.fleet_summary()["summary"]
    assert s["total"] == 3
    assert s["schedulable"] == 2            # healthy + warming
    assert s["by_status"]["degraded"] == 1


# ── API integration: the /conditions endpoint serves the tracker ──────────────

def test_conditions_endpoint_serves_tracker():
    import json
    import urllib.request
    import socket
    from theta.agent.health_api import HealthAPIServer

    tr = HealthConditionTracker()
    tr.observe(0, ts=1, warming=False, state=GPUState.UNDER_LOAD)                       # healthy
    tr.observe(1, ts=1, warming=False, state=GPUState.CRITICAL, drift_critical=True)    # critical

    # ephemeral free port
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()

    server = HealthAPIServer(
        port=port,
        get_status=lambda: {"gpus": {}},
        get_poll_latency=lambda: {},
        get_conditions=lambda: tr.fleet_summary(),
        auth_token=None,
        bind_host="127.0.0.1",
    )
    server.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/conditions", timeout=2) as r:
            body = json.loads(r.read())
        assert body["summary"]["total"] == 2
        assert body["summary"]["schedulable"] == 1            # only the healthy one
        assert body["gpus"]["1"]["status"] == "critical"
        assert body["gpus"]["1"]["schedulable"] is False

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/conditions/gpu/0", timeout=2) as r:
            g0 = json.loads(r.read())
        assert g0["status"] == "healthy"
    finally:
        server.stop()


def test_telemetry_unavailable_is_unknown_but_schedulable():
    tr = HealthConditionTracker()
    tr.observe(0, ts=100, warming=False, state=GPUState.UNKNOWN, telemetry_unavailable=True)
    h = tr.health(0)
    assert h.status is HealthStatus.UNKNOWN
    assert h.schedulable is True                 # don't drain a vGPU we just can't read
    assert any(c.name == "TelemetryUnavailable" for c in h.conditions)
    # and it does NOT spuriously add cooling conditions
    assert all(c.name == "TelemetryUnavailable" for c in h.conditions)
