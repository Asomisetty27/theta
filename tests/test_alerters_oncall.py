"""
Tests for the PagerDuty + Opsgenie alerters (theta/agent/alerter.py).

Verifies the API-correct payload shape, severity→priority mapping, the stable
dedup key (so on-call tools group rather than spawn duplicate incidents), and
that a failed POST retries without raising.
"""
import asyncio

import pytest

from theta.agent.alerter import PagerDutyAlerter, OpsgenieAlerter, _dedup_key
from theta.agent.metrics import AlertEvent, GPUState


def _alert(gpu=3, sev="critical", state=GPUState.CRITICAL, detector="drift"):
    return AlertEvent(
        gpu_index=gpu, timestamp=1000.0, state=state, prev_state=GPUState.UNDER_LOAD,
        rtheta=2.1, rtheta_baseline=0.7, drift_sigma=4.2, confidence=0.95,
        message="GPU running hot — cooling degraded", context={"severity": sev, "detector": detector},
    )


class _FakeResp:
    def raise_for_status(self): pass


class _FakeClient:
    """Records POSTs; optionally fails the first N to exercise retry."""
    def __init__(self, fail_first=0):
        self.calls = []
        self._fail = fail_first
    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._fail > 0:
            self._fail -= 1
            raise RuntimeError("network blip")
        return _FakeResp()
    async def aclose(self): pass


def test_dedup_key_stable_per_gpu_and_kind():
    a = _alert(gpu=3, detector="drift")
    b = _alert(gpu=3, detector="drift")
    c = _alert(gpu=3, detector="peer_relative")
    assert _dedup_key(a) == _dedup_key(b)      # same gpu + kind → same key
    assert _dedup_key(a) != _dedup_key(c)      # different kind → different key
    assert "gpu3" in _dedup_key(a)


def test_pagerduty_payload_shape_and_severity():
    pd = PagerDutyAlerter(routing_key="R0UT1NGK3Y")
    p = pd._build_payload(_alert(sev="critical"))
    assert p["routing_key"] == "R0UT1NGK3Y"
    assert p["event_action"] == "trigger"
    assert p["dedup_key"] == "theta-gpu3-drift"
    assert p["payload"]["severity"] == "critical"
    assert p["payload"]["source"] == "theta/gpu3"
    assert p["payload"]["custom_details"]["drift_sigma"] == 4.2
    # warning maps through
    assert pd._build_payload(_alert(sev="warning"))["payload"]["severity"] == "warning"


def test_opsgenie_payload_shape_and_priority():
    og = OpsgenieAlerter(api_key="KEY123")
    p = og._build_payload(_alert(sev="critical"))
    assert p["alias"] == "theta-gpu3-drift"
    assert p["priority"] == "P1"                       # critical → P1
    assert "theta" in p["tags"]
    assert og._build_payload(_alert(sev="warning"))["priority"] == "P3"
    assert og._build_payload(_alert(sev="info"))["priority"] == "P5"


def test_opsgenie_eu_region_endpoint():
    assert "api.eu.opsgenie.com" in OpsgenieAlerter(api_key="k", region="eu").ENDPOINT
    assert "api.opsgenie.com" in OpsgenieAlerter(api_key="k").ENDPOINT


def test_pagerduty_send_posts_to_events_api():
    pd = PagerDutyAlerter(routing_key="rk")
    pd._client = _FakeClient()
    asyncio.run(pd.send(_alert()))
    assert pd._client.calls[0]["url"] == "https://events.pagerduty.com/v2/enqueue"
    assert pd._client.calls[0]["json"]["routing_key"] == "rk"


def test_opsgenie_send_sets_genie_auth_header():
    og = OpsgenieAlerter(api_key="KEY123")
    og._client = _FakeClient()
    asyncio.run(og.send(_alert()))
    assert og._client.calls[0]["headers"]["Authorization"] == "GenieKey KEY123"


def test_send_retries_then_succeeds():
    pd = PagerDutyAlerter(routing_key="rk", max_retries=3)
    pd._client = _FakeClient(fail_first=2)     # fail twice, succeed third
    # patch the module's asyncio.sleep to a real instant no-op (no recursion)
    import theta.agent.alerter as mod
    orig = mod.asyncio.sleep
    async def _no_sleep(*_a, **_k): return None
    mod.asyncio.sleep = _no_sleep
    try:
        asyncio.run(pd.send(_alert()))
    finally:
        mod.asyncio.sleep = orig
    assert len(pd._client.calls) == 3          # retried and eventually posted
