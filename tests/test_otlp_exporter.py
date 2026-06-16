"""
Tests for the OTLP metrics exporter (theta/agent/otlp_exporter.py).

Verifies it is inert when disabled / SDK-absent, accepts hot-path updates, and —
when the OTel SDK is present — actually emits the per-GPU gauges through a real
in-memory metric reader (no network).
"""
import pytest

from theta.agent.otlp_exporter import OTLPExporter, OTLP_AVAILABLE


def test_inert_when_no_endpoint():
    ex = OTLPExporter(endpoint=None)
    ex.update_gpu(0, rtheta=0.7, temp=65, power=300)   # must not raise
    ex.start()
    ex.shutdown()
    assert ex._enabled is False


def test_update_ignored_when_disabled():
    ex = OTLPExporter(endpoint=None)
    ex.update_gpu(0, rtheta=0.7)
    assert ex._snap.rtheta == {}                        # nothing recorded


@pytest.mark.skipif(not OTLP_AVAILABLE, reason="opentelemetry SDK not installed")
def test_emits_gauges_through_inmemory_reader(monkeypatch):
    # Swap the real MeterProvider for one backed by an InMemoryMetricReader, so the
    # observable-gauge callbacks are exercised end-to-end without a collector.
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    import theta.agent.otlp_exporter as mod

    reader = InMemoryMetricReader()

    ex = OTLPExporter(endpoint="http://localhost:4318/v1/metrics")
    assert ex._enabled is True
    ex.update_gpu(0, rtheta=0.72, temp=66, power=305, readiness=1.0, schedulable=True)
    ex.update_gpu(1, rtheta=2.10, temp=82, power=240, readiness=0.0, schedulable=False)

    # Build the provider with the in-memory reader instead of the OTLP one.
    from opentelemetry.metrics import Observation
    resource = mod.Resource.create({"service.name": "theta"})
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    meter = provider.get_meter("theta.agent")
    s = ex._snap
    def _obs(table):
        return lambda options: [Observation(v, {"gpu_index": str(g)}) for g, v in table.items()]
    meter.create_observable_gauge("theta.gpu.rtheta", callbacks=[_obs(s.rtheta)])
    meter.create_observable_gauge("theta.gpu.schedulable", callbacks=[_obs(s.schedulable)])

    data = reader.get_metrics_data()
    # Flatten all metric names + data points
    points = {}
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                for dp in m.data.data_points:
                    points.setdefault(m.name, {})[dp.attributes["gpu_index"]] = dp.value

    assert points["theta.gpu.rtheta"]["0"] == pytest.approx(0.72)
    assert points["theta.gpu.rtheta"]["1"] == pytest.approx(2.10)
    assert points["theta.gpu.schedulable"]["0"] == 1.0
    assert points["theta.gpu.schedulable"]["1"] == 0.0
    provider.shutdown()
