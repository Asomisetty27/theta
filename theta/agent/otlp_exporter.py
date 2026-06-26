"""
OpenTelemetry (OTLP) metrics export — meet fleets where their observability is.

Prometheus pull is the default, but many fleets standardize on OpenTelemetry and
run an OTel Collector that fans out to Prometheus/Datadog/Grafana Cloud/etc. This
exporter pushes Theta's core signals over OTLP/HTTP so Theta drops into that
pipeline with no scrape config.

Optional by design: the OTel SDK is an extra (`pip install runtheta[otlp]`). If it
isn't installed this module is inert — same pattern as the Prometheus exporter's
availability gate — so the base agent has zero new hard dependencies.

Implementation uses OTel **observable gauges**: the agent updates a plain in-memory
snapshot each cycle (cheap, no SDK on the hot path), and the SDK's periodic reader
calls our callbacks to read it at export time. R_θ, temperature, power, drift σ,
readiness, and schedulable are exported per GPU (gpu_index attribute).
"""

from __future__ import annotations

from typing import Optional

try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.metrics import Observation
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    OTLP_AVAILABLE = True
except Exception:  # SDK not installed → inert
    OTLP_AVAILABLE = False

import structlog

from .. import __version__

log = structlog.get_logger(__name__)


# Per-GPU snapshot the agent writes each cycle; callbacks read it at export time.
# Plain dicts keyed by gpu_index → value. No SDK objects on the hot path.
class _Snapshot:
    __slots__ = ("rtheta", "temp", "power", "drift_sigma", "readiness", "schedulable")

    def __init__(self):
        self.rtheta:      dict[int, float] = {}
        self.temp:        dict[int, float] = {}
        self.power:       dict[int, float] = {}
        self.drift_sigma: dict[int, float] = {}
        self.readiness:   dict[int, float] = {}
        self.schedulable: dict[int, float] = {}


class OTLPExporter:
    """OTLP/HTTP metrics export for Theta. Inert if the OTel SDK isn't installed."""

    def __init__(self, endpoint: Optional[str], interval_sec: float = 30.0,
                 headers: Optional[dict] = None):
        self._enabled = bool(endpoint) and OTLP_AVAILABLE
        self._endpoint = endpoint
        self._snap = _Snapshot()
        self._provider = None
        if endpoint and not OTLP_AVAILABLE:
            log.warning("otlp_unavailable",
                        note="OTLP endpoint set but opentelemetry SDK not installed; "
                             "run `pip install runtheta[otlp]`. Export disabled.")

    # ---- hot-path snapshot updates (called by the daemon each cycle) ----------
    def update_gpu(self, gpu: int, *, rtheta=None, temp=None, power=None,
                   drift_sigma=None, readiness=None, schedulable=None) -> None:
        if not self._enabled:
            return
        s = self._snap
        if rtheta is not None:
            s.rtheta[gpu] = float(rtheta)
        if temp is not None:
            s.temp[gpu] = float(temp)
        if power is not None:
            s.power[gpu] = float(power)
        if drift_sigma is not None:
            s.drift_sigma[gpu] = float(drift_sigma)
        if readiness is not None:
            s.readiness[gpu] = float(readiness)
        if schedulable is not None:
            s.schedulable[gpu] = 1.0 if schedulable else 0.0

    # ---- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if not self._enabled or self._provider is not None:
            return
        exporter = OTLPMetricExporter(endpoint=self._endpoint)
        reader = PeriodicExportingMetricReader(exporter)  # default 60s; collector can override
        resource = Resource.create({
            "service.name": "theta",
            "service.version": __version__,
        })
        self._provider = MeterProvider(metric_readers=[reader], resource=resource)
        meter = self._provider.get_meter("theta.agent")

        def _obs(table, unit_desc):
            def cb(options):  # CallbackOptions
                return [Observation(v, {"gpu_index": str(g)}) for g, v in table.items()]
            return cb

        s = self._snap
        meter.create_observable_gauge("theta.gpu.rtheta", callbacks=[_obs(s.rtheta, 1)],
                                      unit="Cel/W", description="Effective thermal resistance R_θ")
        meter.create_observable_gauge("theta.gpu.temperature", callbacks=[_obs(s.temp, 1)],
                                      unit="Cel", description="Junction temperature")
        meter.create_observable_gauge("theta.gpu.power", callbacks=[_obs(s.power, 1)],
                                      unit="W", description="GPU power draw")
        meter.create_observable_gauge("theta.gpu.drift_sigma", callbacks=[_obs(s.drift_sigma, 1)],
                                      unit="1", description="R_θ deviation from baseline (σ)")
        meter.create_observable_gauge("theta.gpu.readiness", callbacks=[_obs(s.readiness, 1)],
                                      unit="1", description="Detector confidence (1=confident)")
        meter.create_observable_gauge("theta.gpu.schedulable", callbacks=[_obs(s.schedulable, 1)],
                                      unit="1", description="1 if fit to schedule new work")
        log.info("otlp_export_started", endpoint=self._endpoint)

    def shutdown(self) -> None:
        if self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception:
                pass
            self._provider = None
