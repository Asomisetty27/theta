"""
AMD ROCm telemetry collector (amdsmi backend).

Implements the same TelemetryCollector protocol as NVMLCollector, producing
identical RawSample objects, so the daemon and every downstream module work
unchanged on AMD Instinct (MI200/MI300) hardware. Built against `amdsmi` — AMD's
current official Python library (successor to rocm_smi_lib / pyrsmi).

A nice property for Theta specifically: AMD exposes JUNCTION (hotspot) temperature
directly via amdsmi_get_temp_metric(..., TEMPERATURE_TYPE_HOTSPOT/JUNCTION), which
is exactly the T_junction that R_θ wants — no edge-to-junction estimation needed.

Every field read is defensive (wrapped), because amdsmi surface varies across ROCm
releases and SKUs — same philosophy as the NVML collector. Missing fields degrade
to safe defaults rather than failing the whole sample.

STATUS: implemented against the amdsmi API but NOT yet verified on real AMD silicon
(no MI300 in hand at build time — Cal Poly EE access is on the roadmap). The pure
mapping/parse logic is unit-tested with a fake amdsmi; hardware validation is the
open item before this is advertised as production-ready for AMD.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from .metrics import RawSample

log = structlog.get_logger(__name__)


def _amdsmi():
    """Import amdsmi lazily so NVIDIA-only hosts never need it installed."""
    import amdsmi
    return amdsmi


class ROCmCollector:
    """AMD Instinct collector via amdsmi. Same protocol/shape as NVMLCollector."""

    vendor = "amd"

    def __init__(self, config):
        self._config = config
        self._handles: list = []
        self._names: list[str] = []
        self._initialized = False

    # ── lifecycle ──────────────────────────────────────────────────────────────
    async def __aenter__(self) -> "ROCmCollector":
        amdsmi = _amdsmi()
        amdsmi.amdsmi_init()
        self._initialized = True
        handles = amdsmi.amdsmi_get_processor_handles()
        idxs = self._config.gpu_indices or list(range(len(handles)))
        self._handles = [handles[i] for i in idxs if i < len(handles)]
        self._names = [self._device_name(amdsmi, h) for h in self._handles]
        log.info("rocm_initialized", n_gpus=len(self._handles), names=self._names)
        return self

    async def __aexit__(self, *_) -> None:
        if self._initialized:
            try:
                _amdsmi().amdsmi_shut_down()
            except Exception:
                pass
            self._initialized = False

    @staticmethod
    def _device_name(amdsmi, handle) -> str:
        try:
            info = amdsmi.amdsmi_get_gpu_asic_info(handle)
            name = info.get("market_name") if isinstance(info, dict) else None
            return name or "AMD Instinct"
        except Exception:
            return "AMD Instinct"

    # ── collection ─────────────────────────────────────────────────────────────
    async def collect_all(self) -> list[RawSample]:
        return await asyncio.to_thread(self._collect_all_sync)

    def _collect_all_sync(self) -> list[RawSample]:
        amdsmi = _amdsmi()
        return [self._collect_one(amdsmi, i, h) for i, h in enumerate(self._handles)]

    def _collect_one(self, amdsmi, idx: int, handle) -> RawSample:
        t0 = time.time()

        # Temperature — prefer HOTSPOT/JUNCTION (what R_θ wants); fall back to EDGE.
        temp = 0.0
        for sensor_name in ("HOTSPOT", "JUNCTION", "EDGE"):
            try:
                sensor = getattr(amdsmi.AmdSmiTemperatureType, sensor_name)
                metric = amdsmi.AmdSmiTemperatureMetric.CURRENT
                temp = float(amdsmi.amdsmi_get_temp_metric(handle, sensor, metric))
                if temp > 0:
                    break
            except Exception:
                continue

        # Power (W) — average socket power.
        power_w = 0.0
        try:
            p = amdsmi.amdsmi_get_power_info(handle)
            if isinstance(p, dict):
                power_w = float(p.get("average_socket_power")
                                or p.get("current_socket_power") or 0.0)
        except Exception:
            pass

        # Activity (utilization %).
        util = mem_util = 0.0
        try:
            a = amdsmi.amdsmi_get_gpu_activity(handle)
            if isinstance(a, dict):
                util     = float(a.get("gfx_activity", 0.0))
                mem_util = float(a.get("umc_activity", 0.0))
        except Exception:
            pass

        # Clocks (MHz) — GFX (compute) and MEM.
        clock_sm = clock_mem = sm_max = 0
        try:
            g = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
            if isinstance(g, dict):
                clock_sm = int(g.get("cur_clk") or g.get("clk") or 0)
                sm_max   = int(g.get("max_clk") or 0)
        except Exception:
            pass
        try:
            m = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
            if isinstance(m, dict):
                clock_mem = int(m.get("cur_clk") or m.get("clk") or 0)
        except Exception:
            pass

        # ECC (uncorrectable = the GPU-death signal).
        ecc_sbit = ecc_dbit = 0
        try:
            e = amdsmi.amdsmi_get_gpu_total_ecc_count(handle)
            if isinstance(e, dict):
                ecc_sbit = int(e.get("correctable_count", 0))
                ecc_dbit = int(e.get("uncorrectable_count", 0))
        except Exception:
            pass

        # Perf-state: AMD has no NVIDIA-style P0–P15. Approximate from activity so
        # the classifier's P-state heuristics still have a signal (high load → P0,
        # idle → P8). Documented approximation; the CUDA-zombie P0 case is NVIDIA-
        # specific and simply won't trigger on AMD.
        perf_state = 0 if util >= 30 else 8

        return RawSample(
            gpu_index=idx, timestamp=t0,
            temp_junction=temp, power_w=power_w,
            util_pct=util, mem_util_pct=mem_util, perf_state=perf_state,
            clock_sm_mhz=clock_sm, clock_mem_mhz=clock_mem,
            fan_speed_pct=None,
            ecc_sbit=ecc_sbit, ecc_dbit=ecc_dbit,
            throttle_reasons=0, sm_clock_max_mhz=sm_max,
            poll_latency_s=time.time() - t0,
        )

    # ── HAL protocol ───────────────────────────────────────────────────────────
    @property
    def gpu_count(self) -> int:
        return len(self._handles)

    @property
    def gpu_names(self) -> list[str]:
        return list(self._names)

    @property
    def capabilities(self) -> list:
        # MIG/vGPU detection is NVIDIA-specific; AMD partitioning (CPX/SPX) is a
        # separate model. Report all devices as physical for now.
        from .device_caps import DeviceCapability, DeviceMode
        return [DeviceCapability(DeviceMode.PHYSICAL, True, True, True,
                                 "AMD Instinct — physical GPU")
                for _ in self._handles]
