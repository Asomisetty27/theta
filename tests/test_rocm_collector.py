"""
Tests for the AMD ROCm collector (theta/agent/rocm_collector.py).

No AMD hardware required: a fake `amdsmi` module is injected so the field-mapping
logic — junction-temp preference, mW/W handling, activity→util, ECC, perf-state
approximation, and defensive degradation on missing fields — is exercised end to
end into RawSample. (Hardware validation on real MI300 silicon remains open.)
"""
import sys
import types

import pytest

from theta.agent.rocm_collector import ROCmCollector


def _fake_amdsmi(*, hotspot=80.0, power=560.0, gfx=95.0, umc=40.0,
                 gfx_clk=2100, gfx_max=2400, mem_clk=1600,
                 ecc_corr=3, ecc_uncorr=1, drop=()):
    """Build a stand-in amdsmi module. `drop` lists calls that should raise."""
    m = types.ModuleType("amdsmi")

    class _TempType:  # enum-ish
        HOTSPOT = "hotspot"; JUNCTION = "junction"; EDGE = "edge"
    class _TempMetric:
        CURRENT = "current"
    class _ClkType:
        GFX = "gfx"; MEM = "mem"
    m.AmdSmiTemperatureType = _TempType
    m.AmdSmiTemperatureMetric = _TempMetric
    m.AmdSmiClkType = _ClkType

    def guard(name, fn):
        def wrapped(*a, **k):
            if name in drop:
                raise RuntimeError(f"{name} unsupported")
            return fn(*a, **k)
        return wrapped

    m.amdsmi_init = guard("init", lambda: None)
    m.amdsmi_shut_down = guard("shutdown", lambda: None)
    m.amdsmi_get_processor_handles = guard("handles", lambda: ["h0", "h1"])
    m.amdsmi_get_gpu_asic_info = guard("asic", lambda h: {"market_name": "AMD Instinct MI300X"})
    m.amdsmi_get_temp_metric = guard("temp",
        lambda h, sensor, metric: hotspot if sensor == "hotspot" else 0.0)
    m.amdsmi_get_power_info = guard("power", lambda h: {"average_socket_power": power})
    m.amdsmi_get_gpu_activity = guard("activity", lambda h: {"gfx_activity": gfx, "umc_activity": umc})
    def _clk(h, t):
        return ({"cur_clk": gfx_clk, "max_clk": gfx_max} if t == "gfx"
                else {"cur_clk": mem_clk})
    m.amdsmi_get_clock_info = guard("clk", _clk)
    m.amdsmi_get_gpu_total_ecc_count = guard("ecc",
        lambda h: {"correctable_count": ecc_corr, "uncorrectable_count": ecc_uncorr})
    return m


@pytest.fixture
def inject_amdsmi(monkeypatch):
    def _inject(mod):
        monkeypatch.setitem(sys.modules, "amdsmi", mod)
    return _inject


def _collect(indices=None):
    import asyncio
    from theta.agent.collector import CollectorConfig
    coll = ROCmCollector(CollectorConfig(interval_sec=1.0, gpu_indices=indices))
    async def run():
        async with coll as c:
            return await c.collect_all(), c.gpu_names, c.capabilities
    return asyncio.run(run())


def test_maps_all_fields_into_rawsample(inject_amdsmi):
    inject_amdsmi(_fake_amdsmi())
    samples, names, caps = _collect(None)
    assert len(samples) == 2
    s = samples[0]
    assert s.temp_junction == 80.0           # HOTSPOT preferred
    assert s.power_w == 560.0
    assert s.util_pct == 95.0 and s.mem_util_pct == 40.0
    assert s.clock_sm_mhz == 2100 and s.sm_clock_max_mhz == 2400
    assert s.clock_mem_mhz == 1600
    assert s.ecc_sbit == 3 and s.ecc_dbit == 1
    assert names[0] == "AMD Instinct MI300X"
    assert caps[0].mode.value == "physical"


def test_perf_state_approximation(inject_amdsmi):
    inject_amdsmi(_fake_amdsmi(gfx=95.0))
    busy, *_ = _collect(None)
    assert busy[0].perf_state == 0           # high activity → P0-like
    inject_amdsmi(_fake_amdsmi(gfx=2.0))
    idle, *_ = _collect(None)
    assert idle[0].perf_state == 8           # idle → P8-like


def test_gpu_index_filter(inject_amdsmi):
    inject_amdsmi(_fake_amdsmi())
    samples, *_ = _collect(indices=[1])
    assert len(samples) == 1                  # only the requested device


def test_defensive_on_missing_fields(inject_amdsmi):
    # ECC + clocks unsupported on this SKU → safe defaults, sample still produced.
    inject_amdsmi(_fake_amdsmi(drop=("ecc", "clk")))
    samples, *_ = _collect(None)
    s = samples[0]
    assert s.ecc_sbit == 0 and s.ecc_dbit == 0
    assert s.clock_sm_mhz == 0
    assert s.temp_junction == 80.0            # unaffected fields still read


def test_falls_back_to_edge_temp(inject_amdsmi):
    # Hotspot returns 0 (unsupported) — collector should try EDGE next.
    m = _fake_amdsmi()
    m.amdsmi_get_temp_metric = lambda h, sensor, metric: (70.0 if sensor == "edge" else 0.0)
    inject_amdsmi(m)
    samples, *_ = _collect(None)
    assert samples[0].temp_junction == 70.0


def test_hal_prefer_amd_returns_rocm_collector(inject_amdsmi):
    inject_amdsmi(_fake_amdsmi())
    from theta.agent.hal import select_collector
    from theta.agent.collector import CollectorConfig
    coll = select_collector(CollectorConfig(interval_sec=1.0), prefer="amd")
    assert coll.vendor == "amd"
    assert isinstance(coll, ROCmCollector)
