"""
Tests for DCGMEnricher field mapping — focused on the throttle-cause fields
(241 thermal-violation, 240 power-violation, 112 clock-event bitmask) added so
Theta can attribute clock loss to thermal vs power without computing R_theta.

We don't need a live nv-hostengine: we drive get_latest()'s value-unpacking
indirectly by constructing the enricher, forcing it "available", and feeding a
fake DCGM sample-values structure. The goal is to pin that the three new fields
are requested and land on the RawSample.
"""
from theta.agent import dcgm_collector as dc
from theta.agent.dcgm_collector import (
    DCGMEnricher,
    _FI_DEV_THERMAL_VIOLATION,
    _FI_DEV_POWER_VIOLATION,
    _FI_DEV_CLOCK_EVENT_REASONS,
    _FIELD_IDS,
)
from theta.agent.metrics import RawSample


def _raw():
    return RawSample(
        gpu_index=0, timestamp=0.0, temp_junction=60.0, power_w=200.0,
        util_pct=90.0, mem_util_pct=50.0, perf_state=0,
        clock_sm_mhz=1400, clock_mem_mhz=900,
    )


def test_throttle_fields_are_watched():
    # The three throttle-cause field IDs must be in the watched set, else DCGM
    # never streams them.
    assert _FI_DEV_THERMAL_VIOLATION == 241
    assert _FI_DEV_POWER_VIOLATION == 240
    assert _FI_DEV_CLOCK_EVENT_REASONS == 112
    for fid in (241, 240, 112):
        assert fid in _FIELD_IDS


class _FakeEntry:
    def __init__(self, value):
        self.value = value


class _FakeSamples:
    """Mimics group.samples.GetLatest(fg).values — {gpu_index: {field_id: entry}}."""
    def __init__(self, row):
        self.values = {0: row}


class _FakeGroupSamples:
    def __init__(self, row):
        self._row = row

    def GetLatest(self, _fg):
        return _FakeSamples(self._row)


class _FakeGroup:
    def __init__(self, row):
        self.samples = _FakeGroupSamples(row)


def _enricher_with(row):
    e = DCGMEnricher.__new__(DCGMEnricher)   # bypass __init__/_try_init (no daemon)
    e._available = True
    e._group = _FakeGroup(row)
    e._field_group = object()
    return e


def test_throttle_fields_land_on_sample():
    row = {
        _FI_DEV_THERMAL_VIOLATION:   _FakeEntry(123456),   # us
        _FI_DEV_POWER_VIOLATION:     _FakeEntry(7890),     # us
        _FI_DEV_CLOCK_EVENT_REASONS: _FakeEntry(0x40),     # hw_thermal_slowdown bit
    }
    e = _enricher_with(row)
    s = _raw()
    e.enrich(0, s)
    assert s.thermal_violation_us == 123456
    assert s.power_violation_us == 7890
    assert s.dcgm_clock_event_reasons == 0x40


def test_blank_dcgm_values_map_to_zero():
    # DCGM returns a large sentinel (e.g. INT32_BLANK 0x7fffffff) for unavailable
    # fields; _val() must coerce those to 0, not leak the sentinel.
    row = {
        _FI_DEV_THERMAL_VIOLATION:   _FakeEntry(0x7fffffff),
        _FI_DEV_POWER_VIOLATION:     _FakeEntry(0x7fffffff),
        _FI_DEV_CLOCK_EVENT_REASONS: _FakeEntry(0x7fffffff),
    }
    e = _enricher_with(row)
    s = _raw()
    e.enrich(0, s)
    assert s.thermal_violation_us == 0
    assert s.power_violation_us == 0
    assert s.dcgm_clock_event_reasons == 0


def test_enrich_noop_when_unavailable():
    e = DCGMEnricher.__new__(DCGMEnricher)
    e._available = False
    s = _raw()
    e.enrich(0, s)   # must not raise, fields stay default
    assert s.thermal_violation_us == 0
    assert s.power_violation_us == 0
