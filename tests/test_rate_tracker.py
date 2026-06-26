"""
Tests for the fabric/power rate tracker — the piece that activates FABRIC_LINK /
POWER_DELIVERY in the live daemon by converting cumulative counters into rates.

The properties that matter: first sample is silent (no interval), old settled
history doesn't read as a live fault, a counter reset can't produce a negative
spike, and the power-violation fraction is a real fraction of the interval.
"""

from theta.agent.rate_tracker import RateTracker, FabricPowerRates, clock_efficiency


def test_first_sample_yields_zeros():
    rt = RateTracker()
    r = rt.update(0, 100.0, nvlink_errors=42, power_violation_us=5000)
    assert r == FabricPowerRates(0.0, 0.0, 0.0)


def test_nvlink_error_rate_is_per_second():
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=10, power_violation_us=0)
    r = rt.update(0, 110.0, nvlink_errors=30, power_violation_us=0)  # +20 over 10s
    assert r.nvlink_error_rate == 2.0


def test_settled_history_reads_as_zero_rate():
    # A GPU that logged errors long ago but is no longer accruing them is healthy.
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=1000, power_violation_us=0)
    r = rt.update(0, 105.0, nvlink_errors=1000, power_violation_us=0)  # no new errors
    assert r.nvlink_error_rate == 0.0


def test_counter_reset_clamps_to_zero():
    # GPU reboots → counter drops to 0. Must not produce a negative rate.
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=500, power_violation_us=9_000_000)
    r = rt.update(0, 110.0, nvlink_errors=3, power_violation_us=10)
    assert r.nvlink_error_rate == 0.0
    assert r.power_violation_rate == 0.0


def test_power_violation_is_fraction_of_interval():
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=0, power_violation_us=0)
    # 3s of throttle accumulated over a 10s interval → 0.30 of the interval.
    r = rt.update(0, 110.0, nvlink_errors=0, power_violation_us=3_000_000)
    assert abs(r.power_violation_rate - 0.30) < 1e-9


def test_power_violation_rate_clamped_to_one():
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=0, power_violation_us=0)
    # More accumulated µs than wall-clock (counter quirk) must not exceed 1.0.
    r = rt.update(0, 101.0, nvlink_errors=0, power_violation_us=5_000_000)
    assert r.power_violation_rate == 1.0


def test_nonpositive_interval_yields_zeros():
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=0, power_violation_us=0)
    assert rt.update(0, 100.0, nvlink_errors=99, power_violation_us=99) == FabricPowerRates()


def test_pcie_replays_optional():
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=0, power_violation_us=0, pcie_replays=0)
    r = rt.update(0, 110.0, nvlink_errors=0, power_violation_us=0, pcie_replays=50)
    assert r.pcie_replay_rate == 5.0


def test_per_gpu_isolation():
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=0, power_violation_us=0)
    rt.update(1, 100.0, nvlink_errors=0, power_violation_us=0)
    r0 = rt.update(0, 110.0, nvlink_errors=100, power_violation_us=0)
    r1 = rt.update(1, 110.0, nvlink_errors=0, power_violation_us=0)
    assert r0.nvlink_error_rate == 10.0
    assert r1.nvlink_error_rate == 0.0


def test_reset_forgets_history():
    rt = RateTracker()
    rt.update(0, 100.0, nvlink_errors=10, power_violation_us=0)
    rt.reset(0)
    # After reset, the next sample is "first" again → zeros, no spurious delta.
    assert rt.update(0, 110.0, nvlink_errors=10, power_violation_us=0) == FabricPowerRates()


def test_clock_efficiency_helper():
    assert clock_efficiency(1980, 1980) == 1.0
    assert clock_efficiency(990, 1980) == 0.5
    assert clock_efficiency(500, 0) == 1.0   # unknown ceiling → treat as at-boost (no false suppression)


def test_end_to_end_activates_fabric_in_causal():
    # The whole point: tracker output, fed to reason(), surfaces FABRIC_LINK.
    from theta.agent.causal import reason
    from theta.agent.fault_classifier import FaultCause
    from theta.agent.metrics import GPUState

    rt = RateTracker()
    rt.update(3, 100.0, nvlink_errors=0, power_violation_us=0)
    rates = rt.update(3, 110.0, nvlink_errors=50, power_violation_us=0)  # 5 err/s

    exp = reason(
        gpu_index=3, smoothed_state=GPUState.UNDER_LOAD, state_confidence=0.9,
        alternative_states=[], fault_cause=FaultCause.NOMINAL, fault_confidence=0.5,
        rtheta_current=0.72, rtheta_baseline=0.72, rtheta_k_sigma=0.2,
        nvlink_error_rate=rates.nvlink_error_rate,
    )
    assert exp.hypothesis.cause is FaultCause.FABRIC_LINK
