"""
Tests for the TDP-scaled power-plausibility guard (B200 readiness gap #2).

Per-die NVML reporting on dual-die GPUs (B200) can report ~half the module's
power while T_junction reflects the hotter die. That halves the R_θ denominator
and doubles R_θ — a spurious degradation signal. The old guard only caught
near-zero power (< 40% of idle floor = 34 W on a B200), so a 450 W per-die
report on a 1000 W part sailed through. power_reading_suspect() now also scales
to TDP.

Critical invariants pinned here:
  - catches the dual-die half-power case on a B200,
  - does NOT fire on a healthy fully-loaded GPU,
  - does NOT mask real degradation (degradation raises R_θ without lowering
    power, so a degrading-but-correctly-reporting GPU is never dropped).
"""

from theta.agent.collector import (
    power_reading_suspect,
    UNDERREPORT_LOAD_UTIL,
    UNDERREPORT_TDP_FRAC,
)

# (T4, H100, B200) TDP / idle-floor pairs
T4 = dict(idle_floor_w=10.0, tdp_w=70.0)
B200 = dict(idle_floor_w=85.0, tdp_w=1000.0)


class TestNearZeroTier:
    def test_one_die_reads_near_zero_while_active(self):
        # B200 die reads ~5 W while util 90% — clear under-report
        assert power_reading_suspect(5.0, 90.0, **B200) == "near_zero"

    def test_genuine_idle_not_flagged(self):
        # Idle GPU at low util drawing idle power — legitimate, not suspect
        assert power_reading_suspect(85.0, 2.0, **B200) is None


class TestTdpFloorTier:
    def test_b200_dual_die_half_power_is_caught(self):
        # The headline sim case: 450 W reported (half of ~900 W) at 90% util.
        # Old idle-floor gate (34 W) missed this; TDP floor (500 W) catches it.
        assert power_reading_suspect(450.0, 90.0, **B200) == "below_tdp_floor_at_load"

    def test_healthy_full_load_not_flagged(self):
        # Healthy B200 at 900 W / 90% util — must NOT be dropped.
        assert power_reading_suspect(900.0, 90.0, **B200) is None

    def test_degradation_at_full_power_not_masked(self):
        # A degrading GPU still draws full power (degradation raises R_θ via
        # temperature, not by lowering watts). 880 W at 95% util must pass so
        # the R_θ rise is actually observed downstream.
        assert power_reading_suspect(880.0, 95.0, **B200) is None

    def test_below_floor_but_light_load_not_flagged(self):
        # 300 W at 40% util — below the TDP floor but NOT heavy load, so we
        # don't second-guess it (could be a light/bursty workload).
        assert power_reading_suspect(300.0, 40.0, **B200) is None

    def test_t4_scales_too(self):
        # T4 (70 W TDP): floor = 35 W. 20 W at 90% util is suspect; 60 W is fine.
        assert power_reading_suspect(20.0, 90.0, **T4) == "below_tdp_floor_at_load"
        assert power_reading_suspect(60.0, 90.0, **T4) is None

    def test_unknown_tdp_skips_tier2(self):
        # No profile TDP → tier-2 disabled, only near-zero applies.
        assert power_reading_suspect(450.0, 90.0, idle_floor_w=85.0, tdp_w=0.0) is None


class TestThresholdBoundaries:
    def test_exactly_at_util_gate(self):
        # At exactly the util gate, the TDP floor applies.
        below = B200["tdp_w"] * UNDERREPORT_TDP_FRAC - 1.0
        assert power_reading_suspect(below, UNDERREPORT_LOAD_UTIL, **B200) == "below_tdp_floor_at_load"

    def test_just_under_util_gate_not_judged(self):
        below = B200["tdp_w"] * UNDERREPORT_TDP_FRAC - 1.0
        assert power_reading_suspect(below, UNDERREPORT_LOAD_UTIL - 0.1, **B200) is None
