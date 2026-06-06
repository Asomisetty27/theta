"""
Pytest suite for the E-LT simulation.

Run:  sim/.venv/bin/python -m pytest sim/tests/test_elt.py -v
(requires pytest in the sim venv: sim/.venv/bin/pip install pytest)
"""

import numpy as np
import pytest

from sim.elt import params as P
from sim.elt.thermal_model import simulate, steady_state, Scenario
from sim.elt import degradation as deg
from sim.elt.detector import (
    apply_sensor_model, windowed_rtheta, fit_baseline, detect_anomaly,
    DetectorConfig,
)
from sim.elt.experiment import run_trial
from sim.elt import validate as V


# ── Calibration / validation ─────────────────────────────────────────────────
def test_validation_all_pass():
    ok, checks = V.run_all()
    assert ok, "\n" + V.format_report(checks)


def test_load_point_exact():
    tj = steady_state(P.LOAD_POWER_W, Scenario(duration_s=1.0), P.DEFAULT)[0]
    assert abs(tj - P.LOAD_TEMP_C) < 0.5


def test_idle_point_within_slack():
    tj = steady_state(P.IDLE_POWER_W, Scenario(duration_s=1.0), P.DEFAULT)[0]
    assert abs(tj - P.IDLE_TEMP_C) < 3.0


# ── Thermal model invariants ──────────────────────────────────────────────────
def test_healthy_run_does_not_throttle():
    sim = simulate(Scenario(duration_s=600.0))
    assert sim.t_throttle is None
    assert sim.tj_true.max() < P.THROTTLE_TEMP_C


def test_monotonic_rtheta_under_tim():
    scn, _ = deg.tim_degradation(3600, baseline_s=300, severity=2.4, variant="gradual")
    sim = simulate(scn)
    # true R_theta should be (weakly) increasing once degradation begins
    after = sim.rtheta_true[sim.t > 600]
    diffs = np.diff(after)
    assert np.mean(diffs > 0) > 0.9   # overwhelmingly increasing


def test_all_modes_reach_throttle():
    for scn in (
        deg.tim_degradation(6 * 3600, 600, 2.4, "gradual")[0],
        deg.airflow_restriction(2700, 600, 0.45, "gradual")[0],
        deg.fan_reduction(900, 180, 0.40, "step")[0],
    ):
        sim = simulate(scn)
        assert sim.t_throttle is not None, scn.label


def test_power_held_flat_pre_throttle():
    """The experimental control: power is constant until throttle engages."""
    scn, _ = deg.airflow_restriction(2700, 600, 0.45, "gradual")
    sim = simulate(scn)
    pre = sim.p_eff[sim.t < sim.t_throttle]
    assert np.std(pre) < 1e-6   # exactly flat before throttle


# ── Sensor model ──────────────────────────────────────────────────────────────
def test_temp_is_integer_quantised():
    sim = simulate(Scenario(duration_s=120.0))
    tel = apply_sensor_model(sim, np.random.default_rng(0))
    assert np.allclose(tel.temp_j, np.round(tel.temp_j))


def test_virtual_ambient_is_constant():
    sim = simulate(Scenario(duration_s=120.0))
    tel = apply_sensor_model(sim, np.random.default_rng(0), ambient_mode="virtual")
    assert np.std(tel.t_ref) == 0.0


# ── Detector ──────────────────────────────────────────────────────────────────
def test_detector_recovers_baseline():
    sim = simulate(Scenario(duration_s=900.0))
    tel = apply_sensor_model(sim, np.random.default_rng(3))
    rtheta, stable = windowed_rtheta(tel, DetectorConfig())
    base = fit_baseline(rtheta, stable, tel.t, 900.0)
    assert abs(base.mean - P.R_THETA_LOAD) < 0.02


def test_no_false_positive_on_healthy_run():
    sim = simulate(Scenario(duration_s=1200.0))
    tel = apply_sensor_model(sim, np.random.default_rng(5))
    rtheta, stable = windowed_rtheta(tel, DetectorConfig())
    # use first half as baseline, check second half raises no anomaly at k=3
    base = fit_baseline(rtheta, stable, tel.t, 600.0)
    res = detect_anomaly(rtheta, stable, tel.t, base, 3.0, DetectorConfig())
    assert res.t_anomaly is None


# ── End-to-end ────────────────────────────────────────────────────────────────
def test_leadtime_positive_and_ordered():
    """Slow modes give more lead time than fast modes."""
    tim = run_trial("tim", "gradual", 6 * 3600, 600, seed=1)
    fan = run_trial("fan", "step", 900, 180, seed=1)
    assert tim.lead_times[3.0] > 0
    assert fan.lead_times[3.0] > 0
    assert tim.lead_times[3.0] > fan.lead_times[3.0]


def test_anomaly_precedes_throttle():
    tr = run_trial("airflow", "gradual", 2700, 600, seed=2)
    assert tr.t_anomaly[3.0] < tr.t_throttle


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
