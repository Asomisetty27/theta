"""
Validation: prove the calibrated model reproduces Stage 1 ground truth and that
the numerics are sound. Run before trusting any lead-time number.

Checks:
  1. Steady-state operating points match Stage 1 (idle, load) within tolerance.
  2. Throttle point: power required to throttle a healthy GPU is physical.
  3. Energy balance at steady state: heat in == heat out across every node.
  4. Thermal time constants are in the expected range (fast junction, slow sink).
  5. Detector recovers the known baseline R_theta on a healthy (no-degradation) run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import params as P
from .thermal_model import simulate, steady_state, Scenario
from .detector import apply_sensor_model, windowed_rtheta, fit_baseline, DetectorConfig


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _steady_check() -> list[Check]:
    checks = []
    prm = P.DEFAULT
    healthy = Scenario(duration_s=1.0)   # no degradation

    tj_idle = steady_state(P.IDLE_POWER_W, healthy, prm)[0]
    tj_load = steady_state(P.LOAD_POWER_W, healthy, prm)[0]

    # Load is calibrated exactly; idle has a documented residual.
    checks.append(Check(
        "steady load T_j == Stage 1 (81 C)",
        abs(tj_load - P.LOAD_TEMP_C) < 0.5,
        f"sim={tj_load:.2f}C measured={P.LOAD_TEMP_C}C resid={tj_load-P.LOAD_TEMP_C:+.2f}C",
    ))
    checks.append(Check(
        "steady idle T_j ~ Stage 1 (39 C, +/-3C slack)",
        abs(tj_idle - P.IDLE_TEMP_C) < 3.0,
        f"sim={tj_idle:.2f}C measured={P.IDLE_TEMP_C}C resid={tj_idle-P.IDLE_TEMP_C:+.2f}C "
        f"(within sensor 1C quantisation + assumed-ambient slack)",
    ))
    return checks


def _energy_balance_check() -> list[Check]:
    """At steady state, heat into each node equals heat out (within tolerance)."""
    prm = P.DEFAULT
    healthy = Scenario(duration_s=1.0)
    y = steady_state(P.LOAD_POWER_W, healthy, prm)
    tj, tc, ts = y
    from .thermal_model import _airflow
    rct = prm.r_ct0
    rsa = P.r_sa(_airflow(tj, 0.0, healthy, prm), prm.r_sa_ref)

    q_in = P.LOAD_POWER_W
    q_jc = (tj - tc) / prm.r_jc
    q_ct = (tc - ts) / rct
    q_sa = (ts - prm.t_amb_c) / rsa
    max_err = max(abs(q_in - q_jc), abs(q_jc - q_ct), abs(q_ct - q_sa))
    return [Check(
        "steady-state energy balance (Q_in == Q_out each node)",
        max_err < 1e-3,
        f"max node imbalance = {max_err:.2e} W (Q_in={q_in:.2f} "
        f"Q_jc={q_jc:.2f} Q_ct={q_ct:.2f} Q_sa={q_sa:.2f})",
    )]


def _time_constant_check() -> list[Check]:
    """Junction time constant fast (<2s), heatsink slow (20-150s)."""
    prm = P.DEFAULT
    tau_jc = prm.r_jc * prm.c_j
    tau_sa = prm.r_sa_ref / (P.fan_duty(81.0) ** prm.conv_exp) * prm.c_s
    return [
        Check("junction time constant < 2 s", tau_jc < 2.0,
              f"tau_jc = Rjc*Cj = {tau_jc:.2f} s"),
        Check("heatsink time constant in 20-150 s", 20.0 < tau_sa < 150.0,
              f"tau_sa = Rsa*Cs = {tau_sa:.1f} s"),
    ]


def _throttle_physics_check() -> list[Check]:
    """The power that throttles a HEALTHY GPU must exceed the load power cap."""
    prm = P.DEFAULT
    healthy = Scenario(duration_s=1.0)
    # bisect power that drives healthy steady T_j to throttle
    from scipy.optimize import brentq
    f = lambda pw: steady_state(pw, healthy, prm)[0] - prm.throttle_c
    p_throttle = brentq(f, 1.0, 300.0)
    return [Check(
        "healthy GPU throttles only above load power (78 W cap)",
        p_throttle > P.LOAD_POWER_W,
        f"healthy throttle power = {p_throttle:.1f} W (> load {P.LOAD_POWER_W} W: "
        f"a healthy GPU at load does NOT throttle, as observed)",
    )]


def _detector_baseline_check() -> list[Check]:
    """On a healthy run the detector must recover the known steady R_theta."""
    healthy = Scenario(duration_s=900.0)
    sim = simulate(healthy)
    rng = np.random.default_rng(7)
    tel = apply_sensor_model(sim, rng, ambient_mode="true")
    rtheta, stable = windowed_rtheta(tel, DetectorConfig())
    base = fit_baseline(rtheta, stable, tel.t, 900.0)
    expected = P.R_THETA_LOAD
    return [Check(
        "detector baseline R_theta == known load R_theta",
        abs(base.mean - expected) < 0.02,
        f"detector μ={base.mean:.4f} expected={expected:.4f} "
        f"σ={base.std:.5f} n={base.n}",
    )]


def run_all() -> tuple[bool, list[Check]]:
    checks: list[Check] = []
    checks += _steady_check()
    checks += _energy_balance_check()
    checks += _time_constant_check()
    checks += _throttle_physics_check()
    checks += _detector_baseline_check()
    ok = all(c.passed for c in checks)
    return ok, checks


def format_report(checks: list[Check]) -> str:
    lines = ["E-LT model validation", "=" * 60]
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"[{mark}] {c.name}")
        lines.append(f"       {c.detail}")
    n_pass = sum(c.passed for c in checks)
    lines.append("-" * 60)
    lines.append(f"{n_pass}/{len(checks)} checks passed")
    return "\n".join(lines)


if __name__ == "__main__":
    ok, checks = run_all()
    print(format_report(checks))
    raise SystemExit(0 if ok else 1)
