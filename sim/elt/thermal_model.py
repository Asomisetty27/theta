"""
Transient 3-node Cauer thermal network, integrated with a stiff ODE solver.

State vector  y = [T_j, T_c, T_s]  (junction, case, heatsink) in degrees C.

    C_j dT_j/dt = P_eff(t,T_j)            - (T_j - T_c)/R_jc
    C_c dT_c/dt = (T_j - T_c)/R_jc        - (T_c - T_s)/R_ct(t)
    C_s dT_s/dt = (T_c - T_s)/R_ct(t)     - (T_s - T_amb)/R_sa(airflow(t,T_j))

Time-varying inputs:
  * R_ct(t)        : TIM resistance, raised by the TIM-degradation mode
  * airflow(t,T_j) : fan curve (auto-ramps with T_j) x airflow degradation factor
  * P_eff          : demanded workload power, reduced once thermal throttling engages

The solver is BDF (implicit, stiff-stable): the junction time constant (~0.3 s) and
heatsink time constant (~70 s) span >2 decades, so the system is stiff. A precise
throttle-crossing time is captured with a solve_ivp event (no grid-snapping error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from . import params as P


# ─────────────────────────────────────────────────────────────────────────────
# Scenario: everything that defines one run except the fixed physical params
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Scenario:
    """One simulated E-LT run."""
    duration_s: float                       # total wall-clock to integrate
    workload_power_w: float = P.LOAD_POWER_W # FIXED load power (the critical control)
    # degradation callables: t (s) -> multiplier. Default = no degradation.
    rct_mult_fn: Callable[[float], float] = lambda t: 1.0      # TIM dry-out (>=1)
    airflow_mult_fn: Callable[[float], float] = lambda t: 1.0  # airflow restriction (<=1)
    fan_cap_fn: Callable[[float], float] = lambda t: 1.0       # fan duty cap (<=1)
    baseline_s: float = 0.0                 # healthy window before degradation begins
    label: str = "scenario"


@dataclass
class SimResult:
    """Ground-truth + sensed telemetry on a uniform 1 Hz grid."""
    t: np.ndarray              # time (s)
    tj_true: np.ndarray        # true junction temp (C)
    tc_true: np.ndarray        # case temp (C)
    ts_true: np.ndarray        # heatsink temp (C)
    p_demand: np.ndarray       # demanded workload power (W)
    p_eff: np.ndarray          # effective power after throttle (W)
    rct: np.ndarray            # instantaneous TIM resistance (C/W)
    rsa: np.ndarray            # instantaneous convective resistance (C/W)
    rtheta_true: np.ndarray    # true (T_j - T_amb)/P_eff  (C/W)
    throttling: np.ndarray     # bool: thermal throttle active this sample
    t_throttle: Optional[float]  # exact first-throttle time (s) or None
    params: P.ThermalParams
    scenario: Scenario


# ─────────────────────────────────────────────────────────────────────────────
# Soft throttle: hold T_j near the limit by clock/power-limiting. Smooth (logistic)
# so the integrator stays stable; the *exact* crossing time comes from the event.
# ─────────────────────────────────────────────────────────────────────────────
def _throttle_factor(tj: float, prm: P.ThermalParams) -> float:
    """
    Fraction of demanded power delivered. Exactly 1.0 at or below the thermal
    limit (a real GPU runs at full clocks until it hits the limit); above the
    limit the clock/power governor reduces power toward the floor to hold the
    junction near the limit. One-sided so the first 93 C crossing — the
    ground-truth throttle event we measure lead time against — is unaffected.
    """
    over = tj - prm.throttle_c
    if over <= 0.0:
        return 1.0
    width = max(P.THROTTLE_HYSTERESIS_C, 0.5)
    # smooth above the limit: 0 at the limit, ->(1-floor) reduction well above
    s = 1.0 - np.exp(-over / width)
    return 1.0 - (1.0 - P.THROTTLE_POWER_FLOOR) * s


def _airflow(tj: float, t: float, scn: Scenario, prm: P.ThermalParams) -> float:
    """Effective normalised airflow: auto fan curve, capped and restricted."""
    duty = P.fan_duty(tj, prm.fan_duty_min, prm.fan_duty_max,
                      prm.fan_knee_lo, prm.fan_knee_hi)
    duty *= scn.fan_cap_fn(t)                 # fan/pump reduction mode
    airflow = duty * scn.airflow_mult_fn(t)   # airflow restriction mode
    return max(airflow, 1e-3)


def _rhs(t: float, y: np.ndarray, scn: Scenario, prm: P.ThermalParams) -> np.ndarray:
    """Cauer-network right-hand side dy/dt."""
    tj, tc, ts = y
    rct = prm.r_ct0 * scn.rct_mult_fn(t)
    rsa = P.r_sa(_airflow(tj, t, scn, prm), prm.r_sa_ref)

    p_eff = scn.workload_power_w * _throttle_factor(tj, prm)

    q_jc = (tj - tc) / prm.r_jc
    q_ct = (tc - ts) / rct
    q_sa = (ts - prm.t_amb_c) / rsa

    dtj = (p_eff - q_jc) / prm.c_j
    dtc = (q_jc - q_ct) / prm.c_c
    dts = (q_ct - q_sa) / prm.c_s
    return np.array([dtj, dtc, dts])


def steady_state(power_w: float, scn: Scenario, prm: P.ThermalParams,
                 at_t: float = 0.0) -> np.ndarray:
    """Self-consistent steady state at fixed degradation (used as initial condition)."""
    from scipy.optimize import brentq

    def tj_residual(tj: float) -> float:
        rct = prm.r_ct0 * scn.rct_mult_fn(at_t)
        rsa = P.r_sa(_airflow(tj, at_t, scn, prm), prm.r_sa_ref)
        return tj - (prm.t_amb_c + power_w * (prm.r_jc + rct + rsa))

    tj = brentq(tj_residual, prm.t_amb_c, 600.0, xtol=1e-6)
    rct = prm.r_ct0 * scn.rct_mult_fn(at_t)
    rsa = P.r_sa(_airflow(tj, at_t, scn, prm), prm.r_sa_ref)
    # case/heatsink steady temps from the same heat flow Q = power_w
    ts = prm.t_amb_c + power_w * rsa
    tc = ts + power_w * rct
    return np.array([tj, tc, ts])


def simulate(scn: Scenario, prm: P.ThermalParams = P.DEFAULT,
             dt_s: float = P.SAMPLE_PERIOD_S) -> SimResult:
    """
    Integrate the scenario and return uniformly-sampled ground-truth telemetry.
    Start from the healthy steady state at the workload power.
    """
    y0 = steady_state(scn.workload_power_w, scn, prm, at_t=0.0)
    t_eval = np.arange(0.0, scn.duration_s + dt_s, dt_s)

    # Event: T_j crosses the throttle temperature upward -> exact t_throttle.
    def cross(t, y, *_):
        return y[0] - prm.throttle_c
    cross.direction = 1.0
    cross.terminal = False

    sol = solve_ivp(
        _rhs, (0.0, scn.duration_s), y0,
        method="BDF", t_eval=t_eval, events=cross,
        args=(scn, prm), rtol=1e-7, atol=1e-9, max_step=dt_s,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")

    tj, tc, ts = sol.y[0], sol.y[1], sol.y[2]

    # Re-derive the time-varying quantities on the grid (vectorised where cheap)
    rct = np.array([prm.r_ct0 * scn.rct_mult_fn(t) for t in sol.t])
    rsa = np.array([P.r_sa(_airflow(tj_i, t, scn, prm), prm.r_sa_ref)
                    for tj_i, t in zip(tj, sol.t)])
    thr_factor = np.array([_throttle_factor(tj_i, prm) for tj_i in tj])
    p_demand = np.full_like(sol.t, scn.workload_power_w)
    p_eff = p_demand * thr_factor
    rtheta_true = (tj - prm.t_amb_c) / np.maximum(p_eff, 1e-6)
    throttling = tj >= prm.throttle_c

    t_throttle = float(sol.t_events[0][0]) if sol.t_events[0].size else None

    return SimResult(
        t=sol.t, tj_true=tj, tc_true=tc, ts_true=ts,
        p_demand=p_demand, p_eff=p_eff, rct=rct, rsa=rsa,
        rtheta_true=rtheta_true, throttling=throttling,
        t_throttle=t_throttle, params=prm, scenario=scn,
    )
