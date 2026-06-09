"""
Cal Poly AI Factory simulation — DGX B200, 8 GPUs per node.

Exercises the full Theta pipeline with physically grounded B200 values
to find bugs before the real install. Runs without NVML (no GPU required).

Physics basis:
  B200 SXM6 with liquid cold-plate cooling, 20°C coolant supply:
    junction-to-coolant R_thermal ≈ 0.060 C/W (cold-plate, high-efficiency)
    idle power (NV-hostengine, NVLink fabric active): ~88W
    T_j_idle = 20 + 0.060 * 88 = 25.3°C
    T_j_load = 20 + 0.060 * 900 = 74°C
    T_j_degraded = 20 + 0.074 * 900 = 86.6°C  (+23% thermal resistance)
    T_j_zombie   = 20 + 0.060 * 108 = 26.5°C   (CUDA context stuck)

  Virtual ambient T_ref locked from idle window ≈ 25°C.

  Resulting R_theta (T_ref = 25°C):
    idle:      (25.3 - 25) / 88   = 0.0034 C/W  → below MIN_DELTA_T → INVALID
    load:      (74   - 25) / 900  = 0.0544 C/W
    degraded:  (86.6 - 25) / 900  = 0.0689 C/W  (+26.7% above healthy load)
    zombie:    (26.5 - 25) / 108  = 0.0139 C/W  (low R_theta + P0 pstate)
"""

from __future__ import annotations

import sys
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from theta.agent.metrics import (
    RawSample, GPUState, compute_rtheta, enrich,
)
from theta.agent.baseline import BaselineManager
from theta.agent.window import SteadyStateWindow, SIGMA_STRICT
from theta.agent.classifier import StateClassifier
from theta.agent.calibrate import (
    CalibrationManager, CalibrationResult, derive_thresholds,
)
from theta.agent.detector import DriftDetector
from theta.agent.correlator import FleetCorrelator
from theta.agent.hw_profiles import resolve_or_default


# ── B200 ground-truth physics constants ──────────────────────────────────────

COOLANT_INLET_C   = 20.0    # °C — DGX B200 facility spec, liquid cold-plate
R_JUNCTION_CW     = 0.060   # C/W — junction-to-coolant effective resistance
IDLE_POWER_W      = 88.0    # W  — GPU at idle with NVLink fabric active
LOAD_POWER_W      = 900.0   # W  — GPU at ~90% utilization
DEGRADED_R_MULT   = 1.23    # thermal resistance +23% (TIM partial degradation)
ZOMBIE_POWER_W    = 108.0   # W  — CUDA context stuck (elevated P0 draw)

def _b200_tj(power_w: float, r_mult: float = 1.0) -> float:
    """Compute B200 junction temperature from power + optional R_thermal multiplier."""
    return COOLANT_INLET_C + R_JUNCTION_CW * r_mult * power_w

T_J_IDLE      = _b200_tj(IDLE_POWER_W)           # 25.3°C
T_J_LOAD      = _b200_tj(LOAD_POWER_W)           # 74.0°C
T_J_DEGRADED  = _b200_tj(LOAD_POWER_W, DEGRADED_R_MULT)  # 86.6°C
T_J_ZOMBIE    = _b200_tj(ZOMBIE_POWER_W)          # 26.5°C

# ── Result tracking ───────────────────────────────────────────────────────────

@dataclass
class Finding:
    severity: str   # CRITICAL / WARNING / INFO
    title:    str
    detail:   str

findings: list[Finding] = []

def CRIT(title: str, detail: str):
    findings.append(Finding("CRITICAL", title, detail))
    print(f"\n  ❌ CRITICAL: {title}")
    print(f"     {detail}")

def WARN(title: str, detail: str):
    findings.append(Finding("WARNING", title, detail))
    print(f"\n  ⚠️  WARNING:  {title}")
    print(f"     {detail}")

def OK(title: str, detail: str = ""):
    print(f"  ✓  {title}" + (f" — {detail}" if detail else ""))

def HDR(title: str):
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_sample(
    gpu: int,
    t_j: float,
    power_w: float,
    util_pct: float,
    pstate: int,
    ts: float,
    noise: float = 0.0,
) -> RawSample:
    return RawSample(
        gpu_index        = gpu,
        timestamp        = ts,
        temp_junction    = t_j + noise,
        power_w          = power_w + abs(noise) * 0.5,
        util_pct         = util_pct,
        mem_util_pct     = util_pct * 0.7,
        perf_state       = pstate,
        clock_sm_mhz     = 2800 if pstate == 0 else 300,
        clock_mem_mhz    = 8000 if pstate == 0 else 800,
        fan_speed_pct    = None,           # B200 liquid cooled — no fan
        ecc_sbit         = 0,
        ecc_dbit         = 0,
        throttle_reasons = 0,
        sm_clock_max_mhz = 2800,
        poll_latency_s   = 0.002,
    )

def push_stable_window(
    window: SteadyStateWindow,
    bm: BaselineManager,
    gpu: int,
    t_j: float,
    power_w: float,
    util_pct: float,
    pstate: int,
    t_ref: float,
    n_samples: int = 12,
    dt: float = 1.5,
) -> Optional[object]:
    """Push N samples spaced dt apart and return the final window result."""
    ts = time.time()
    result = None
    for i in range(n_samples):
        ts += dt
        noise = 0.05 * math.sin(i * 1.3)
        s = make_sample(gpu, t_j, power_w, util_pct, pstate, ts, noise)
        r, valid = compute_rtheta(t_j + noise, t_ref, power_w)
        if valid and r is not None:
            result = window.update(gpu, ts, r, power_w, util_pct, pstate)
    return result


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — Hardware profile check
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 1 — B200 hardware profile inspection")

profile = resolve_or_default("NVIDIA B200 SXM6")
print(f"\n  GPU name input:           'NVIDIA B200 SXM6'")
print(f"  Profile resolved:         {profile.canonical_name}")
print(f"  Profile confidence:       {profile.confidence}")
print(f"  rtheta_load_threshold:    {profile.rtheta_load_threshold} C/W  (extrapolated)")
print(f"  rtheta_idle_threshold:    {profile.rtheta_idle_threshold} C/W  (extrapolated)")
print(f"  rtheta_expected_load:     {profile.rtheta_expected_under_load} C/W  (extrapolated)")
print(f"  rtheta_expected_idle:     {profile.rtheta_expected_idle} C/W  (extrapolated)")
print(f"  expected_ambient_c:       {profile.expected_ambient_c} °C")
print(f"  idle_floor_w:             {profile.idle_floor_w} W")
print(f"  TDP:                      {profile.tdp_w} W")
print(f"  cooling:                  {profile.cooling}")
print()
print(f"  Simulated B200 ground-truth (physics):")
print(f"    T_j idle  = {T_J_IDLE:.1f} °C   (coolant {COOLANT_INLET_C}°C + R·{IDLE_POWER_W}W)")
print(f"    T_j load  = {T_J_LOAD:.1f} °C   (coolant {COOLANT_INLET_C}°C + R·{LOAD_POWER_W}W)")
print(f"    T_j degraded = {T_J_DEGRADED:.1f} °C  (+{(DEGRADED_R_MULT-1)*100:.0f}% thermal resistance)")
print(f"    T_j zombie   = {T_J_ZOMBIE:.1f} °C")

# Check for physically implausible profile values
r_load_implied_max = (profile.junction_max_c - COOLANT_INLET_C) / LOAD_POWER_W
if profile.rtheta_expected_under_load > r_load_implied_max:
    CRIT(
        "B200 profile R_theta_load is physically impossible",
        f"profile rtheta_expected_under_load={profile.rtheta_expected_under_load} C/W "
        f"implies T_j = {COOLANT_INLET_C + profile.rtheta_expected_under_load * LOAD_POWER_W:.0f}°C "
        f"at {LOAD_POWER_W}W — exceeds T_j_max={profile.junction_max_c}°C. "
        f"Extrapolation broke. --ambient calibration will produce wrong thresholds."
    )
else:
    OK("Profile R_theta_load is physically plausible")


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — Virtual ambient locking on B200
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 2 — Virtual ambient locking on B200")

bm = BaselineManager(_file=Path("/tmp/sim_b200_baselines.json"))
ts = time.time()

print(f"\n  Feeding 30 idle samples (T_j={T_J_IDLE:.1f}°C, P={IDLE_POWER_W}W)…")
for i in range(30):
    ts += 1.0
    noise = 0.1 * math.sin(i * 0.8)
    bm.update(0, T_J_IDLE + noise, util=0.0, pstate=8, ts=ts)

t_ref = bm.get_t_ref(0, "NVIDIA B200 SXM6")
b = bm.get_baseline(0)
print(f"  T_ref locked at:  {t_ref:.2f} °C  (source: {b.source if b else 'profile_prior'})")
print(f"  Expected ≈ {T_J_IDLE:.1f} °C")

# Compute R_theta at idle and load with this T_ref
r_idle, v_idle = compute_rtheta(T_J_IDLE, t_ref, IDLE_POWER_W)
r_load, v_load = compute_rtheta(T_J_LOAD, t_ref, LOAD_POWER_W)
r_degd, v_degd = compute_rtheta(T_J_DEGRADED, t_ref, LOAD_POWER_W)
r_zomb, v_zomb = compute_rtheta(T_J_ZOMBIE, t_ref, ZOMBIE_POWER_W)

def _fmt(v):
    return f"{v:.4f}" if v is not None else "  N/A "

print(f"\n  R_theta with T_ref={t_ref:.2f}°C:")
print(f"    idle:       {_fmt(r_idle)} C/W  valid={v_idle}   (T_j={T_J_IDLE:.1f}°C, P={IDLE_POWER_W}W)")
print(f"    load:       {_fmt(r_load)} C/W  valid={v_load}   (T_j={T_J_LOAD:.1f}°C, P={LOAD_POWER_W}W)")
print(f"    degraded:   {_fmt(r_degd)} C/W  valid={v_degd}   (T_j={T_J_DEGRADED:.1f}°C, P={LOAD_POWER_W}W)")
print(f"    zombie:     {_fmt(r_zomb)} C/W  valid={v_zomb}   (T_j={T_J_ZOMBIE:.1f}°C, P={ZOMBIE_POWER_W}W)")

if not v_idle:
    WARN(
        "B200 idle R_theta is invalid (ΔT below noise floor)",
        f"T_j_idle={T_J_IDLE:.1f}°C ≈ T_ref={t_ref:.1f}°C → ΔT={T_J_IDLE-t_ref:.2f}°C < MIN_DELTA_T=0.5°C. "
        f"Daemon will NEVER observe CLEAN_IDLE state on B200 with virtual ambient. "
        f"Calibration must use external ambient (BMC) as T_ref instead of idle junction."
    )
elif r_idle is not None and r_load is not None and r_idle < r_load:
    CRIT(
        "R_theta ordering INVERTED on B200 (idle < load)",
        f"R_theta_idle={r_idle:.4f} < R_theta_load={r_load:.4f}. "
        f"Classification rules assume idle > load (T4 pattern). "
        f"All B200 GPUs will be permanently classified as CLEAN_IDLE (R_theta < load_threshold)."
    )
else:
    OK(f"R_theta ordering correct (idle {r_idle:.4f} > load {r_load:.4f})")


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — Calibration paths
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 3 — Calibration: profile-based vs measured")

# Path A: --ambient bypass
# BEFORE fix: derive_thresholds(profile.rtheta_expected_idle, None) with T4 ratio scaling
# AFTER fix (cli.py): for liquid-cooled hardware, use profile.rtheta_load_threshold directly
is_liquid_cooled = getattr(profile, "t_ref_strategy", "idle_window") == "coolant_inlet"
if is_liquid_cooled:
    # cli.py fix: use profile thresholds directly
    thr_a_load = profile.rtheta_load_threshold
    thr_a_idle = profile.rtheta_idle_threshold
    source_label = "profile thresholds (liquid-cooled fix)"
else:
    thr_a_load, thr_a_idle = derive_thresholds(profile.rtheta_expected_idle, None)
    source_label = "derive_thresholds() idle-only"

print(f"\n  Path A — --ambient {COOLANT_INLET_C}°C ({source_label}):")
print(f"    profile.rtheta_load_threshold = {profile.rtheta_load_threshold:.4f} C/W")
print(f"    derived load_threshold        = {thr_a_load:.4f} C/W")
print(f"    derived idle_threshold        = {thr_a_idle:.4f} C/W")
# Use virtual T_ref (25.3°C) for load R_theta since that's what --ambient 20°C + idle window gives
r_load_ambient = r_load  # with T_ref=t_ref=25.3°C
print(f"    actual R_theta at load        = {_fmt(r_load_ambient)} C/W  (virtual T_ref={t_ref:.1f}°C)")
if r_load_ambient is not None:
    if r_load_ambient < thr_a_load:
        OK(f"  load R_theta ({r_load_ambient:.4f}) < load_threshold ({thr_a_load:.4f}) → UNDER_LOAD ✓")
    else:
        CRIT(
            "--ambient calibration produces wrong load threshold",
            f"Real load R_theta={r_load_ambient:.4f} C/W > derived load_threshold={thr_a_load:.4f} C/W. "
            f"GPU under full load would be classified as DRIFTING, not UNDER_LOAD."
        )

# Path B: measured calibration (what `theta calibrate` would produce from real data)
if v_idle and v_load and r_idle is not None and r_load is not None:
    thr_b_load, thr_b_idle = derive_thresholds(r_idle, r_load)
    print(f"\n  Path B — measured calibration (real idle + load windows):")
    print(f"    measured R_theta_idle   = {r_idle:.4f} C/W")
    print(f"    measured R_theta_load   = {r_load:.4f} C/W")
    print(f"    derived load_threshold  = {thr_b_load:.4f} C/W")
    print(f"    derived idle_threshold  = {thr_b_idle:.4f} C/W")
    if thr_b_load > thr_b_idle:
        CRIT(
            "derive_thresholds() produces inverted thresholds when idle R_theta < load R_theta",
            f"load_threshold={thr_b_load:.4f} > idle_threshold={thr_b_idle:.4f}. "
            f"derive_thresholds() assumes R_theta_idle > R_theta_load (T4 pattern). "
            f"On B200 with virtual ambient this is reversed. Classification rules will misfire."
        )
    else:
        OK(f"Thresholds sane: load={thr_b_load:.4f}, idle={thr_b_idle:.4f}")
else:
    print(f"\n  Path B skipped — idle or load R_theta invalid (see Scenario 2 findings)")


# Path C: BMC-as-T_ref (correct approach for liquid-cooled B200)
# With T_ref = coolant inlet (20°C), R_theta has a wider, sensible range
r_load_bmc, v_load_bmc = compute_rtheta(T_J_LOAD, COOLANT_INLET_C, LOAD_POWER_W)
r_idle_bmc, v_idle_bmc = compute_rtheta(T_J_IDLE, COOLANT_INLET_C, IDLE_POWER_W)
r_degd_bmc, v_degd_bmc = compute_rtheta(T_J_DEGRADED, COOLANT_INLET_C, LOAD_POWER_W)

if r_idle_bmc is not None and r_load_bmc is not None:
    thr_c_load, thr_c_idle = derive_thresholds(r_idle_bmc, r_load_bmc)
    print(f"\n  Path C — BMC T_ref={COOLANT_INLET_C}°C (coolant inlet as reference):")
    print(f"    R_theta_idle  = {r_idle_bmc:.4f} C/W")
    print(f"    R_theta_load  = {r_load_bmc:.4f} C/W")
    print(f"    R_theta_degd  = {r_degd_bmc:.4f} C/W  (+{(r_degd_bmc/r_load_bmc - 1)*100:.1f}%)")
    print(f"    load_threshold = {thr_c_load:.4f} C/W")
    print(f"    idle_threshold = {thr_c_idle:.4f} C/W")
    if r_idle_bmc > r_load_bmc:
        OK("BMC T_ref: R_theta ordering correct (idle > load)",
           f"idle={r_idle_bmc:.4f} > load={r_load_bmc:.4f}")
        # Check degraded detection
        if r_degd_bmc is not None and r_degd_bmc > thr_c_load:
            OK(f"BMC T_ref: degraded GPU detectable ({r_degd_bmc:.4f} > load_threshold {thr_c_load:.4f})")
        else:
            WARN("Degraded GPU not detectable via BMC T_ref path",
                 f"R_theta_degraded={r_degd_bmc:.4f} <= load_threshold={thr_c_load:.4f}")
    else:
        WARN("BMC T_ref: R_theta ordering still inverted",
             f"idle={r_idle_bmc:.4f} < load={r_load_bmc:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — Classifier behavior with each calibration path
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 4 — Classifier output with each calibration path")

from theta.agent.classifier import _rule_classify

scenarios = {
    "idle":     (r_idle,  IDLE_POWER_W,   0, 8),
    "load":     (r_load,  LOAD_POWER_W,   95, 0),
    "degraded": (r_degd,  LOAD_POWER_W,   95, 0),
    "zombie":   (r_zomb,  ZOMBIE_POWER_W, 0, 0),
}

_T4_LT = 0.87
_T4_IT = 1.50
_bmc_path = (thr_c_load, thr_c_idle) if (r_idle_bmc is not None and r_load_bmc is not None) else (_T4_LT, _T4_IT)

for cal_label, (lt, it) in [
    ("T4 defaults (uncalibrated)", (_T4_LT, _T4_IT)),
    ("--ambient path (profile estimate)", (thr_a_load, thr_a_idle)),
    ("BMC T_ref path", _bmc_path),
]:
    print(f"\n  {cal_label}  (load_thr={lt:.4f}, idle_thr={it:.4f}):")
    for name, (rtheta, power, util, pstate) in scenarios.items():
        if rtheta is None:
            print(f"    {name:12s} → SKIP (R_theta invalid)")
            continue
        state, conf = _rule_classify(rtheta, power, pstate, lt, it)
        expected = {
            "idle": "CLEAN_IDLE",
            "load": "UNDER_LOAD",
            "degraded": "UNDER_LOAD",
            "zombie": "ZOMBIE_RECOVERY",
        }[name]
        correct = state.name == expected
        mark = "✓" if correct else "✗"
        print(f"    {name:12s} → {state.name:25s} conf={conf:.2f}  {mark}  (expected {expected})")


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — Drift detection with B200 values (relative detector)
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 5 — Drift detection (relative, BMC T_ref)")

# Drift detection works on relative deviations from baseline.
# Use BMC T_ref path since it produces sensible R_theta range.
if r_load_bmc is not None and r_degd_bmc is not None:
    b200_min_std = getattr(profile, "drift_min_std", 0.010)
    print(f"\n  B200 profile.drift_min_std = {b200_min_std:.4f} C/W")

    # Test with old T4 floor (0.010) and new B200 floor (0.002)
    for std_label, min_std_val in [("T4 floor (0.010, broken)", 0.010), (f"B200 floor ({b200_min_std:.3f}, fixed)", b200_min_std)]:
        detector = DriftDetector(k_warn=2.0, k_critical=3.5, min_std=min_std_val)
        ts_d = time.time()

        # Feed 25 healthy load windows to build a baseline
        for i in range(25):
            ts_d += 5.0
            noise = 0.001 * math.sin(i * 1.7)
            result = detector.update(0, ts_d, r_load_bmc + noise, GPUState.UNDER_LOAD)

        # Now feed degraded values
        print(f"\n  Feeding 5 degraded windows with {std_label}:")
        dr = result
        for i in range(5):
            ts_d += 5.0
            dr = detector.update(0, ts_d, r_degd_bmc, GPUState.UNDER_LOAD)
            if dr.baseline_mean:
                sigma = dr.sigma_score
                print(f"    sample {i+1}: R_theta={r_degd_bmc:.4f}  sigma={sigma:.2f}  "
                      f"drifting={dr.is_drifting}  critical={dr.is_critical}")

        if dr.is_drifting or dr.is_critical:
            OK(f"Drift detector fires with {std_label}",
               f"sigma={dr.sigma_score:.2f} > k_warn=2.0")
        else:
            _sigma_str = f"{dr.sigma_score:.2f}" if dr.sigma_score is not None else "N/A"
            if "broken" in std_label:
                WARN(
                    "Drift detector does NOT fire with T4 noise floor (expected failure)",
                    f"sigma={_sigma_str} < k_warn=2.0 because min_std={min_std_val} swamps signal."
                )
            else:
                CRIT(
                    "Drift detector still does NOT fire after B200 floor fix",
                    f"sigma={_sigma_str} < k_warn=2.0 even with min_std={min_std_val}."
                )


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 6 — NVLink correlated heat (all 8 GPUs simultaneously)
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 6 — NVLink correlated fleet event (all 8 GPUs)")

gpu_names = {i: "NVIDIA B200 SXM6" for i in range(8)}
correlator = FleetCorrelator()
correlator.register_gpu_names(gpu_names)

ts = time.time()

# All 8 GPUs enter DRIFTING simultaneously (NVLink all-reduce heat spike)
gpu_states = {i: GPUState.DRIFTING for i in range(8)}
alert = correlator.check(gpu_states, ts)

if alert is not None:
    nvlink = alert.context.get("nvlink_correlated", False)
    drain  = alert.context.get("drain_recommended", True)
    print(f"\n  Alert fired: {alert.context.get('severity')}")
    print(f"  NVLink correlated: {nvlink}")
    print(f"  Drain recommended: {drain}")
    print(f"  Message snippet: {alert.message[:120]}…")
    if nvlink and not drain:
        OK("NVLink correlation detected — drain_recommended=False ✓")
    elif not nvlink:
        WARN(
            "NVLink correlation NOT detected on B200 all-GPU event",
            "All 8 GPUs on a DGX B200 use NVLink fabric. 8/8 simultaneous DRIFTING "
            "should be flagged as likely NVLink workload, not cooling failure."
        )
else:
    CRIT("Fleet correlator emitted no alert for 8/8 simultaneous DRIFTING",
         "Cooldown or min_gpus threshold incorrectly filtered the event.")

# Subset case: 4/8 GPUs drifting (genuinely suspicious, NVLink less likely)
ts += 200  # past cooldown
gpu_states_partial = {i: (GPUState.DRIFTING if i < 4 else GPUState.UNDER_LOAD) for i in range(8)}
alert2 = correlator.check(gpu_states_partial, ts)
if alert2:
    nvlink2 = alert2.context.get("nvlink_correlated", False)
    print(f"\n  4/8 GPUs drifting — NVLink correlated: {nvlink2}  drain: {alert2.context.get('drain_recommended')}")
    if not nvlink2:
        OK("Partial-fleet event correctly NOT flagged as NVLink correlated")
    else:
        WARN("Partial-fleet event incorrectly flagged as NVLink correlated",
             "NVLink suppression should only fire when ALL GPUs are simultaneously affected.")


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 7 — Power = 0 / underreporting edge case (dual-die NVML)
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 7 — B200 dual-die power reporting edge case")

# Simulate what happens if NVML reports half the actual power (per-die report)
# GPU at 900W actual → NVML reports 450W
suspicious_power = LOAD_POWER_W / 2.0

# The new sanity check in collector.py skips when power < 0.4 * idle_floor_w while util > 15%.
# idle_floor_w for B200 = 85W. Trigger: power < 34W while util > 15%.
b200_idle_floor = profile.idle_floor_w
trigger_threshold = b200_idle_floor * 0.4
print(f"\n  B200 idle_floor_w      = {b200_idle_floor} W")
print(f"  Sanity check triggers at: power < {trigger_threshold:.1f} W while util > 15%")
print(f"  Simulated NVML report:    power = {suspicious_power:.0f} W (half of actual {LOAD_POWER_W}W)")
print(f"  Util during test:         90%")

if suspicious_power < trigger_threshold:
    OK("Sanity check WOULD fire — sample dropped ✓")
else:
    r_suspicious, _ = compute_rtheta(T_J_LOAD, COOLANT_INLET_C, suspicious_power)
    if r_suspicious is not None:
        WARN(
            "Power underreporting NOT caught by sanity check",
            f"Power={suspicious_power:.0f}W > trigger threshold={trigger_threshold:.1f}W. "
            f"R_theta would be computed as {r_suspicious:.4f} C/W instead of "
            f"{r_load_bmc:.4f} C/W — a {(r_suspicious/r_load_bmc - 1)*100:.0f}% overestimate. "
            f"Per-die NVML reporting on B200 (450W each) would produce spurious drift alerts. "
            f"Threshold needs tightening: consider power < 0.7 * idle_floor for B200."
        )


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 8 — Pre-flight check (calibration gate)
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 8 — Pre-flight calibration gate")

from theta.agent.calibrate import CalibrationManager as CM

tmp_cal = CM(_file=Path("/tmp/sim_b200_calibration_empty.json"))

# Ensure the file doesn't exist
if Path("/tmp/sim_b200_calibration_empty.json").exists():
    Path("/tmp/sim_b200_calibration_empty.json").unlink()
tmp_cal2 = CM(_file=Path("/tmp/sim_b200_calibration_empty.json"))

# Simulate the pre-flight check logic from daemon._check_hardware_ready()
from theta.agent.hw_profiles import resolve_or_default
prof = resolve_or_default("NVIDIA B200 SXM6")

would_block = prof.confidence == "extrapolated" and tmp_cal2.get(0) is None
if would_block:
    OK("Pre-flight gate correctly BLOCKS uncalibrated B200",
       f"confidence={prof.confidence}, no calibration for GPU 0")
else:
    CRIT("Pre-flight gate FAILS to block uncalibrated B200",
         "daemon.run() would start with T4 thresholds on B200.")


# ════════════════════════════════════════════════════════════════════════════
# SCENARIO 9 — SIGMA_STRICT compatibility with B200 R_theta range
# ════════════════════════════════════════════════════════════════════════════

HDR("SCENARIO 9 — SIGMA_STRICT window threshold compatibility")

print(f"\n  SIGMA_STRICT = {SIGMA_STRICT:.3f} C/W  (the steady-state filter threshold)")
print(f"  B200 R_theta under load: {r_load_bmc:.4f} C/W  (BMC T_ref={COOLANT_INLET_C}°C)")
print(f"  Natural B200 noise band: ~{r_load_bmc * 0.05:.5f} C/W  (5% of R_theta at load)")

natural_sigma = r_load_bmc * 0.04  # estimated 4% natural noise
if natural_sigma < SIGMA_STRICT:
    OK(f"Natural B200 noise ({natural_sigma:.5f} C/W) < SIGMA_STRICT ({SIGMA_STRICT:.3f} C/W) — windows should stabilize")
else:
    WARN(
        "B200 natural noise may exceed SIGMA_STRICT",
        f"Estimated noise σ={natural_sigma:.5f} C/W > SIGMA_STRICT={SIGMA_STRICT:.3f}. "
        f"Steady-state windows may never be marked stable. All classification skipped."
    )

# Check the absolute range problem differently
print(f"\n  B200 full R_theta range (BMC T_ref): idle={r_idle_bmc:.4f} → load={r_load_bmc:.4f} C/W")
print(f"  Range width: {abs(r_idle_bmc - r_load_bmc):.4f} C/W")
print(f"  T4 range for comparison: idle=1.28 → load=0.72 = 0.56 C/W")
print(f"  B200 range is ~{0.56 / abs(r_idle_bmc - r_load_bmc):.0f}× NARROWER than T4")
WARN(
    "B200 R_theta dynamic range is much narrower than T4",
    f"Full range={abs(r_idle_bmc - r_load_bmc):.4f} C/W vs T4's 0.56 C/W. "
    f"Drift detector baseline noise floor matters much more at this scale. "
    f"k_warn=2.0 may need recalibration — small measurement noise causes many false drift alerts. "
    f"Recommend collecting 48h of B200 baseline data before tuning k_warn."
)


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 64}")
print(f"  AI FACTORY SIMULATION REPORT")
print(f"  DGX B200  ·  8 GPUs  ·  liquid cold-plate  ·  20°C coolant")
print(f"{'═' * 64}")

crits  = [f for f in findings if f.severity == "CRITICAL"]
warns  = [f for f in findings if f.severity == "WARNING"]
passed = 9 - len(crits) - len(warns)

print(f"\n  {len(crits)} CRITICAL  ·  {len(warns)} WARNING  ·  {passed}/9 scenarios passed\n")

for f in crits:
    print(f"  ❌ [{f.severity}] {f.title}")
    print(f"     {f.detail[:120]}…" if len(f.detail) > 120 else f"     {f.detail}")
    print()

for f in warns:
    print(f"  ⚠️  [{f.severity}] {f.title}")
    print(f"     {f.detail[:120]}…" if len(f.detail) > 120 else f"     {f.detail}")
    print()

print(f"{'─' * 64}")
print(f"\n  RECOMMENDED FIXES before AI Factory install:")
if crits:
    print(f"\n  MUST FIX (daemon will produce wrong output without these):")
    for i, f in enumerate(crits, 1):
        print(f"    {i}. {f.title}")
if warns:
    print(f"\n  SHOULD FIX (reduces false alerts and improves accuracy):")
    for i, f in enumerate(warns, 1):
        print(f"    {i}. {f.title}")
print()
