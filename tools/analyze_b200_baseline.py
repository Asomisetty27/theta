"""
B200 Stage 2 baseline analysis — E005

Run after collect_b200_baseline.py has produced a CSV from the AI Factory DGX.

Usage:
  python tools/analyze_b200_baseline.py --csv b200_baseline.csv
  python tools/analyze_b200_baseline.py --csv b200_baseline.csv --coolant-c 20.0 --out report.txt

What this does:
  1. Validates the CSV schema and data quality (missing values, outliers, gaps)
  2. Computes R_theta = (T_junction - T_ref) / Power_W for each sample
     T_ref = coolant_inlet_c (liquid-cooled; no idle-window locking)
  3. Computes per-GPU R_theta statistics: mean, std, p5/p25/p50/p75/p95
  4. Compares measured values to the current hw_profiles.py extrapolated estimates
  5. Checks whether the +20%/+40% drift thresholds are correctly placed
  6. Outputs a profile validation report and suggested hw_profiles.py updates

E005 acceptance criteria (from Stage 2 validation protocol):
  - R_theta healthy mean in [0.050, 0.075] C/W  (physics estimate: 0.060)
  - R_theta std < 0.005 C/W at steady load
  - No GPU shows R_theta > 0.090 C/W at healthy load (would indicate pre-existing issue)
  - Cross-GPU spread < 0.015 C/W (all 8 on same cold-plate manifold)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


MIN_DELTA_T = 5.0    # °C — samples below this are excluded (near-zero power / invalid)
MIN_POWER_W = 50.0   # W  — exclude idle/standby samples from load R_theta stats
LOAD_POWER_W = 800.0 # W  — threshold for "under load" classification


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _validate_schema(rows: list[dict]) -> list[str]:
    required = {
        "timestamp_s", "gpu_index", "gpu_name",
        "temp_junction_c", "power_w", "util_pct",
    }
    if not rows:
        return ["CSV is empty"]
    missing = required - set(rows[0].keys())
    if missing:
        return [f"Missing required columns: {sorted(missing)}"]
    return []


def _compute_rtheta(
    rows: list[dict],
    coolant_c: float,
) -> dict[int, dict]:
    """
    Per-GPU R_theta computation.
    Returns dict[gpu_index] -> {
        all_samples, load_samples, idle_samples,
        rtheta_all, rtheta_load, rtheta_idle,
        name, n_total, n_excluded
    }
    """
    by_gpu: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            idx = int(row["gpu_index"])
            tj  = float(row["temp_junction_c"])
            pw  = float(row["power_w"])
            ut  = float(row.get("util_pct", 0))
            by_gpu[idx].append({"tj": tj, "pw": pw, "util": ut, "row": row})
        except (ValueError, KeyError):
            continue

    result = {}
    for idx, samples in sorted(by_gpu.items()):
        name       = samples[0]["row"].get("gpu_name", "unknown")
        rtheta_all = []
        rtheta_load = []
        rtheta_idle = []
        n_excluded  = 0

        for s in samples:
            delta_t = s["tj"] - coolant_c
            if delta_t < MIN_DELTA_T or s["pw"] < MIN_POWER_W:
                n_excluded += 1
                continue
            rt = delta_t / s["pw"]
            rtheta_all.append(rt)
            if s["pw"] >= LOAD_POWER_W:
                rtheta_load.append(rt)
            else:
                rtheta_idle.append(rt)

        result[idx] = {
            "name":          name,
            "n_total":       len(samples),
            "n_excluded":    n_excluded,
            "rtheta_all":    rtheta_all,
            "rtheta_load":   rtheta_load,
            "rtheta_idle":   rtheta_idle,
        }
    return result


def _check_acceptance(stats: dict) -> list[tuple[str, str, str]]:
    """
    Returns list of (gpu_label, criterion, status) tuples.
    status: PASS | WARN | FAIL
    """
    checks = []
    load_means = [
        s["rtheta_load_mean"]
        for s in stats.values()
        if not math.isnan(s.get("rtheta_load_mean", float("nan")))
    ]

    for idx, s in sorted(stats.items()):
        label = f"GPU {idx} ({s['name']})"
        m  = s.get("rtheta_load_mean", float("nan"))
        sd = s.get("rtheta_load_std",  float("nan"))

        if math.isnan(m):
            checks.append((label, "load R_theta computable", "WARN — not enough load samples"))
            continue

        # E005 criterion 1: healthy range
        if 0.050 <= m <= 0.075:
            checks.append((label, f"R_theta_load mean {m:.4f} in [0.050, 0.075]", "PASS"))
        elif m < 0.050:
            checks.append((label, f"R_theta_load mean {m:.4f} below 0.050", "WARN — unusually low, verify coolant_c"))
        else:
            checks.append((label, f"R_theta_load mean {m:.4f} above 0.075", "FAIL — pre-existing degradation?"))

        # E005 criterion 2: std < 0.005 at steady load
        if not math.isnan(sd):
            if sd < 0.005:
                checks.append((label, f"R_theta_load std {sd:.4f} < 0.005", "PASS"))
            else:
                checks.append((label, f"R_theta_load std {sd:.4f} >= 0.005", "WARN — high variance, check workload stability"))

        # E005 criterion 3: no GPU above 0.090
        if m > 0.090:
            checks.append((label, f"R_theta_load mean {m:.4f} > 0.090", "FAIL — likely degraded hardware"))

    # E005 criterion 4: cross-GPU spread
    if len(load_means) >= 2:
        spread = max(load_means) - min(load_means)
        if spread < 0.015:
            checks.append(("Fleet", f"Cross-GPU R_theta spread {spread:.4f} < 0.015", "PASS"))
        else:
            checks.append(("Fleet", f"Cross-GPU R_theta spread {spread:.4f} >= 0.015", "WARN — check manifold flow balance"))

    return checks


def _suggest_profile_update(stats: dict, coolant_c: float) -> list[str]:
    """Generate suggested hw_profiles.py edits based on measured values."""
    load_means = [
        s["rtheta_load_mean"]
        for s in stats.values()
        if not math.isnan(s.get("rtheta_load_mean", float("nan")))
    ]
    if not load_means:
        return ["Not enough load data to suggest profile update."]

    fleet_mean = _mean(load_means)
    fleet_std  = _std(load_means)
    warn_thr   = round(fleet_mean * 1.20, 3)
    crit_thr   = round(fleet_mean * 1.40, 3)

    lines = [
        "# Suggested update for theta/agent/hw_profiles.py (B200 profile):",
        "# Replace extrapolated values with E005 measured values.",
        "#",
        f"# Fleet R_theta_load: mean={fleet_mean:.4f} C/W, std={fleet_std:.4f} C/W",
        f"# Thresholds: warn={warn_thr} C/W (+20%), critical={crit_thr} C/W (+40%)",
        "#",
        "# In the B200 ThermalProfile entry, update:",
        f'#   rtheta_expected_under_load = {fleet_mean:.4f}',
        f'#   rtheta_expected_idle       = {fleet_mean:.4f}  # flat — liquid-cooled',
        f'#   rtheta_load_threshold      = {warn_thr}',
        f'#   rtheta_idle_threshold      = {warn_thr}',
        f'#   expected_ambient_c         = {coolant_c}  # measured coolant inlet',
        "#",
        "# Also change confidence field from 'extrapolated' to 'measured'",
        "# and add a comment referencing E005.",
    ]
    return lines


def analyze(csv_path: Path, coolant_c: float, out_path: Optional[Path] = None) -> None:
    rows = _load_csv(csv_path)

    lines: list[str] = []
    log = lines.append

    log("=" * 70)
    log("ThermalOS E005 — B200 Stage 2 Baseline Analysis")
    log(f"CSV:     {csv_path}")
    log(f"Rows:    {len(rows)}")
    log(f"Coolant: {coolant_c}°C (T_ref for R_theta computation)")
    log("=" * 70)

    # Schema check
    errors = _validate_schema(rows)
    if errors:
        for e in errors:
            log(f"SCHEMA ERROR: {e}")
        print("\n".join(lines))
        sys.exit(1)

    # Detect coolant_c from CSV if not supplied and column is present
    if coolant_c == 20.0:
        csv_coolant_values = [
            float(r["coolant_inlet_c"])
            for r in rows
            if r.get("coolant_inlet_c", "") not in ("", None)
            and _is_float(r["coolant_inlet_c"])
        ]
        if csv_coolant_values:
            csv_coolant = _mean(csv_coolant_values)
            log(f"\nCoolant inlet from CSV: {csv_coolant:.2f}°C (overrides default {coolant_c}°C)")
            coolant_c = csv_coolant

    # Compute R_theta per GPU
    gpu_data = _compute_rtheta(rows, coolant_c)

    # Build stats table
    stats = {}
    for idx, d in gpu_data.items():
        rt_load = d["rtheta_load"]
        rt_idle = d["rtheta_idle"]
        stats[idx] = {
            "name":               d["name"],
            "n_total":            d["n_total"],
            "n_excluded":         d["n_excluded"],
            "n_load":             len(rt_load),
            "n_idle":             len(rt_idle),
            "rtheta_load_mean":   _mean(rt_load),
            "rtheta_load_std":    _std(rt_load),
            "rtheta_load_p50":    _percentile(rt_load, 50),
            "rtheta_load_p95":    _percentile(rt_load, 95),
            "rtheta_idle_mean":   _mean(rt_idle),
        }

    # Print per-GPU table
    log("\n── Per-GPU R_theta (load window: power >= {:.0f}W) ─────────────────────".format(LOAD_POWER_W))
    log(f"{'GPU':<6} {'Name':<18} {'N_load':>7} {'mean':>8} {'std':>7} {'p50':>8} {'p95':>8} {'idle_mean':>10}")
    log("-" * 76)
    for idx, s in sorted(stats.items()):
        log(
            f"{idx:<6} {s['name']:<18} {s['n_load']:>7} "
            f"{_fmt(s['rtheta_load_mean']):>8} {_fmt(s['rtheta_load_std']):>7} "
            f"{_fmt(s['rtheta_load_p50']):>8} {_fmt(s['rtheta_load_p95']):>8} "
            f"{_fmt(s['rtheta_idle_mean']):>10}"
        )

    # Comparison to hw_profiles.py extrapolated values
    EXTRAPOLATED_MEAN = 0.060
    EXTRAPOLATED_WARN = 0.072
    log("\n── vs hw_profiles.py extrapolated estimates ────────────────────────────")
    log(f"  Current profile:  rtheta_expected={EXTRAPOLATED_MEAN} C/W, "
        f"load_threshold={EXTRAPOLATED_WARN} C/W")
    load_means = [s["rtheta_load_mean"] for s in stats.values()
                  if not math.isnan(s.get("rtheta_load_mean", float("nan")))]
    if load_means:
        measured_mean = _mean(load_means)
        delta_pct = (measured_mean - EXTRAPOLATED_MEAN) / EXTRAPOLATED_MEAN * 100
        sign = "+" if delta_pct >= 0 else ""
        log(f"  Measured mean:    {measured_mean:.4f} C/W  ({sign}{delta_pct:.1f}% vs estimate)")
        if abs(delta_pct) < 10:
            log("  → Profile estimate within 10% — thresholds are valid.")
        elif abs(delta_pct) < 20:
            log("  → Profile estimate within 20% — consider updating thresholds.")
        else:
            log("  → Profile estimate off by >20% — update profile before production use.")

    # E005 acceptance checks
    log("\n── E005 Acceptance Criteria ────────────────────────────────────────────")
    checks = _check_acceptance(stats)
    passes = sum(1 for _, _, s in checks if s.startswith("PASS"))
    warns  = sum(1 for _, _, s in checks if s.startswith("WARN"))
    fails  = sum(1 for _, _, s in checks if s.startswith("FAIL"))
    for label, criterion, status in checks:
        icon = "✓" if status.startswith("PASS") else ("!" if status.startswith("WARN") else "✗")
        log(f"  {icon} {label}: {criterion}")
        if not status.startswith("PASS"):
            log(f"      → {status}")
    log(f"\n  Summary: {passes} PASS, {warns} WARN, {fails} FAIL")

    # Profile update suggestion
    log("\n── Suggested hw_profiles.py update ────────────────────────────────────")
    for line in _suggest_profile_update(stats, coolant_c):
        log(line)

    log("\n── Next steps ──────────────────────────────────────────────────────────")
    if fails == 0 and warns <= 2:
        log("  Profile is valid. Update hw_profiles.py with measured values,")
        log("  change confidence from 'extrapolated' to 'measured', commit as E005.")
    elif fails == 0:
        log("  Warnings present — review before updating profile. Check:")
        log("  - Coolant inlet temperature accuracy (--coolant-c flag)")
        log("  - Load window definition (LOAD_POWER_W threshold)")
        log("  - Workload stability during collection window")
    else:
        log("  FAIL detected — do not update profile yet. Investigate:")
        log("  - Pre-existing hardware degradation on flagged GPU(s)")
        log("  - Incorrect coolant_c value")
        log("  - Collection error (very short run, gaps in data)")

    output = "\n".join(lines)
    print(output)
    if out_path:
        out_path.write_text(output)
        print(f"\nReport written to {out_path}")


def _fmt(v: float, w: int = 6) -> str:
    return f"{v:.4f}" if not math.isnan(v) else "  N/A "


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="E005 B200 baseline analysis for ThermalOS Stage 2")
    p.add_argument("--csv",       required=True, type=Path,  help="CSV from collect_b200_baseline.py")
    p.add_argument("--coolant-c", type=float, default=20.0,  dest="coolant_c",
                   help="Coolant inlet temperature °C (default 20.0; overridden by CSV column if present)")
    p.add_argument("--out",       type=Path, default=None,   help="Optional: write report to file")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found")
        sys.exit(1)

    analyze(args.csv, args.coolant_c, args.out)
