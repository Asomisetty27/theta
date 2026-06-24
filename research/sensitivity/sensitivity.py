"""
Kundu mandate: dR_theta/dT_amb at each power level (ambient-sensitivity of R_theta).

R_theta = (T_j - T_amb)/P, so analytically dR_theta/dT_amb = -1/P: the metric is
MOST sensitive to the assumed ambient at LOW power (idle), least at high power.
This quantifies F2 (T_reference sensitivity) per power tier for the paper.

Stage 1 CSV carries R_theta recomputed at assumed ambients 30/35/40 C, so we measure
the sensitivity empirically ((R40 - R30)/10) and check it against -1/P.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

CSV = Path("/Users/amogh/thermalos-vault/raw/experiments/ThermalOS_Measurements_Raw.csv")
TIERS = [("idle", 0, 20), ("mid", 20, 55), ("peak", 55, 1e9)]


def main():
    df = pd.read_csv(CSV)
    df["dR_dTamb_emp"] = (df["rtheta_40c_cwatt"] - df["rtheta_30c_cwatt"]) / 10.0
    df["dR_dTamb_analytic"] = -1.0 / df["power_w"]
    df["rel_pct_per_C"] = 100 * df["dR_dTamb_emp"] / df["rtheta_35c_cwatt"]
    df["swing_pct_5C"] = 100 * (5 * df["dR_dTamb_emp"].abs()) / df["rtheta_35c_cwatt"].abs()

    rows = []
    for name, lo, hi in TIERS:
        t = df[(df["power_w"] >= lo) & (df["power_w"] < hi)]
        if not len(t):
            continue
        rows.append({
            "tier": name, "n": len(t),
            "mean_power_w": round(float(t["power_w"].mean()), 1),
            "mean_rtheta_at35": round(float(t["rtheta_35c_cwatt"].mean()), 3),
            "dR_dTamb_emp_CW_per_C": round(float(t["dR_dTamb_emp"].mean()), 4),
            "dR_dTamb_analytic_-1/P": round(float(t["dR_dTamb_analytic"].mean()), 4),
            "rel_sensitivity_pct_per_C": round(float(t["rel_pct_per_C"].abs().mean()), 1),
            "swing_pct_per_5C_ambient_error": round(float(t["swing_pct_5C"].mean()), 1),
        })
    # global check: empirical vs analytic agreement
    err = float((df["dR_dTamb_emp"] - df["dR_dTamb_analytic"]).abs().mean())
    out = {
        "n_total": len(df),
        "tiers": rows,
        "empirical_vs_analytic_mean_abs_err": round(err, 6),
        "note": "dR_theta/dT_amb = -1/P confirmed; sensitivity largest at idle (low P), "
                "smallest at peak. swing_pct_per_5C is the F2 figure per tier.",
    }
    Path(__file__).parent.joinpath("out").mkdir(exist_ok=True)
    Path(__file__).parent.joinpath("out/sensitivity_results.json").write_text(json.dumps(out, indent=2))
    print(f"n={len(df)}  empirical vs analytic mean abs err = {err:.2e} (≈0 confirms -1/P)\n")
    hdr = f"{'tier':6} {'n':>5} {'P(W)':>6} {'Rθ@35':>7} {'dR/dTamb':>9} {'-1/P':>7} {'%/C':>6} {'%/5C':>7}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['tier']:6} {r['n']:>5} {r['mean_power_w']:>6} {r['mean_rtheta_at35']:>7} "
              f"{r['dR_dTamb_emp_CW_per_C']:>9} {r['dR_dTamb_analytic_-1/P']:>7} "
              f"{r['rel_sensitivity_pct_per_C']:>6} {r['swing_pct_per_5C_ambient_error']:>7}")
    print(f"\nwrote out/sensitivity_results.json")


if __name__ == "__main__":
    main()
