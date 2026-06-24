"""
Kundu mandate #2: Bayesian (Naive Bayes) vs Random Forest thermal-state classifier,
made reproducible (stratified k-fold CV) with the steady-state-window ablation
computed from the data, plus the Bayesian model equation.

Replaces hardcoded R_theta thresholds with a probabilistic classifier (Kundu's
direction). The headline methodological result: the steady-state gate (sigma_Rtheta
< 0.03 C/W over a short window, the window.py rule) lifts simple Gaussian NB from
~mid-80s to ~RF-level accuracy. That gate is what makes a simple, interpretable
Bayesian model competitive with RF, which is the point for the paper.

States (4): clean_idle / under_load / zombie_recovery / child_exit_recovery.
Features: rtheta_cwatt, power_w, util_pct, perf_state (P0/P8 -> 0/8).
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

CSV = Path("/Users/amogh/thermalos-vault/raw/experiments/ThermalOS_Measurements_Raw.csv")
FEATURES = ["rtheta_cwatt", "power_w", "util_pct", "perf_state_num"]
SIGMA_GATE = 0.03   # C/W, window.py steady-state rule
WIN = 10            # rolling window (samples) for sigma


def to_class(phase: str):
    p = str(phase)
    if p == "clean_idle" or p.endswith("pre_load_baseline"):
        return 0  # clean_idle (idle states)
    if "separate_process_load" in p or p.startswith("under_load"):
        return 1  # under_load
    if "extended_post_load_recovery" in p:
        return 2  # zombie_recovery (same-process, stuck P0)
    if "recovery_after_child_exit" in p:
        return 3  # child_exit_recovery
    return None   # post_load_cooldown / other -> drop (ambiguous)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["perf_state_num"] = df["perf_state"].astype(str).str.replace("P", "", regex=False)
    df["perf_state_num"] = pd.to_numeric(df["perf_state_num"], errors="coerce").fillna(0)
    df["y"] = df["phase"].map(to_class)
    df = df.dropna(subset=["y"] + FEATURES)
    df["y"] = df["y"].astype(int)
    # steady-state mask: rolling sigma of R_theta within each (experiment, phase) segment
    df = df.sort_values(["experiment_id", "phase", "trial_second"])
    df["roll_sigma"] = (df.groupby(["experiment_id", "phase"])["rtheta_cwatt"]
                          .transform(lambda s: s.rolling(WIN, min_periods=WIN).std()))
    df["steady"] = df["roll_sigma"] < SIGMA_GATE
    return df


def cv_scores(X, y) -> dict:
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    out = {}
    for name, clf in [("naive_bayes", GaussianNB()),
                      ("random_forest", RandomForestClassifier(n_estimators=200, random_state=0)),
                      ("decision_tree", DecisionTreeClassifier(random_state=0))]:
        r = cross_validate(clf, X, y, cv=skf, scoring=["accuracy", "f1_macro"])
        out[name] = {"accuracy": round(float(r["test_accuracy"].mean()), 4),
                     "f1_macro": round(float(r["test_f1_macro"].mean()), 4)}
    return out


def nb_equation(df: pd.DataFrame) -> dict:
    """Gaussian NB per-class feature means/vars = the model 'equation' Kundu asked for."""
    nb = GaussianNB().fit(df[FEATURES].values, df["y"].values)
    names = {0: "clean_idle", 1: "under_load", 2: "zombie_recovery", 3: "child_exit_recovery"}
    eq = {"priors": {names[int(c)]: round(float(p), 4) for c, p in zip(nb.classes_, nb.class_prior_)},
          "per_class_gaussians": {}}
    for i, c in enumerate(nb.classes_):
        eq["per_class_gaussians"][names[int(c)]] = {
            f: {"mean": round(float(nb.theta_[i][j]), 4), "var": round(float(nb.var_[i][j]), 5)}
            for j, f in enumerate(FEATURES)}
    return eq


def main():
    df = prep(pd.read_csv(CSV))
    allrows, steady = df, df[df["steady"]]
    res = {
        "n_all": int(len(allrows)), "n_steady": int(len(steady)),
        "class_counts": {str(k): int(v) for k, v in allrows["y"].value_counts().items()},
        "raw": cv_scores(allrows[FEATURES].values, allrows["y"].values),
        "steady_state": cv_scores(steady[FEATURES].values, steady["y"].values),
        "nb_model_equation": nb_equation(steady),
    }
    Path(__file__).parent.joinpath("out").mkdir(exist_ok=True)
    Path(__file__).parent.joinpath("out/classifier_results.json").write_text(json.dumps(res, indent=2))
    print(f"n_all={res['n_all']}  n_steady={res['n_steady']}  classes={res['class_counts']}\n")
    print(f"{'model':16} {'raw acc':>8} {'raw F1':>8}   {'steady acc':>11} {'steady F1':>10}")
    print("-" * 60)
    for m in ("naive_bayes", "random_forest", "decision_tree"):
        r, s = res["raw"][m], res["steady_state"][m]
        print(f"{m:16} {r['accuracy']:>8} {r['f1_macro']:>8}   {s['accuracy']:>11} {s['f1_macro']:>10}")
    print("\nNB steady-state class means (R_theta, power, util, pstate):")
    for c, g in res["nb_model_equation"]["per_class_gaussians"].items():
        print(f"  {c:20} R_theta={g['rtheta_cwatt']['mean']:.3f}  P={g['power_w']['mean']:.0f}  "
              f"util={g['util_pct']['mean']:.0f}  pstate={g['perf_state_num']['mean']:.1f}")
    print("\nwrote out/classifier_results.json")


if __name__ == "__main__":
    main()
