"""
Lead-time survival pipeline for R_theta-based GPU cooling-degradation early warning.

Turns the "does R_theta rise before a throttle/failure, with usable lead time?"
question (Q_lead_time) into a measured result instead of an assertion.

Pipeline:
  1. load per-GPU telemetry (node, gpu_ordinal, temp_c, power_w over time)
  2. R_theta = (temp - T_ref) / power_w
  3. peer-relative robust-z: subtract each GPU's healthy-window baseline, then
     z = (residual - median_peers) / (1.4826 * MAD_peers). Baseline-correction
     removes baseboard-position structure; the cross-peer step cancels a shared
     T_ref error (same logic as the E009/F7 median-polish).
  4. event = faulty GPU's ground-truth degradation crossing a severity level
  5. evaluate:
       - early-warning LEAD TIME (detector alert -> event) + false-alarm rate
       - time-varying Cox PH: hazard of event vs peer-z (+ concordance/C-index)
       - Weibull baseline survival curve
  6. write results JSON

Built and tested on the synthetic Della scenarios (ground-truth faults). It is a
DETECTOR TESTBED, not real-world evidence: 8 GPUs / 1 event per scenario, so the
Cox model is demonstrative, not powered. The pipeline is built so it becomes
statistically powered the moment a real multi-fleet export (GWDG/NCSA) lands.
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path
import numpy as np
import pandas as pd

T_REF = 25.0          # assumed inlet/ambient (C). Peer-z is baseline-corrected so this mostly cancels.
HEALTHY_FRAC = 0.20   # first 20% of the window defines each GPU's healthy baseline
K_WARN = 5.0          # peer-z alert threshold (robust sigma); 5 not 3 because a month
                      # of 1-min samples crosses 3-sigma by chance for every unit
SUSTAIN = 10          # consecutive alert samples required (debounce / hysteresis)
SEVERITIES = [1.10, 1.15, 1.20, 1.30]   # ground-truth multiplier event levels to report


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["power_w"] = df["nvidia_gpu_power_usage_milliwatts"] / 1000.0
    df["temp_c"] = df["nvidia_gpu_temperature_celsius"]
    df["rtheta"] = (df["temp_c"] - T_REF) / df["power_w"]
    df["gpu"] = df["node"].astype(str) + ":" + df["gpu_ordinal"].astype(str)
    return df[["timestamp", "node", "gpu_ordinal", "gpu", "temp_c", "power_w", "rtheta"]]


SMOOTH_W = 20   # rolling-median steady-state window (samples), matches E009


def peer_z(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["gpu", "timestamp"]).copy()
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    cut = t0 + HEALTHY_FRAC * (t1 - t0)
    base = (df[df["timestamp"] <= cut].groupby("gpu")["rtheta"].median().rename("baseline"))
    df = df.merge(base, on="gpu", how="left")
    df["residual"] = df["rtheta"] - df["baseline"]
    # rolling-median smoothing per GPU suppresses per-sample noise (steady-state window)
    df["res_s"] = df.groupby("gpu")["residual"].transform(
        lambda s: s.rolling(SMOOTH_W, min_periods=1).median())
    # fixed healthy-window noise floor (pooled robust sigma) — stable z denominator,
    # NOT the tiny instantaneous cross-peer MAD (which trips healthy units on noise)
    hres = df.loc[df["timestamp"] <= cut, "res_s"].values
    sigma = 1.4826 * np.median(np.abs(hres - np.median(hres))) + 1e-9
    # remove common-mode drift (median across peers per timestamp), scale by healthy sigma
    med = df.groupby("timestamp")["res_s"].transform("median")
    df["peer_z"] = (df["res_s"] - med) / sigma
    return df


def faulty_gpu(df: pd.DataFrame, gt: pd.DataFrame) -> tuple[str, str]:
    col = [c for c in gt.columns if c.endswith("_r_theta_multiplier")][0]
    ordinal = int(re.search(r"gpu(\d+)_", col).group(1))
    node = df["node"].iloc[0]
    return f"{node}:{ordinal}", col


def event_times(gt: pd.DataFrame, col: str, levels: list[float]) -> dict[float, int | None]:
    out = {}
    for lv in levels:
        hit = gt[gt[col] >= lv]
        out[lv] = int(hit["timestamp"].iloc[0]) if len(hit) else None
    return out


def first_alert(df_gpu: pd.DataFrame) -> int | None:
    a = (df_gpu["peer_z"] >= K_WARN).astype(int).values
    run = 0
    for i, v in enumerate(a):
        run = run + 1 if v else 0
        if run >= SUSTAIN:
            return int(df_gpu["timestamp"].values[i - SUSTAIN + 1])
    return None


def lead_time_eval(df: pd.DataFrame, fid: str, evt: dict) -> dict:
    alerts = {gpu: first_alert(g) for gpu, g in df.groupby("gpu")}
    f_alert = alerts[fid]
    healthy_fp = [gpu for gpu, a in alerts.items() if gpu != fid and a is not None]
    res = {
        "faulty_gpu": fid,
        "faulty_first_alert_ts": f_alert,
        "false_alarm_gpus": healthy_fp,
        "false_alarm_rate": round(len(healthy_fp) / max(1, df["gpu"].nunique() - 1), 3),
        "lead_time_hours": {},
    }
    for lv, ets in evt.items():
        if ets is None or f_alert is None:
            res["lead_time_hours"][str(lv)] = None
        else:
            res["lead_time_hours"][str(lv)] = round((ets - f_alert) / 3600.0, 2)
    return res


def survival_models(df: pd.DataFrame, fid: str, event_ts: int | None) -> dict:
    from lifelines import CoxTimeVaryingFitter, WeibullFitter
    # hourly downsample for tractability (day-scale leads)
    d = df.copy()
    d["hr"] = (d["timestamp"] - d["timestamp"].min()) // 3600
    h = d.groupby(["gpu", "hr"]).agg(peer_z=("peer_z", "mean"), ts=("timestamp", "min")).reset_index()
    end = h["hr"].max() + 1
    rows = []
    for gpu, g in h.groupby("gpu"):
        g = g.sort_values("hr")
        ev_hr = None
        if gpu == fid and event_ts is not None:
            ev_hr = int((event_ts - d["timestamp"].min()) // 3600)
        for _, r in g.iterrows():
            start = int(r["hr"]); stop = start + 1
            event = 1 if (ev_hr is not None and start == ev_hr) else 0
            rows.append({"id": gpu, "start": start, "stop": stop, "event": event, "peer_z": float(r["peer_z"])})
            if event:
                break
    cp = pd.DataFrame(rows)
    out = {"n_units": cp["id"].nunique(), "n_events": int(cp["event"].sum())}
    try:
        ctv = CoxTimeVaryingFitter(penalizer=0.1)
        ctv.fit(cp, id_col="id", event_col="event", start_col="start", stop_col="stop")
        hr = float(np.exp(ctv.params_["peer_z"]))
        out["cox_timevarying"] = {
            "hazard_ratio_per_unit_peer_z": round(hr, 3),
            "coef": round(float(ctv.params_["peer_z"]), 4),
            "p_value": round(float(ctv.summary.loc["peer_z", "p"]), 5),
            "interpretation": "HR>1 means higher peer-z raises the hazard of the degradation event",
        }
    except Exception as e:
        out["cox_timevarying"] = {"error": str(e)}
    try:
        # per-unit time-to-event for a Weibull baseline (faulty=event, others=censored at end)
        per = []
        for gpu, g in h.groupby("gpu"):
            if gpu == fid and event_ts is not None:
                dur = int((event_ts - d["timestamp"].min()) // 3600); ev = 1
            else:
                dur = int(end); ev = 0
            per.append({"dur": max(dur, 1), "event": ev})
        pe = pd.DataFrame(per)
        wf = WeibullFitter().fit(pe["dur"], pe["event"])
        out["weibull"] = {"lambda_hr": round(float(wf.lambda_), 2), "rho_shape": round(float(wf.rho_), 3)}
    except Exception as e:
        out["weibull"] = {"error": str(e)}
    return out


def run_scenario(name: str, csv: Path, gt_csv: Path) -> dict:
    df = peer_z(load(csv))
    gt = pd.read_csv(gt_csv)
    fid, col = faulty_gpu(df, gt)
    evt = event_times(gt, col, SEVERITIES)
    lead = lead_time_eval(df, fid, evt)
    surv = survival_models(df, fid, evt.get(1.15))   # use the +15% "actionable" event for survival
    peak_z = round(float(df[df["gpu"] == fid]["peer_z"].max()), 1)
    return {"scenario": name, "faulty_gpu": fid, "fault_column": col,
            "event_ts": {str(k): v for k, v in evt.items()}, "peak_peer_z_faulty": peak_z,
            "lead_time": lead, "survival": surv}


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/amogh/thermalos-vault/raw/experiments/princeton_della_2026_06_11/synthetic")
    out_dir = Path(__file__).parent / "out"; out_dir.mkdir(exist_ok=True)
    results = []
    for name, stem in [("gradual_TIM_dryout", "scenario_gradual"), ("step_cooling_event", "scenario_step")]:
        results.append(run_scenario(name, base / f"{stem}.csv", base / f"{stem}_groundtruth.csv"))
    (out_dir / "leadtime_results.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"\n=== {r['scenario']} (faulty {r['faulty_gpu']}, peak peer-z {r['peak_peer_z_faulty']}) ===")
        print(f"  false-alarm rate on healthy GPUs: {r['lead_time']['false_alarm_rate']}")
        print(f"  lead time (hrs) by event severity: {r['lead_time']['lead_time_hours']}")
        cox = r["survival"].get("cox_timevarying", {})
        print(f"  Cox time-varying: {cox}")
    print(f"\nwrote {out_dir/'leadtime_results.json'}")


if __name__ == "__main__":
    main()
