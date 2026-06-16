"""
Tests for the per-job R_θ report (theta/agent/jobreport.py) — the jobstats path.

Builds synthetic Prometheus query_range exports (the exact shape jobstats produces:
nvidia_gpu_temperature_celsius / power_usage_milliwatts / duty_cycle, labelled by
instance + ordinal) and checks the pipeline aligns, segments steady-load, computes
R_θ, and flags a degraded unit with the two-tier (flagged / watch) output.
"""
import json

from theta.agent.jobreport import (
    load_exports, steady_rtheta, build_report, _metric_kind, _align,
)


def _series(name, instance, ordinal, values):
    return {"metric": {"__name__": name, "instance": instance, "ordinal": str(ordinal)},
            "values": [[t, str(v)] for t, v in values]}


def _export(name, rows):
    """rows: list of (instance, ordinal, [(ts,val)...])."""
    return {"status": "success",
            "data": {"resultType": "matrix",
                     "result": [_series(name, i, o, vals) for i, o, vals in rows]}}


def _synthetic_job(tmp_path, n_nodes=4, n_ord=8, samples=120, degraded=("n2", 3)):
    """8-GPU-per-node fleet at ~650 W with HGX position structure + one degraded GPU."""
    pos = {0: 6, 1: -3, 2: -7, 3: 3, 4: 7, 5: -6, 6: 5, 7: -4}  # °C offsets ~ ±11% R_θ
    temp_rows, pow_rows, util_rows = [], [], []
    for n in range(n_nodes):
        node = f"n{n}"
        inst = f"della-{node}:9445"
        for o in range(n_ord):
            base_t = 60 + pos[o] + n * 0.3
            if (node, o) == degraded:
                base_t += 20                      # degraded: much hotter at same power
            temps = [(1000 + 15 * k, base_t + (k % 3)) for k in range(samples)]
            pows  = [(1000 + 15 * k, 650_000) for k in range(samples)]   # milliwatts
            utils = [(1000 + 15 * k, 95) for k in range(samples)]
            temp_rows.append((inst, o, temps))
            pow_rows.append((inst, o, pows))
            util_rows.append((inst, o, utils))
    p = tmp_path
    (p / "t.json").write_text(json.dumps(_export("nvidia_gpu_temperature_celsius", temp_rows)))
    (p / "p.json").write_text(json.dumps(_export("nvidia_gpu_power_usage_milliwatts", pow_rows)))
    (p / "u.json").write_text(json.dumps(_export("nvidia_gpu_duty_cycle", util_rows)))
    return [p / "t.json", p / "p.json", p / "u.json"]


def test_metric_kind_classifies_jobstats_names():
    assert _metric_kind("nvidia_gpu_temperature_celsius") == "T"
    assert _metric_kind("nvidia_gpu_power_usage_milliwatts") == "P"
    assert _metric_kind("nvidia_gpu_duty_cycle") == "U"
    assert _metric_kind("nvidia_gpu_memory_used") is None


def test_align_converts_power_mw_to_w_and_intersects():
    series = {("n0", 0): {"T": {1: 70.0, 2: 71.0}, "P": {1: 650_000.0, 2: 650_000.0},
                          "U": {1: 95.0, 2: 95.0, 3: 95.0}}}
    out = _align(series)
    assert out[("n0", 0)]["P"] == [650.0, 650.0]   # mW → W
    assert out[("n0", 0)]["t"] == [1, 2]           # only shared timestamps


def test_full_pipeline_flags_degraded_unit(tmp_path):
    files = _synthetic_job(tmp_path, degraded=("n2", 3))
    aligned = load_exports(files)
    assert len(aligned) == 32                       # 4 nodes × 8
    stats = steady_rtheta(aligned)
    assert len(stats) == 32
    rep = build_report("testjob", stats)
    assert rep.method == "median_polish"
    assert "n2:3" in rep.flagged                     # the degraded GPU
    assert all(k == "n2:3" for k in rep.flagged)     # nobody else flagged
    assert rep.fleet_mean_r is not None


def test_warmup_and_load_filter_exclude_idle(tmp_path):
    # A GPU that is idle (low power) the whole job yields no steady-load samples.
    inst = "della-n0:9445"
    temp = _export("nvidia_gpu_temperature_celsius",
                   [(inst, 0, [(1000 + 15 * k, 35) for k in range(120)])])
    powr = _export("nvidia_gpu_power_usage_milliwatts",
                   [(inst, 0, [(1000 + 15 * k, 40_000) for k in range(120)])])   # 40 W idle
    util = _export("nvidia_gpu_duty_cycle",
                   [(inst, 0, [(1000 + 15 * k, 0) for k in range(120)])])
    (tmp_path / "t.json").write_text(json.dumps(temp))
    (tmp_path / "p.json").write_text(json.dumps(powr))
    (tmp_path / "u.json").write_text(json.dumps(util))
    aligned = load_exports([tmp_path / "t.json", tmp_path / "p.json", tmp_path / "u.json"])
    stats = steady_rtheta(aligned)
    assert stats == []                               # idle → no steady-load R_θ


def test_single_node_uses_within_node_method(tmp_path):
    files = _synthetic_job(tmp_path, n_nodes=1, degraded=("n0", 3))
    rep = build_report("solo", steady_rtheta(load_exports(files)))
    assert rep.method == "within_node"
    assert any("single node" in n for n in rep.notes)


def test_watch_tier_catches_marginal(tmp_path):
    # Degrade a unit only mildly (small temp bump) → should land on watch, not flagged.
    pos = {0: 6, 1: -3, 2: -7, 3: 3, 4: 7, 5: -6, 6: 5, 7: -4}
    temp_rows, pow_rows, util_rows = [], [], []
    for n in range(4):
        inst = f"della-n{n}:9445"
        for o in range(8):
            t = 60 + pos[o] + (5 if (n, o) == (2, 1) else 0)   # mild bump on n2:1
            temp_rows.append((inst, o, [(1000 + 15 * k, t) for k in range(120)]))
            pow_rows.append((inst, o, [(1000 + 15 * k, 650_000) for k in range(120)]))
            util_rows.append((inst, o, [(1000 + 15 * k, 95) for k in range(120)]))
    (tmp_path / "t.json").write_text(json.dumps(_export("nvidia_gpu_temperature_celsius", temp_rows)))
    (tmp_path / "p.json").write_text(json.dumps(_export("nvidia_gpu_power_usage_milliwatts", pow_rows)))
    (tmp_path / "u.json").write_text(json.dumps(_export("nvidia_gpu_duty_cycle", util_rows)))
    rep = build_report("j", steady_rtheta(load_exports(
        [tmp_path / "t.json", tmp_path / "p.json", tmp_path / "u.json"])))
    # mild degradation surfaces somewhere (watch or flagged), not silently dropped
    assert "n2:1" in rep.watch or "n2:1" in rep.flagged
