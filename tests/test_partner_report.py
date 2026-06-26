"""
Tests for the partner-export pipeline — CSV ingestion, fleet-relative α/β, and
end-to-end attribution. The payoff test: a multi-power job export resolves a
conduction fault to EXACT, where a steady-load summary could only reach
subsystem-level.
"""

import csv

from theta.agent.jobreport import load_csv
from theta.agent.partner_report import fleet_alpha_beta, analyze
from theta.agent.fault_classifier import FaultCause


def _write_csv(path, rows, header=("timestamp", "node", "gpu", "temp", "power", "util")):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _job_rows(node, gpu, n=60, t0=0, temp_at=lambda p: 25 + 0.05 * p,
              powers=None):
    """Per-GPU rows sweeping power so T-vs-P is fittable."""
    powers = powers or ([300] * (n // 2) + [600] * (n - n // 2))
    return [(t0 + i * 30, node, gpu, round(temp_at(p), 2), p, 90)
            for i, p in enumerate(powers)]


def test_csv_ingestion_aligns_series(tmp_path):
    p = tmp_path / "job.csv"
    rows = _job_rows("n1", 0) + _job_rows("n1", 1)
    _write_csv(p, rows)
    aligned = load_csv([p])
    assert ("n1", 0) in aligned and ("n1", 1) in aligned
    # power normalized to watts (the CSV gave watts; auto-detect kept them)
    assert max(aligned[("n1", 0)]["P"]) <= 700


def test_csv_uuid_gpus_get_stable_ordinals(tmp_path):
    p = tmp_path / "job.csv"
    rows = _job_rows("n1", "GPU-aaa") + _job_rows("n1", "GPU-bbb")
    _write_csv(p, rows)
    aligned = load_csv([p])
    # two distinct UUIDs on one node → ordinals 0 and 1
    assert {k[1] for k in aligned} == {0, 1}


def test_csv_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv(p, [(0, "n1", 0, 50)], header=("timestamp", "node", "gpu", "temp"))
    try:
        load_csv([p])
        assert False, "expected ValueError for missing power column"
    except ValueError as e:
        assert "power" in str(e)


def test_fleet_alpha_beta_flags_conduction_outlier(tmp_path):
    # 7 healthy GPUs tracking a normal T-vs-P; 1 with a steeper slope (conduction).
    p = tmp_path / "fleet.csv"
    rows = []
    for g in range(7):
        rows += _job_rows("n1", g, temp_at=lambda p: 25 + 0.05 * p)
    # GPU7: same low-power temp, much hotter at high power → steeper slope
    rows += _job_rows("n1", 7, temp_at=lambda p: 25 + 0.075 * p)
    _write_csv(p, rows)
    ab = load_csv([p])
    decomp = fleet_alpha_beta(ab)
    assert decomp["n1:7"].beta_z > 2.0          # slope outlier
    assert abs(decomp["n1:0"].beta_z) < 2.0     # healthy peer


def test_end_to_end_conduction_reaches_exact(tmp_path):
    # Two nodes (so median-polish runs), one GPU with a conduction signature.
    p = tmp_path / "job.csv"
    rows = []
    for node in ("n1", "n2"):
        for g in range(8):
            slope = 0.075 if (node, g) == ("n1", 7) else 0.05
            rows += _job_rows(node, g, temp_at=lambda pw, s=slope: 25 + s * pw)
    _write_csv(p, rows)
    aligned = load_csv([p])
    report, attributions, text = analyze(aligned)
    # the conduction unit should flag and attribute to TIM, discriminated by slope
    by_key = {a.key: a for a in attributions}
    assert "n1:7" in by_key
    v = by_key["n1:7"].verdict
    assert v.headline_cause is FaultCause.TIM_DEGRADATION
    assert v.discriminated
    assert "Theta R_θ analysis" in text


def test_offset_fault_attributes_to_offset_family(tmp_path):
    # One GPU uniformly hotter at all power (offset) across two nodes.
    p = tmp_path / "job.csv"
    rows = []
    for node in ("n1", "n2"):
        for g in range(8):
            off = 8.0 if (node, g) == ("n1", 3) else 0.0
            rows += _job_rows(node, g, temp_at=lambda pw, o=off: 25 + 0.05 * pw + o)
    _write_csv(p, rows)
    aligned = load_csv([p])
    report, attributions, _ = analyze(aligned)
    by_key = {a.key: a for a in attributions}
    assert "n1:3" in by_key
    v = by_key["n1:3"].verdict
    # offset → dust/airflow family, NOT conduction
    assert v.headline_cause in (FaultCause.DUST_ACCUMULATION, FaultCause.AIRFLOW_BLOCKAGE)
