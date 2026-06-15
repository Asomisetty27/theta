"""
Tests for `theta fleet-scan` — the position-conditioned cross-node anomaly CLI.

Covers: the 3/3 detection on a synthetic Della-like fleet (position structure +
a position-masked degraded unit), the single-node guard, and both accepted input
shapes (record list + results.json). No vault data required.
"""
import json

from typer.testing import CliRunner

from theta.cli import app

runner = CliRunner()


def _della_like_fleet():
    """8 nodes × 8 ordinals, real HGX position structure, one masked anomaly."""
    mu = 0.057
    pos = {0: +0.006, 1: -0.003, 2: -0.007, 3: +0.003,
           4: +0.007, 5: -0.006, 6: +0.005, 7: -0.004}
    recs = []
    for n in range(8):
        node_eff = (n - 3.5) * 0.0004
        for o in range(8):
            recs.append({"node": f"n{n}", "ordinal": o,
                         "rtheta": mu + node_eff + pos[o], "power": 650.0})
    # Degrade one GPU in a structurally COOL slot (ordinal 2) — within-node it
    # looks fine; only position correction surfaces it. (The j13g2:2 case.)
    for r in recs:
        if r["node"] == "n5" and r["ordinal"] == 2:
            r["rtheta"] += 0.010
    return recs


def test_fleet_scan_flags_position_masked_unit(tmp_path):
    p = tmp_path / "fleet.json"
    p.write_text(json.dumps(_della_like_fleet()))
    res = runner.invoke(app, ["fleet-scan", str(p)])
    assert res.exit_code == 0, res.output
    assert "n5:2" in res.output           # the masked unit is flagged
    assert "1 unit" in res.output         # exactly one


def test_fleet_scan_uniform_fleet_is_clean(tmp_path):
    mu, pos = 0.060, {o: (o - 3.5) * 0.002 for o in range(8)}
    recs = [{"node": f"n{n}", "ordinal": o, "rtheta": mu + pos[o] + n * 0.0003,
             "power": 600.0} for n in range(6) for o in range(8)]
    p = tmp_path / "uniform.json"
    p.write_text(json.dumps(recs))
    res = runner.invoke(app, ["fleet-scan", str(p)])
    assert res.exit_code == 0, res.output
    assert "No units above" in res.output


def test_fleet_scan_single_node_guard(tmp_path):
    recs = [{"node": "solo", "ordinal": o, "rtheta": 0.06 + 0.001 * o, "power": 650.0}
            for o in range(8)]
    p = tmp_path / "single.json"
    p.write_text(json.dumps(recs))
    res = runner.invoke(app, ["fleet-scan", str(p)])
    assert res.exit_code == 1
    assert "≥2 nodes" in res.output or ">=2 nodes" in res.output


def test_fleet_scan_results_json_shape(tmp_path):
    # The E009 results.json shape: {"steady_bad": {"node:ord": {r_mean, P_mean}}}
    block = {}
    for r in _della_like_fleet():
        block[f'{r["node"]}:{r["ordinal"]}'] = {"r_mean": r["rtheta"], "P_mean": r["power"]}
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"steady_bad": block}))
    res = runner.invoke(app, ["fleet-scan", str(p)])
    assert res.exit_code == 0, res.output
    assert "n5:2" in res.output


def test_fleet_scan_missing_file():
    res = runner.invoke(app, ["fleet-scan", "/nonexistent/path.json"])
    assert res.exit_code == 2
    assert "not found" in res.output
