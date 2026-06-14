"""
Characterization tests for the peer-relative detector (theta/agent/peer.py).

The headline test reproduces E009: a single degraded GPU among matched-power
node-mates is blind-flagged with NO temporal history — the capability the
temporal DriftDetector structurally cannot provide. The remainder pin the
guard rails that keep the false-positive budget tight (the thing that makes
this safe to ship as a fleet default).
"""

from theta.agent.peer import (
    PeerRelativeDetector, _power_matched, MIN_GROUP, SUSTAINED,
)


def _drive(det, snapshot, cycles, t0=1000.0):
    """Run `cycles` evaluations on the same snapshot; return the last result."""
    out = {}
    for i in range(cycles):
        out = det.evaluate(snapshot, t0 + i)
    return out


def test_e009_reproduction_blind_flags_degraded_unit():
    # 8-GPU H100 node, all at ~650 W (matched load, the E009 incident job).
    # Healthy same-model node-mates under matched load cluster tightly (~1 °C)
    # → R_θ ≈ 0.055 C/W. The degraded unit ran 80 °C @ 653 W → R_θ ≈ 0.084
    # (+50%). On this tight, low-R_θ fleet the relative floor is the binding
    # scale (a T4-scale absolute floor would have MISSED it); it recovers the
    # E009-scale robust-z (the real unit measured +15.6).
    healthy = [(650.0, 0.0540 + 0.0004 * i) for i in range(7)]  # 0.0540..0.0564
    snapshot = {g: healthy[g] for g in range(7)}
    snapshot[7] = (653.0, 0.084)                                # degraded

    det = PeerRelativeDetector()
    res = _drive(det, snapshot, SUSTAINED)

    deg = res[7]
    assert deg.is_anomaly, "degraded unit should be flagged peer-relative"
    assert deg.robust_z > 8.0, f"expected an E009-scale robust-z, got {deg.robust_z}"
    assert deg.n_peers >= MIN_GROUP - 1
    # No healthy unit should be flagged.
    assert not any(res[g].is_anomaly for g in range(7))


def test_uniform_fleet_produces_no_anomaly():
    # A genuinely uniform fleet must stay silent indefinitely.
    snapshot = {g: (400.0, 0.30 + 0.002 * g) for g in range(8)}
    det = PeerRelativeDetector()
    res = _drive(det, snapshot, SUSTAINED * 3)
    assert not any(r.is_anomaly for r in res.values())
    assert all(abs(r.robust_z) < 4.0 for r in res.values())


def test_below_min_group_never_alarms():
    # Three GPUs (group < MIN_GROUP) with a blatant outlier: no peer group is
    # trustworthy enough, so the detector must say nothing.
    snapshot = {0: (500.0, 0.12), 1: (500.0, 0.12), 2: (500.0, 0.40)}
    det = PeerRelativeDetector()
    res = _drive(det, snapshot, SUSTAINED * 2)
    assert all(not r.is_anomaly for r in res.values())
    assert res[2].robust_z is None  # not enough peers to evaluate


def test_power_conditioning_excludes_unmatched_load():
    # One idle GPU (low power → naturally high R_θ) among loaded peers must NOT
    # be flagged: it has no matched-power peer group, so it is not compared
    # against the loaded cohort. This is the R_θ(P)-is-a-curve guard.
    snapshot = {g: (600.0, 0.12 + 0.003 * g) for g in range(6)}
    snapshot[6] = (40.0, 1.25)   # idle: high R_θ, but unmatched power
    det = PeerRelativeDetector()
    res = _drive(det, snapshot, SUSTAINED * 2)
    assert res[6].robust_z is None         # no matched-power peers → not judged
    assert not any(r.is_anomaly for r in res.values())


def test_single_cycle_spike_does_not_alert():
    base = {g: (650.0, 0.12) for g in range(7)}
    det = PeerRelativeDetector()
    # One spiking cycle for GPU 7…
    spike = dict(base); spike[7] = (650.0, 0.30)
    r1 = det.evaluate(spike, 1000.0)
    assert not r1[7].is_anomaly, "a single anomalous cycle must not alert"
    # …then it returns to normal; sustained counter decays, no alert.
    ok = dict(base); ok[7] = (650.0, 0.121)
    r2 = det.evaluate(ok, 1005.0)
    assert not r2[7].is_anomaly


def test_sustained_outlier_escalates_to_critical():
    snapshot = {g: (650.0, 0.12) for g in range(7)}
    snapshot[7] = (650.0, 0.40)   # ~+230%, way past Z_CRITICAL
    det = PeerRelativeDetector()
    res = _drive(det, snapshot, SUSTAINED)
    assert res[7].is_critical
    assert res[7].confidence == 1.0


def test_mad_floor_prevents_screaming_z_on_near_uniform_fleet():
    # All GPUs nearly identical (MAD ≈ 0) with one a hair high. Without a scale
    # floor this would divide by ~0 and scream; the floor keeps z sane.
    snapshot = {g: (500.0, 0.2000) for g in range(7)}
    snapshot[7] = (500.0, 0.2030)   # +1.5% — real but trivial
    det = PeerRelativeDetector()
    res = _drive(det, snapshot, SUSTAINED * 2)
    assert not res[7].is_anomaly
    assert abs(res[7].robust_z) < 4.0


def test_power_matched_helper():
    assert _power_matched(600.0, 650.0, 0.15)        # within 15%
    assert not _power_matched(600.0, 800.0, 0.15)    # 33% apart
    assert not _power_matched(600.0, 0.0, 0.15)      # zero/invalid power
