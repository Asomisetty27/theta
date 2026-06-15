#!/usr/bin/env python3
"""
Validate the agent's peer-relative detectors against the REAL Princeton Della
H100 export (E009) — not the synthetic testbed.

Reads the immutable per-GPU steady-load R_θ that the original E009 analysis
derived from 64 production H100s (results.json), then runs BOTH live detectors:

  1. PeerRelativeDetector  (within-node, single-node-agent scope)
  2. median_polish_z       (position-conditioned, fleet-service scope)

and checks them against E009's three blind-flagged units:
  j13g2:7 (+15.6σ, 80°C) · j12g2:6 (+4.4σ, 72°C) · j13g2:2 (+3.2σ, 61°C)

Expected, honest result:
  * within-node catches 1/3 (the unambiguous j13g2:7), 0 false positives
  * position-conditioned (median polish) catches 3/3, 0 false positives

Run:  python tools/validate_e009_princeton.py [path-to-results.json]
Default path points at the vault's raw export.
"""
import json
import sys
from pathlib import Path

from theta.agent.peer import PeerRelativeDetector, median_polish_z, SUSTAINED

DEFAULT = Path.home() / (
    "thermalos-vault/raw/experiments/princeton_della_2026_06_11/"
    "analysis_out/results.json"
)
TARGETS = {  # E009 blind flags → (median-polish z reported, °C)
    "j13g2:7": (15.6, 80.2),
    "j12g2:6": (4.4, 71.9),
    "j13g2:2": (3.2, 60.7),
}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"FAIL: real Princeton export not found at {path}")
        return 2

    steady = json.load(open(path))["steady_bad"]   # {"node:ord": {r_mean,P_mean,...}}

    # ── 1. within-node PeerRelativeDetector (single-node-agent scope) ─────────
    nodes: dict[str, dict[int, tuple[float, float]]] = {}
    for k, g in steady.items():
        node, ordn = k.split(":")
        nodes.setdefault(node, {})[int(ordn)] = (g["P_mean"], g["r_mean"])

    within_flags: dict[str, float] = {}
    for node, snap in nodes.items():
        det = PeerRelativeDetector()
        res = {}
        for _ in range(SUSTAINED):
            res = det.evaluate(snap, 1000.0)
        for ordn, r in res.items():
            if r.is_anomaly:
                within_flags[f"{node}:{ordn}"] = r.robust_z

    # ── 2. position-conditioned median polish (fleet-service scope) ───────────
    fleet = {
        k: (k.split(":")[0], int(k.split(":")[1]), g["r_mean"])
        for k, g in steady.items()
    }
    z = median_polish_z(fleet)
    polish_flags = {u: zz for u, zz in z.items() if zz > 3.0}

    # ── report ────────────────────────────────────────────────────────────────
    print(f"Real Princeton export: {len(steady)} production H100s, 8 nodes\n")

    print("E009 blind-flagged units — detector reproduction:")
    print(f"  {'unit':<10}{'within-node z':>14}{'polish z':>11}{'E009 z':>9}  status")
    for u, (rep_z, temp) in TARGETS.items():
        wz = within_flags.get(u)
        pz = z.get(u)
        wstr = f"{wz:+.2f}*" if wz is not None else f"{_within_z(u, nodes):+.2f} "
        print(f"  {u:<10}{wstr:>14}{pz:>+11.2f}{rep_z:>+9.1f}  ({temp:.0f}°C)")

    within_hits = set(within_flags) & set(TARGETS)
    polish_hits = set(polish_flags) & set(TARGETS)
    within_fp = set(within_flags) - set(TARGETS)
    polish_fp = set(polish_flags) - set(TARGETS)

    print("\nWithin-node (single-node agent):")
    print(f"  caught {len(within_hits)}/3 {sorted(within_hits)} · false positives: {sorted(within_fp) or 'none'}")
    print("Position-conditioned median polish (fleet service):")
    print(f"  caught {len(polish_hits)}/3 {sorted(polish_hits)} · false positives: {sorted(polish_fp) or 'none'}")

    ok = (
        within_hits == {"j13g2:7"}
        and not within_fp
        and polish_hits == set(TARGETS)
        and not polish_fp
    )
    print("\n" + ("PASS — reproduces E009 exactly (1/3 within-node, 3/3 fleet, 0 FP)"
                  if ok else "MISMATCH — see above"))
    return 0 if ok else 1


def _within_z(uid: str, nodes) -> float:
    """Within-node robust-z for a unit even when not flagged (for the report)."""
    det = PeerRelativeDetector()
    node, ordn = uid.split(":")
    res = {}
    for _ in range(SUSTAINED):
        res = det.evaluate(nodes[node], 1000.0)
    return res[int(ordn)].robust_z or 0.0


if __name__ == "__main__":
    raise SystemExit(main())
