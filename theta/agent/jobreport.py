"""
Per-job R_θ report card — the SLURM/jobstats integration.

jobstats (Princeton, Ohio Supercomputer Center, and a growing list of HPC sites)
already scrapes per-GPU NVML telemetry into Prometheus, labelled by SLURM `jobid`,
node `instance`, and HGX `ordinal`. It reports utilization/efficiency but NOT
cooling health — it has the raw temp and power, it just never divides them. Theta
does: given a job ID, pull that job's GPU telemetry from the SAME Prometheus and
produce a per-job R_θ report card that flags cooling-degraded units.

This is the E009 analysis productized as a live, on-demand surface. It runs on the
exact metric set jobstats collects, so it drops into an existing jobstats site with
no new agent and no new telemetry:
    nvidia_gpu_temperature_celsius        (°C)
    nvidia_gpu_power_usage_milliwatts     (mW → W)
    nvidia_gpu_duty_cycle                 (utilization %)

Two input modes (CLI `theta report`):
  * live   — query_range against a Prometheus endpoint for a job ID
  * export — read saved Prometheus query_range JSON (the format the Princeton
             study used) — offline, and the validation fixture for this code.

Method mirrors the validated E009 pipeline (raw/code/princeton_della_analysis.py):
align the three series per node:ordinal, drop warm-up, take steady-LOAD samples,
compute R_θ = (T − T_ref)/P, then run peer-relative + (multi-node) median polish.
T_ref defaults to an assumed ambient; all *detection* is peer-relative so T_ref
cancels and the absolute assumption only affects displayed magnitudes.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .peer import PeerRelativeDetector, median_polish_z, SUSTAINED

DEFAULT_AMBIENT_C = 25.0
WARMUP_FRAC       = 0.15   # drop the first 15% of the window (thermal ramp)
LOAD_FRAC         = 0.60   # "loaded" = power above 60% of the GPU's p90 power
MIN_STEADY        = 30     # minimum steady samples before a GPU is scored


@dataclass
class GpuJobStat:
    node:     str
    ordinal:  int
    r_mean:   float
    r_std:    float
    t_mean:   float
    p_mean:   float
    n:        int

    @property
    def key(self) -> str:
        return f"{self.node}:{self.ordinal}"


@dataclass
class JobReport:
    jobid:    str
    gpus:     list[GpuJobStat]
    nodes:    list[str]
    flagged:  dict[str, float]              # gpu key → robust-z (z ≥ z_thresh: act)
    watch:    dict[str, float]              # gpu key → robust-z (elevated, sub-threshold)
    method:   str                           # "median_polish" | "within_node" | "none"
    fleet_mean_r: Optional[float] = None
    notes:    list[str] = field(default_factory=list)


# ── loading ───────────────────────────────────────────────────────────────────

def _metric_kind(name: str) -> Optional[str]:
    n = name.lower()
    if "temperature" in n:
        return "T"
    if "power" in n:
        return "P"
    if "duty_cycle" in n or "util" in n:
        return "U"
    return None


def _ingest_result_block(result: list, series: dict, jobid: Optional[str] = None) -> None:
    """Fold one Prometheus `result` array into series[(node,ordinal)][kind].

    If `jobid` is given, only ingest series whose `jobid` label matches — so a
    Prometheus export dir holding several jobs (the common case) is isolated to
    the requested job instead of merging jobs into one (wrong) comparison.
    """
    for s in result:
        m = s.get("metric", {})
        if jobid is not None and "jobid" in m and str(m.get("jobid")) != str(jobid):
            continue
        kind = _metric_kind(m.get("__name__", ""))
        if kind is None:
            continue
        inst = m.get("instance", "?").split(":")[0].replace("della-", "")
        try:
            ordn = int(m.get("ordinal", m.get("minor_number", -1)))
        except (TypeError, ValueError):
            continue
        if ordn < 0:
            continue
        d = series.setdefault((inst, ordn), {"T": {}, "P": {}, "U": {}})
        for ts, val in s.get("values", []):
            try:
                d[kind][int(float(ts))] = float(val)
            except (TypeError, ValueError):
                continue


def load_exports(paths: list[Path], jobid: Optional[str] = None) -> dict:
    """Load saved Prometheus query_range JSON into aligned per-GPU series.

    When `jobid` is set, filter to that job (export dirs often hold many jobs). If
    the filter matches nothing (e.g. an old export with no jobid label), fall back
    to ingesting everything so the tool still works on label-less fixtures.
    """
    def _load(jid: Optional[str]) -> dict:
        series: dict = {}
        for p in paths:
            doc = json.loads(Path(p).read_text())
            result = doc.get("data", {}).get("result", doc if isinstance(doc, list) else [])
            _ingest_result_block(result, series, jobid=jid)
        return _align(series)

    aligned = _load(jobid)
    if jobid is not None and not aligned:
        aligned = _load(None)   # no jobid label in export → use all series
    return aligned


def load_prometheus(prom_url: str, jobid: str, start: float, end: float,
                    step: str = "30s", timeout: float = 30.0) -> dict:
    """Live: query_range the three metrics for a job ID. Requires httpx."""
    import httpx
    series: dict = {}
    metrics = ["nvidia_gpu_temperature_celsius",
               "nvidia_gpu_power_usage_milliwatts",
               "nvidia_gpu_duty_cycle"]
    base = prom_url.rstrip("/") + "/api/v1/query_range"
    with httpx.Client(timeout=timeout) as client:
        for metric in metrics:
            q = f'{metric}{{jobid="{jobid}"}}'
            r = client.get(base, params={"query": q, "start": start,
                                         "end": end, "step": step})
            r.raise_for_status()
            _ingest_result_block(r.json().get("data", {}).get("result", []), series)
    return _align(series)


_CSV_DEFAULT_COLS = {
    "timestamp": ("timestamp", "time", "ts", "datetime"),
    "node":      ("node", "hostname", "host", "instance"),
    "gpu":       ("gpu", "gpu_index", "gpu_uuid", "uuid", "minor_number", "ordinal", "index"),
    "temp":      ("temp", "temperature", "gpu_temp", "temperature_celsius", "nvidia_gpu_temperature_celsius"),
    "power":     ("power", "power_w", "power_watts", "power_usage", "nvidia_gpu_power_usage_milliwatts"),
    "util":      ("util", "utilization", "duty_cycle", "gpu_util", "nvidia_gpu_duty_cycle"),
    "jobid":     ("jobid", "job_id", "job", "job_label"),
}


def _resolve_columns(header: list[str], overrides: Optional[dict] = None) -> dict:
    """Map logical fields → actual CSV column names (case-insensitive, with overrides)."""
    lower = {h.lower().strip(): h for h in header}
    resolved: dict = {}
    for field_name, candidates in _CSV_DEFAULT_COLS.items():
        if overrides and field_name in overrides:
            resolved[field_name] = overrides[field_name]
            continue
        for c in candidates:
            if c in lower:
                resolved[field_name] = lower[c]
                break
    return resolved


def load_csv(paths: list[Path], jobid: Optional[str] = None,
             columns: Optional[dict] = None, power_unit: str = "auto") -> dict:
    """
    Load a per-GPU telemetry CSV into the same aligned shape as `load_exports`.

    Flexible columns (case-insensitive, override via `columns`): timestamp, node,
    gpu (index or UUID), temp °C, power, util %, optional jobid. Non-integer GPU
    ids (UUIDs) are assigned stable per-node ordinals in first-seen order so the
    position-conditioned comparison still works.

    power_unit: "W", "mW", or "auto" (guess from magnitude — values >5000 ⇒ mW).
    """
    import csv

    series: dict = {}
    uuid_ord: dict[tuple[str, str], int] = {}   # (node, raw_gpu) → ordinal
    node_next_ord: dict[str, int] = {}

    for p in paths:
        with Path(p).open(newline="") as fh:
            reader = csv.DictReader(fh)
            cols = _resolve_columns(reader.fieldnames or [], columns)
            missing = [f for f in ("node", "gpu", "temp", "power") if f not in cols]
            if missing:
                raise ValueError(f"CSV {p} missing required column(s) {missing}; "
                                 f"saw {reader.fieldnames}. Pass --columns to map them.")
            for row in reader:
                if jobid is not None and "jobid" in cols:
                    if str(row.get(cols["jobid"], "")) != str(jobid):
                        continue
                node = str(row[cols["node"]]).split(":")[0].replace("della-", "").strip()
                raw_gpu = str(row[cols["gpu"]]).strip()
                try:
                    ordn = int(raw_gpu)
                except ValueError:
                    key = (node, raw_gpu)
                    if key not in uuid_ord:
                        uuid_ord[key] = node_next_ord.get(node, 0)
                        node_next_ord[node] = node_next_ord.get(node, 0) + 1
                    ordn = uuid_ord[key]
                try:
                    ts = int(float(row[cols["timestamp"]])) if "timestamp" in cols else len(
                        series.get((node, ordn), {}).get("T", {}))
                    temp = float(row[cols["temp"]])
                    power = float(row[cols["power"]])
                    util = float(row[cols["util"]]) if "util" in cols else 100.0
                except (TypeError, ValueError):
                    continue
                # Normalize power to milliwatts (so _align's mW→W applies uniformly).
                if power_unit == "mW" or (power_unit == "auto" and power > 5000):
                    pass  # already mW
                else:
                    power *= 1000.0
                d = series.setdefault((node, ordn), {"T": {}, "P": {}, "U": {}})
                d["T"][ts] = temp
                d["P"][ts] = power
                d["U"][ts] = util
    return _align(series)


def _align(series: dict) -> dict:
    """Intersect timestamps across T/P/U per GPU; convert power mW→W."""
    out = {}
    for key, d in series.items():
        ts = sorted(set(d["T"]) & set(d["P"]) & set(d["U"]))
        if not ts:
            continue
        out[key] = {
            "t": ts,
            "T": [d["T"][x] for x in ts],
            "P": [d["P"][x] / 1000.0 for x in ts],   # milliwatts → watts
            "U": [d["U"][x] for x in ts],
        }
    return out


# ── analysis ────────────────────────────────────────────────────────────────--

def steady_rtheta(aligned: dict, ambient: float = DEFAULT_AMBIENT_C) -> list[GpuJobStat]:
    """Per-GPU steady-LOAD R_θ, mirroring the validated E009 segmentation."""
    stats: list[GpuJobStat] = []
    for (node, ordn), d in aligned.items():
        n = len(d["t"])
        if n < MIN_STEADY:
            continue
        warm_start = int(n * WARMUP_FRAC)
        powers = d["P"][warm_start:]
        if not powers:
            continue
        p90 = sorted(powers)[max(0, int(len(powers) * 0.9) - 1)]
        load_thresh = max(LOAD_FRAC * p90, 150.0)   # floor avoids idle/near-idle
        rs, ts, ps = [], [], []
        for T, P in zip(d["T"][warm_start:], powers):
            if P > load_thresh and P > 0:
                rs.append((T - ambient) / P)
                ts.append(T)
                ps.append(P)
        if len(rs) < MIN_STEADY:
            continue
        stats.append(GpuJobStat(
            node=node, ordinal=ordn,
            r_mean=statistics.fmean(rs),
            r_std=statistics.pstdev(rs) if len(rs) > 1 else 0.0,
            t_mean=statistics.fmean(ts), p_mean=statistics.fmean(ps), n=len(rs),
        ))
    return stats


def build_report(jobid: str, stats: list[GpuJobStat],
                 z_thresh: float = 3.0, watch_thresh: float = 2.5) -> JobReport:
    """Run peer-relative (within-node) + median polish (multi-node) over the job.

    Two tiers: `flagged` (z ≥ z_thresh — act now) and `watch` (watch_thresh ≤ z <
    z_thresh — elevated, keep an eye on it). The watch tier is where genuinely
    marginal degradation lives (on the real Princeton job, j13g2:2 — a confirmed
    E009 flag at the margin — lands here at z≈2.85 rather than being lost).
    """
    nodes = sorted({s.node for s in stats})
    notes: list[str] = []
    flagged: dict[str, float] = {}
    watch:   dict[str, float] = {}
    method = "none"

    if not stats:
        notes.append("no steady-load GPU samples found for this job")
        return JobReport(jobid, stats, nodes, flagged, watch, method)

    fleet_mean = statistics.fmean([s.r_mean for s in stats])

    if len(nodes) >= 2:
        # Position-conditioned median polish — the full E009 fleet method.
        fleet = {s.key: (s.node, s.ordinal, s.r_mean) for s in stats}
        zmap = median_polish_z(fleet)
        method = "median_polish"
    else:
        # Single node — within-node peer-relative (matched power).
        snap = {s.ordinal: (s.p_mean, s.r_mean) for s in stats}
        det = PeerRelativeDetector()
        res = {}
        for _ in range(SUSTAINED):
            res = det.evaluate(snap, 0.0)
        node = nodes[0]
        zmap = {f"{node}:{o}": (r.robust_z or 0.0) for o, r in res.items()}
        method = "within_node"
        notes.append("single node — within-node peer detection; position-conditioned "
                     "median polish needs ≥2 nodes (it would catch position-masked units)")

    for k, zz in zmap.items():
        if zz >= z_thresh:
            flagged[k] = zz
        elif zz >= watch_thresh:
            watch[k] = zz

    return JobReport(jobid, stats, nodes, flagged, watch, method,
                     fleet_mean_r=fleet_mean, notes=notes)
