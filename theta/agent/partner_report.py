"""
Partner-export report — the deliverable Theta sends back to a research-computing
collaborator (GWDG/Grete, Princeton/Della, ...) from one telemetry export.

Pipeline: ingest (CSV or Prometheus JSON, via jobreport) → per-GPU steady-load
R_θ → peer-relative + position-conditioned detection → **signature-matrix cause
attribution** → formatted per-GPU breakdown.

The α/β decomposition here is fully peer-relative and hardware-agnostic, so it
runs on any partner's GPUs without a per-model reference: fit each GPU's
T-vs-P line over the job (the e009b method), then z-score its intercept and
slope against the fleet's own distribution. Intercept-elevated = offset fault
(dust/airflow); slope-elevated = conduction fault (TIM/contact). This is what
lets attribution reach EXACT on real multi-power job data, where a single
steady-load summary could only reach subsystem-level.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .jobreport import GpuJobStat, JobReport, steady_rtheta, build_report
from .rtheta_curve import CurveDecomp
from .signature import FeatureVector, classify, SignatureVerdict

# A GPU needs this much power range over the job for a trustworthy T-vs-P slope.
MIN_POWER_SPAN_W = 80.0
MIN_FIT_SAMPLES  = 30
MIN_FLEET_FOR_Z  = 4      # need enough peers to define a robust fleet distribution


def _lstsq(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares (intercept, slope) for y = a + b·x. Slope 0 if degenerate."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-9:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def _robust_sigma(vals: list[float]) -> float:
    med = statistics.median(vals)
    mad = statistics.median([abs(v - med) for v in vals]) or 1e-9
    return 1.4826 * mad


def fleet_alpha_beta(aligned: dict) -> dict[str, CurveDecomp]:
    """
    Per-GPU α/β from each unit's T-vs-P fit, z-scored against the fleet.

    Returns {node:ord → CurveDecomp(alpha_z, beta_z, ...)} only for GPUs whose
    job spanned enough power to fit a slope. α_z = intercept deviation (offset),
    β_z = slope deviation (conduction), both relative to the fleet's own
    distribution — no hardware reference needed.
    """
    fits: dict[str, tuple[float, float, float]] = {}   # key → (intercept, slope, span)
    for (node, ordn), d in aligned.items():
        Ts, Ps = d["T"], d["P"]
        if len(Ts) < MIN_FIT_SAMPLES:
            continue
        span = max(Ps) - min(Ps)
        if span < MIN_POWER_SPAN_W:
            continue
        a, b = _lstsq(Ps, Ts)
        fits[f"{node}:{ordn}"] = (a, b, span)

    if len(fits) < MIN_FLEET_FOR_Z:
        return {}

    intercepts = [v[0] for v in fits.values()]
    slopes = [v[1] for v in fits.values()]
    med_a, sig_a = statistics.median(intercepts), _robust_sigma(intercepts)
    med_b, sig_b = statistics.median(slopes), _robust_sigma(slopes)

    out: dict[str, CurveDecomp] = {}
    for key, (a, b, span) in fits.items():
        out[key] = CurveDecomp(
            alpha_z=(a - med_a) / sig_a,
            beta_z=(b - med_b) / sig_b,
            power_span_w=span, low_power_w=0.0, high_power_w=span,
        )
    return out


@dataclass
class UnitAttribution:
    key:      str
    robust_z: float
    tier:     str            # FLAG | WATCH
    verdict:  SignatureVerdict
    stat:     GpuJobStat


def attribute(report: JobReport, aligned: dict) -> list[UnitAttribution]:
    """Run the signature classifier on every flagged/watch unit in the report."""
    ab = fleet_alpha_beta(aligned)
    stat_by_key = {s.key: s for s in report.gpus}
    multi_node = len(report.nodes) >= 2
    out: list[UnitAttribution] = []

    tiers = [("FLAG", report.flagged), ("WATCH", report.watch)]
    for tier, units in tiers:
        for key, z in sorted(units.items(), key=lambda kv: -kv[1]):
            curve = ab.get(key)
            fv = FeatureVector(
                rtheta_overall_z=z,
                power_range_observed=curve is not None,
                alpha_z=curve.alpha_z if curve else None,
                beta_z=curve.beta_z if curve else None,
                locality="single" if multi_node else "node",
            )
            out.append(UnitAttribution(key, z, tier, classify(fv),
                                       stat_by_key.get(key)))
    return out


def format_report(report: JobReport, attributions: list[UnitAttribution],
                  label: str) -> str:
    """Human-readable partner deliverable."""
    L: list[str] = []
    L.append(f"Theta R_θ analysis — {label}")
    L.append("=" * 66)
    L.append(f"GPUs analyzed : {len(report.gpus)} across {len(report.nodes)} node(s)")
    if report.fleet_mean_r is not None:
        L.append(f"Fleet mean R_θ: {report.fleet_mean_r:.4f} C/W")
    L.append(f"Detection     : {report.method}")
    for note in report.notes:
        L.append(f"  note: {note}")

    if not attributions:
        L.append("\nNo units exceeded the watch threshold — fleet looks healthy.")
        return "\n".join(L)

    L.append(f"\nFlagged/watch units: {len(attributions)}")
    for a in attributions:
        v = a.verdict
        if v.identifiable:
            cause = f"{v.headline_cause.value} [EXACT]"
        elif v.discriminated:
            cause = f"{v.headline_cause.value} [cause-class]"
        else:
            cause = f"thermal degradation [subsystem-level, lean: {v.headline_cause.value}]"
        temp = f"{a.stat.t_mean:.0f}°C, {a.stat.p_mean:.0f} W" if a.stat else "—"
        L.append(f"\n  [{a.tier}] {a.key}  (z {a.robust_z:+.1f}; {temp})")
        L.append(f"     cause     : {cause}")
        if v.top and v.top.supporting:
            L.append(f"     supporting: {'; '.join(v.top.supporting)}")
        for ax in v.missing_axes:
            L.append(f"     missing   : {ax.needs} (via {ax.via})")
    L.append("\nDetection is peer-relative (T_ref cancels); attribution α/β is "
             "fleet-relative, so both are invariant to the absolute ambient assumption.")
    return "\n".join(L)


def analyze(aligned: dict, label: str = "partner export",
            z_thresh: float = 3.0, watch_thresh: float = 2.5) -> tuple[JobReport, list[UnitAttribution], str]:
    """End-to-end: aligned series → report + attribution + formatted text."""
    stats = steady_rtheta(aligned)
    report = build_report(label, stats, z_thresh=z_thresh, watch_thresh=watch_thresh)
    attributions = attribute(report, aligned)
    return report, attributions, format_report(report, attributions, label)
