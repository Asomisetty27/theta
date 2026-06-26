# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`runtheta` (package `theta`) — a GPU thermal-power forensics **agent**. The whole product rests on one signal: **`R_θ = (T_junction − T_ref) / P_GPU`** (°C/W). That ratio separates a busy-hot GPU from a *failing*-hot one; no incumbent (DCGM, Mission Control, Phaidra) computes it. `README.md` is the user/product reference (CLI surface, metrics table, install, the competitive case, the science) — read it for *what* the tool does. This file is the *developer* map: architecture, dev commands, and the invariants that aren't obvious from any single file.

## Commands

```bash
pip install -e ".[dev]"        # editable install + pytest/pytest-asyncio
pytest tests/ -v --tb=short    # full suite (CI runs this on py3.10/3.11/3.12)
pytest tests/test_peer.py -v   # single file
pytest tests/test_peer.py::test_name   # single test
ruff check theta/ --select E,F,W --ignore E501   # lint (exactly what CI enforces)
python -m build                # build wheel/sdist (hatchling) → dist/
theta <cmd>                    # the installed CLI entrypoint (theta.cli:app, Typer)
```

There is **no GPU in CI or on a dev laptop.** Tests run against simulated/synthetic telemetry (`tests/sim_ai_factory.py`, `sim/`) and recorded real exports — never assume a live NVML device. CI also skips model training (no Stage 1 CSV in CI) and relies on the hard-coded rule fallback in the classifier.

## Architecture

Everything lives under `theta/`. The data flow (one 5s tick) is a pipeline, and `theta/agent/daemon.py` (the largest file by far, ~67KB) is the spine that wires it together:

```
collector (NVML/amdsmi/DCGM/Redfish via hal.py)
  → raw T, P, util, P-state
  → window.py            steady-state gate (σ < 0.03 C/W) — classification ONLY runs on stable windows
  → R_θ computed         T_ref comes from baseline.py (virtual ambient, learned from the GPU's own idle windows)
  → classifier.py        Decision Tree → {under_load, clean_idle, zombie_recovery, child_exit_recovery}
  → detector.py          temporal drift (rolling baseline + k·σ)     ── needs warm-up
  → peer.py              cross-sectional fleet detection (median-polish robust-z) ── no warm-up
  → governor.py          trust/false-positive budget: holds inferential alerts while warming, circuit-breaks noisy GPUs
  → alerter.py + exporter.py / otlp_exporter.py   stdout / Slack / PagerDuty / Opsgenie / JSONL / Prometheus / OTLP
```

Module groupings inside `theta/agent/`:
- **Hardware abstraction** — `hal.py` is the seam. Vendor collectors plug in behind it: `collector.py` (NVML, the default), `rocm_collector.py` (AMD amdsmi), `dcgm_collector.py`, `redfish_collector.py` (BMC inlet temp). `device_caps.py` detects MIG/vGPU and degrades gracefully. Add new hardware *here*, behind `hal.py` — nothing downstream should know the vendor.
- **The R_θ core** — `baseline.py` (virtual-ambient `T_ref`), `calibrate.py`, `window.py`, `temporal_filter.py`, `silicon.py`.
- **Detection** — `detector.py` (temporal), `peer.py` (within-node cross-sectional, the E009 method), `unsupervised.py` (critic), `predictor.py` / `predictor_cnn.py` (lead-time), `fault_classifier.py`, `causal.py`, `correlator.py`, `sdc_hunter.py`.
- **Decisioning/output** — `governor.py` (alert trust + FP budget), `health.py` / `health_api.py` (scheduler-facing level state + `/api/v1/conditions`), `alerter.py`, `exporter.py`, `otlp_exporter.py`, `metrics.py`, `jobreport.py` (`theta report` for SLURM/jobstats).
- **Infra** — `state.py`, `secrets.py` (Fernet+PBKDF2 for BMC creds), `safeio.py`, `telemetry.py` (opt-in anonymous upload).

Top level: `cli.py` (Typer command surface), `wizard.py` (the `theta setup` interactive flow, 34KB), `mcp_server.py` (MCP server, port 9102 — this is the `theta` server referenced from the `thermalos-vault` repo's `.mcp.json`), `models/train.py` + `models/bundle/` (the bundled Decision Tree).

Other top-level dirs: `sim/` — standalone physics simulation of the E-LT lead-time testbed (lets the predictive-maintenance claim be validated *before* Sam's physical hardware exists, fall 2026; has its own README + tests). `deploy/` — production systemd install (`install.sh`, `theta-monitor.service`), Grafana/Prometheus provisioning, jobstats integration. `tools/` — one-off analysis scripts (B200 baseline, E009 Princeton validation). `supabase/` + `.supabase/` — opt-in telemetry backend (US-East).

## Invariants worth knowing before you change things

- **Classification only fires on steady-state windows.** The `window.py` σ-gate before the classifier is load-bearing: it takes accuracy 84% → 99.8% and is what kills transient false positives. Don't classify raw samples.
- **The bundled classifier is Tesla T4-trained.** On B200/H100/A100 the R_θ operating range differs, so `theta monitor` *refuses to start* on detected non-T4 hardware until a calibration file exists. Calibration is not optional polish — it's a correctness gate. (See README "Production install".)
- **Temporal vs. peer detection are deliberately different shapes.** `detector.py` needs warm-up and catches drift over time; `peer.py` needs ≥4 matched-power node-mates but no warm-up and catches units degraded *before* the agent started. Peer self-disables below `MIN_GROUP` peers. Keep both — neither subsumes the other.
- **The governor exists to earn trust on a stranger's fleet.** Ground-truth hardware faults (ECC, Xid, throttle) fire immediately; anything *inferred* from R_θ statistics is held while warming and rate-budgeted. Routing a new alert through the wrong tier (immediate vs. inferential) is a behavioral regression even if the detection is correct.
- **`schedulable` ≠ healthy.** A warming-up GPU stays schedulable (the GPU is fine; only the monitor is learning). `TelemetryUnavailable` (e.g. vGPU guest) also stays schedulable — you don't drain a fleet because the monitor can't see. Health conditions are level state, alerts are edge events; they're orthogonal (`health.py` vs `alerter.py`).
- **Optional deps are inert when absent.** OTLP export (`runtheta[otlp]`) no-ops without the SDK so the base agent stays dependency-light. Preserve that pattern for anything new and heavy.
- **Tests are characterization tests** that pin real incidents (e.g. `test_peer.py` reproduces the E009 Princeton blind-flag). If one breaks, the burden is to show the *new* behavior is more correct than the pinned incident — not to re-baseline it away.

## Relationship to sibling repos

This is one of several ThermalOS repos under `~/`. The knowledge/strategy/research vault is `thermalos-vault` (Obsidian; has its own CLAUDE.md and a `theta` MCP server entry pointing at `theta/mcp_server.py` here). Findings referenced in code/README as F1/F2/F6 and experiments E001–E009 are documented there, not in this repo.
