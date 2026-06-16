# Theta

**GPU thermal-power forensics agent.** Computes `R_θ = ΔT / P` in real time from your existing DCGM telemetry. That ratio is the only signal that separates a busy-hot GPU from a failing-hot one — and no incumbent computes it.

```
theta_gpu_rtheta_cwatt{gpu_index="3"} 2.104   # zombie recovery — CUDA context stuck
theta_gpu_rtheta_cwatt{gpu_index="3"} 0.724   # under load — healthy
theta_gpu_rtheta_cwatt{gpu_index="3"} 1.281   # clean idle — normal
```

---

## The problem

A GPU at 82°C could be:
- **Busy and healthy** — running a job at thermal equilibrium
- **Cooling path failing** — ambient temperature up, heatsink degrading
- **CUDA zombie** — process exited but context retained, drawing 31W at 0% utilization

`nvidia-smi`, DCGM, and Mission Control all expose T and P as separate fields. None of them divide the two. Theta does.

---

## Quick start

### pip (single node, free forever)

```bash
pip install runtheta
theta setup        # interactive wizard — 90 seconds to first R_θ reading
theta monitor      # start monitoring
```

### Docker

```bash
docker run --gpus all -p 9101:9101 theta/agent:latest
```

### Docker Compose (agent + Prometheus + Grafana)

```bash
git clone https://github.com/Asomisetty27/theta
cd theta
docker compose --profile metrics up
```

Open `http://localhost:3000` — Grafana dashboard pre-provisioned, no setup required.
Login: `admin` / `theta`

---

## How it works

```
GPU (pynvml)
  → T_junction, P_GPU, util, P-state every 5s
  → R_θ = (T_junction − T_ref) / P_GPU
  → 15s steady-state window  (σ < 0.03 C/W)
  → Decision Tree classifier  →  {under_load, clean_idle, zombie_recovery, child_exit_recovery}
  → Rolling baseline + k·σ drift detector
  → Alert (stdout / Slack webhook / JSONL / Prometheus)
```

**Virtual ambient** — `T_ref` is derived from the GPU's own stable idle windows. No thermocouple, no rack modification, no extra hardware.

**Steady-state filter** — classification only runs on stable windows. This takes Naive Bayes accuracy from 84% → 99.8% and eliminates transient false positives.

**Peer-relative fleet detection** — on a multi-GPU node, Theta also compares each GPU's `R_θ` to its **matched-power node-mates** (median + MAD robust-z, hardware-agnostic relative scale). This is cross-sectional, so unlike the temporal baseline it needs **no warm-up** and catches a unit that has been degraded since before the agent started. On real Princeton H100 telemetry (72 GPUs) this method blind-flagged 3 degraded units — one at robust-z +15.6, two invisible to temperature thresholds. It self-disables on hosts with fewer than 4 matched-power peers, so single-GPU setups never see a peer alert.

**OpenTelemetry (OTLP) export** — Prometheus pull is the default, but fleets standardized on OpenTelemetry can have Theta push its core signals (R_θ, temperature, power, drift σ, readiness, schedulable) over OTLP/HTTP to their OTel Collector — no scrape config. Optional: `pip install runtheta[otlp]`, then `theta monitor --otlp <endpoint>`. Inert if the SDK isn't installed, so the base agent stays dependency-light.

**MIG / vGPU aware** — temperature and power are properties of the physical die, so Theta detects partitioning and virtualization and handles R_θ correctly instead of emitting nonsense. Under **MIG**, R_θ is reported per physical GPU (shared across all instances on that die), not fabricated per-instance. Under **vGPU**, if the guest can't read temperature/power, Theta marks the GPU `TelemetryUnavailable` and says "can't assess" rather than guessing — and keeps it schedulable (you don't drain a fleet because the monitor can't see it). Detection is best-effort and degrades gracefully on drivers/SKUs that don't expose the MIG/virtualization APIs.

**Health-as-conditions (scheduler-facing)** — alerts are edge events; a scheduler deciding whether to cordon or drain a node needs the orthogonal thing — the current *level* state: "is GPU 3 fit to run work right now, what's wrong, and since when?" Theta exposes per-GPU **health conditions** (the node-problem-detector pattern): a status (`healthy` / `warming` / `degraded` / `critical`), a single `schedulable` flag, and named conditions (`CoolingCritical`, `CoolingDegraded`, `ZombieContext`, `Throttling`, `EccErrors`, `TelemetryStale`) with transition timestamps — derived from signals the agent already computes. Read it with `theta health`, the `/api/v1/conditions` endpoint, or the `theta_gpu_schedulable` / `theta_gpu_health_condition` metrics. A GPU still *warming up* stays schedulable (the GPU is fine; only the monitor is learning).

**First-run trust + false-positive budget** — the agent earns trust on a stranger's fleet by being humble. Inferential alerts (anything derived from R_θ statistics — drift, peer, fault-curve, the unsupervised critic) are **held while a GPU is still warming up** ("learning your baseline, not yet confident"); ground-truth hardware faults (ECC, Xid, throttle) fire immediately. A **false-positive circuit breaker** watches the per-GPU alert rate — if it exceeds the budget, that means the thresholds are likely mis-calibrated for this hardware, so the agent goes quiet on that GPU and fires **one** meta-alert recommending `theta calibrate` instead of spraying wrong alarms. While a GPU has an active critical, concurrent lower-severity alerts for it are inhibited (Alertmanager-style). Readiness and suppression are exported (`theta_gpu_readiness`, `theta_alerts_suppressed_total`).

**Per-job report card (`theta report`, SLURM/jobstats)** — jobstats already scrapes per-GPU temperature, power, and utilization into Prometheus, labelled by SLURM `jobid`, node, and HGX ordinal — it just never divides temp by power. `theta report <jobid>` pulls a job's telemetry from that same Prometheus and produces a per-job cooling-health card: per-GPU R_θ, the fleet mean, and any degraded units in two tiers (**flagged** = act, **watch** = elevated). No new agent, no new telemetry. On the real Princeton incident job it reproduces the E009 result — fleet mean R_θ 0.0603 C/W, j13g2:7 flagged at +14.2σ (81 °C) and j12g2:6 at +4.0σ (72 °C, invisible to a temperature threshold), with the marginal j13g2:2 on watch. Works live (`--prom <url> --start --end`) or against saved Prometheus exports (`--export <dir>`).

**Position-conditioned cross-node scan (`theta fleet-scan`)** — across a *fleet* of nodes, HGX baseboard position imposes a thermal structure (±11% of mean R_θ on Della) that masks subtle degradation: a hot GPU in a structurally-cool slot can read below its node median yet be genuinely failing. `theta fleet-scan` pools R_θ across nodes and runs two-way (node × ordinal) **median polish** to remove that structure before scoring. On the real Princeton export it recovers **all 3** flagged units (the within-node detector alone finds 1) at zero false positives. It needs ≥2 nodes (a single node can't separate position from node effect); for a single host, `theta monitor`'s within-node peer detector is the right tool.

**Classifier** — Decision Tree trained on 4,570 rows of Stage 1 Tesla T4 data. 100% 5-fold CV accuracy on steady-state samples. Rules are human-readable and publishable:

```
IF R_θ ≤ 0.87        →  under_load          (n=963, conf=1.00)
IF R_θ > 0.87, P0    →  zombie_recovery     (n=584, conf=1.00)  ← CUDA zombie
IF R_θ > 1.50, P8    →  child_exit_recovery (n=696, conf=1.00)
ELSE                 →  clean_idle / early recovery
```

---

## CLI reference

```
theta setup                         Interactive wizard (run this first)
theta monitor                       Run agent — blocks until Ctrl+C
theta monitor --interval 2          Sample every 2s
theta monitor --gpus 0,1,3          Monitor specific GPUs
theta monitor --webhook <url>       Send alerts to Slack / generic webhook
theta monitor --pagerduty <key>     Route alerts to PagerDuty (Events API v2)
theta monitor --opsgenie <key>      Route alerts to Opsgenie (Alert API)
theta monitor --otlp <endpoint>     Push metrics over OTLP/HTTP (needs runtheta[otlp])
theta monitor --log alerts.jsonl    Append alerts to JSONL file
theta monitor --port 9101           Prometheus metrics port (0 = disabled)
theta monitor --nb                  Use Naive Bayes instead of Decision Tree
theta baseline --gpu 0              Lock virtual ambient T_ref from idle window
theta baseline --gpu 0 --manual 24  Set T_ref manually (°C)
theta classify                      Snapshot classify all GPUs right now
theta fleet-scan export.json        Cross-node position-conditioned anomaly scan
theta report <jobid> --prom <url>   Per-job R_θ report card (SLURM/jobstats)
theta health                        Scheduler-facing health conditions (is each GPU fit to run?)
theta serve --port 9101             Metrics export only (no stdout alerts)
theta train /path/data.csv          Retrain bundled models from new data
```

---

## Prometheus metrics

| Metric | Type | Description |
|---|---|---|
| `theta_gpu_rtheta_cwatt` | gauge | R_θ (C/W) — the core signal |
| `theta_gpu_state_info` | gauge | Current classified state (label: `state`) |
| `theta_gpu_drift_sigma` | gauge | Deviation from baseline in σ units |
| `theta_gpu_temperature_celsius` | gauge | Junction temperature |
| `theta_gpu_power_watts` | gauge | GPU power consumption |
| `theta_gpu_utilization_ratio` | gauge | 0–1 utilization |
| `theta_gpu_perf_state` | gauge | P-state (0=max, 8=idle) |
| `theta_gpu_baseline_tref_celsius` | gauge | Virtual ambient T_ref |
| `theta_gpu_window_rtheta_std` | gauge | Steady-state window σ |
| `theta_gpu_alerts_total` | counter | Alerts (labels: `severity`, `state`) |
| `theta_gpu_readiness` | gauge | 1 = confident, 0 = warming or FP-breaker tripped |
| `theta_alerts_suppressed_total` | counter | Inferential alerts withheld by the governor (label: `reason`) |
| `theta_gpu_schedulable` | gauge | 1 if the GPU is fit to schedule new work, else 0 |
| `theta_gpu_health_condition` | gauge | 1 if a named health condition is active (label: `condition`) |

All metrics include a `gpu_index` label.

---

## Alert payload (webhook / JSONL)

Every alert includes full forensic context:

```json
{
  "source":    "theta",
  "severity":  "critical",
  "gpu_index": 3,
  "state":     "zombie_recovery",
  "prev_state": "under_load",
  "rtheta":    1.541,
  "rtheta_baseline": 0.724,
  "drift_sigma": 4.2,
  "confidence": 1.0,
  "message":   "[CRITICAL] GPU 3 — CUDA zombie detected. R_θ=1.541 at 0% utilisation. Action: release CUDA context.",
  "context": {
    "severity": "critical",
    "duration_prev": 3842.1,
    "history": [
      { "ts": 1748995200.1, "state": "under_load", "r": 0.721, "conf": 0.99 }
    ]
  }
}
```

---

## Why not DCGM / Mission Control / Phaidra?

| Capability | DCGM | Mission Control | Phaidra | **Theta** |
|---|:---:|:---:|:---:|:---:|
| Computes R_θ | ✗ | ✗ | ✗ | **✓** |
| Separates busy-hot vs failing-hot | ✗ | ✗ | ✗ | **✓** |
| CUDA zombie detection | ✗ | ✗ | ✗ | **✓** |
| Drift detection (baseline + k·σ) | ✗ | ✗ | ◐ | **✓** |
| Virtual ambient (no hardware) | ✗ | ✗ | ✗ | **✓** |
| Serves neocloud / mixed fleets | ✓ | ✗ | ✗ | **✓** |
| Open-source agent | ✓ | ✗ | ✗ | **✓** |

Mission Control requires Base Command Manager (BCM) + DGX SuperPOD/BasePOD topology — it supports A100/H100/H200/B200 but only on NVIDIA's own orchestration stack. Theta runs on any NVIDIA GPU reachable by pynvml, regardless of orchestration layer.

---

## Production install (Linux service / DGX / AI Factory)

For shared clusters where the daemon should run as a system service:

```bash
# Run once as root — creates 'theta' system user, venv, systemd unit
sudo bash deploy/install.sh
```

Then calibrate **before** starting the daemon (required for non-T4 hardware):

```bash
# If the GPU has idle windows available:
sudo -u theta /opt/theta/venv/bin/theta calibrate --gpu 0 \
  --calibration-file /etc/theta/calibration.json

# On always-busy DGX nodes (no idle windows):
sudo -u theta /opt/theta/venv/bin/theta calibrate --gpu 0 \
  --ambient <coolant_inlet_temp_c> \
  --calibration-file /etc/theta/calibration.json

# Repeat for each GPU index, then start the service:
sudo systemctl enable --now theta
sudo journalctl -u theta -f
```

**Why calibration is required:** The bundled classifier is trained on Tesla T4 Stage 1 data. On B200/H100/A100 the R_θ operating range is different — running without calibration will systematically misclassify healthy nodes as anomalous. `theta monitor` will refuse to start on detected non-T4 hardware until calibration exists.

---

## IT / Security review

For system administrators evaluating whether to approve this deployment:

**What Theta runs as:**  
A Python daemon process under a dedicated system user (`theta`, no login shell). Requires membership in the `video` or `nvidia` group for NVML GPU access. Does not require root after install.

**Ports opened:**

| Port | Protocol | Purpose | Auth required |
|------|----------|---------|--------------|
| 9101 | TCP | Prometheus metrics scrape endpoint | None (read-only metrics) |
| 9102 | TCP | Health API (causal state, maintenance scores) | Bearer token (`THETA_HEALTH_TOKEN` env var) |

Both ports bind to `localhost` by default. To expose externally, set `bind_host` in config.

**What data leaves the node:**  
Opt-in telemetry only (off by default). When enabled, only aggregate anonymous statistics are uploaded: GPU model class, mean R_θ, ECC error rates, clock efficiency. **No** workload content, hostnames, IP addresses, job IDs, usernames, or model weights ever leave the node. Telemetry destination: Supabase hosted in US-East.

**What it reads:**  
NVML GPU telemetry (temperature, power, utilization, P-state, ECC counters, clock frequencies) via pynvml. Optionally: Redfish BMC inlet temperature (read-only, requires explicit credentials in config). No access to filesystem data, network traffic, job queues, or workload content.

**Coexistence with DCGM:**  
Safe to run alongside DCGM. Both read NVML; NVML is designed for concurrent access. Theta exports on port 9101, DCGM on 9400 — no collision. Theta metrics (`theta_gpu_*`) are additive to DCGM metrics, not duplicates.

**Config file:**  
`/etc/theta/` (production) or `~/.theta/` (single-user). Contains alert webhook URLs and optional BMC credentials (encrypted at rest with Fernet + PBKDF2).

---

## Requirements

- Python 3.10+
- NVIDIA GPU with driver ≥ 450 (for pynvml)
- No DCGM required — pynvml only

For Docker: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## Retrain on your own data

```bash
theta train /path/to/measurements.csv
```

CSV schema: `phase, trial_second, rtheta_cwatt, power_w, util_pct, perf_state, ...`

---

## Research basis

- **F1** — R_θ separates idle (1.28 C/W) from load (0.72 C/W) with 77.9% margin, Tesla T4
- **F2** — Ambient sensitivity: 7.1%/°C at idle vs 2.0%/°C at load (3.5× difference)
- **F6** — CUDA zombie: same-process exit leaves GPU at P0 (~31W), invisible to utilization

Stage 1: 4,570 rows · Tesla T4 · E001–E004 · 9 child-exit trials  
Stage 2 (in progress): Cal Poly DGX B200 AI Factory · E005–E008

---

## License

MIT — free forever for single-node use.

Built at Cal Poly SLO · [asomisetty27@gmail.com](mailto:asomisetty27@gmail.com)
