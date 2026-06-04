# ThermalOS

**GPU thermal-power forensics agent.** Computes `R_θ = ΔT / P` in real time from your existing DCGM telemetry. That ratio is the only signal that separates a busy-hot GPU from a failing-hot one — and no incumbent computes it.

```
thermalos_gpu_rtheta_cwatt{gpu_index="3"} 2.104   # zombie recovery — CUDA context stuck
thermalos_gpu_rtheta_cwatt{gpu_index="3"} 0.724   # under load — healthy
thermalos_gpu_rtheta_cwatt{gpu_index="3"} 1.281   # clean idle — normal
```

---

## The problem

A GPU at 82°C could be:
- **Busy and healthy** — running a job at thermal equilibrium
- **Cooling path failing** — ambient temperature up, heatsink degrading
- **CUDA zombie** — process exited but context retained, drawing 31W at 0% utilization

`nvidia-smi`, DCGM, and Mission Control all expose T and P as separate fields. None of them divide the two. ThermalOS does.

---

## Quick start

### pip (single node, free forever)

```bash
pip install thermalos
thermalos setup        # interactive wizard — 90 seconds to first R_θ reading
thermalos monitor      # start monitoring
```

### Docker

```bash
docker run --gpus all -p 9101:9101 thermalos/agent:latest
```

### Docker Compose (agent + Prometheus + Grafana)

```bash
git clone https://github.com/Asomisetty27/thermalos
cd thermalos
docker compose --profile metrics up
```

Open `http://localhost:3000` — Grafana dashboard pre-provisioned, no setup required.
Login: `admin` / `thermalos`

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
thermalos setup                         Interactive wizard (run this first)
thermalos monitor                       Run agent — blocks until Ctrl+C
thermalos monitor --interval 2          Sample every 2s
thermalos monitor --gpus 0,1,3          Monitor specific GPUs
thermalos monitor --webhook <url>       Send alerts to Slack / generic webhook
thermalos monitor --log alerts.jsonl    Append alerts to JSONL file
thermalos monitor --port 9101           Prometheus metrics port (0 = disabled)
thermalos monitor --nb                  Use Naive Bayes instead of Decision Tree
thermalos baseline --gpu 0              Lock virtual ambient T_ref from idle window
thermalos baseline --gpu 0 --manual 24  Set T_ref manually (°C)
thermalos classify                      Snapshot classify all GPUs right now
thermalos serve --port 9101             Metrics export only (no stdout alerts)
thermalos train /path/data.csv          Retrain bundled models from new data
```

---

## Prometheus metrics

| Metric | Type | Description |
|---|---|---|
| `thermalos_gpu_rtheta_cwatt` | gauge | R_θ (C/W) — the core signal |
| `thermalos_gpu_state_info` | gauge | Current classified state (label: `state`) |
| `thermalos_gpu_drift_sigma` | gauge | Deviation from baseline in σ units |
| `thermalos_gpu_temperature_celsius` | gauge | Junction temperature |
| `thermalos_gpu_power_watts` | gauge | GPU power consumption |
| `thermalos_gpu_utilization_ratio` | gauge | 0–1 utilization |
| `thermalos_gpu_perf_state` | gauge | P-state (0=max, 8=idle) |
| `thermalos_gpu_baseline_tref_celsius` | gauge | Virtual ambient T_ref |
| `thermalos_gpu_window_rtheta_std` | gauge | Steady-state window σ |
| `thermalos_gpu_alerts_total` | counter | Alerts (labels: `severity`, `state`) |

All metrics include a `gpu_index` label.

---

## Alert payload (webhook / JSONL)

Every alert includes full forensic context:

```json
{
  "source":    "thermalos",
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

| Capability | DCGM | Mission Control | Phaidra | **ThermalOS** |
|---|:---:|:---:|:---:|:---:|
| Computes R_θ | ✗ | ✗ | ✗ | **✓** |
| Separates busy-hot vs failing-hot | ✗ | ✗ | ✗ | **✓** |
| CUDA zombie detection | ✗ | ✗ | ✗ | **✓** |
| Drift detection (baseline + k·σ) | ✗ | ✗ | ◐ | **✓** |
| Virtual ambient (no hardware) | ✗ | ✗ | ✗ | **✓** |
| Serves neocloud / mixed fleets | ✓ | ✗ | ✗ | **✓** |
| Open-source agent | ✓ | ✗ | ✗ | **✓** |

Mission Control ships only on Blackwell DGX/GB200. ThermalOS runs on any NVIDIA GPU reachable by pynvml.

---

## Requirements

- Python 3.10+
- NVIDIA GPU with driver ≥ 450 (for pynvml)
- No DCGM required — pynvml only

For Docker: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

## Retrain on your own data

```bash
thermalos train /path/to/measurements.csv
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
