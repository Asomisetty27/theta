# SLURM Job Tracking Setup Guide

This guide explains how to configure SLURM to use Theta's job tracking hooks for per-job thermal provenance tracking.

## Overview

Theta tracks per-job thermal metrics (R_theta baseline, max, delta, energy, thermal stress) by:

1. **Prolog hook** (`theta-prolog.sh`) — runs before job starts, queries Theta health API for baseline R_theta
2. **Daemon** — feeds each GPU sample through the job tracker, accumulating R_theta and energy per job
3. **Epilog hook** (`theta-epilog.sh`) — runs after job ends, computes final report

Result: Per-job JSON reports at `~/.theta/job_<JOBID>.json` with thermal impact summary.

## Installation

### 1. Install Theta Agent

```bash
pip install theta
```

Verify the daemon is running:
```bash
theta monitor
```

### 2. Install Hook Scripts

Copy the hooks to a system location:

```bash
sudo mkdir -p /opt/theta/jobstats
sudo cp theta/deploy/jobstats/theta-prolog.sh /opt/theta/jobstats/
sudo cp theta/deploy/jobstats/theta-epilog.sh /opt/theta/jobstats/
sudo chmod +x /opt/theta/jobstats/theta-*.sh
```

### 3. Configure SLURM

Edit `/etc/slurm/slurm.conf` and add the hook paths. You have two options:

**Option A: Global hooks** (runs on all partitions)

```conf
PrologSlurmctld=/opt/theta/jobstats/theta-prolog.sh
EpilogSlurmctld=/opt/theta/jobstats/theta-epilog.sh
```

**Option B: Partition-specific** (recommended for selective tracking)

```conf
PartitionName=gpu
    Prolog=/opt/theta/jobstats/theta-prolog.sh
    Epilog=/opt/theta/jobstats/theta-epilog.sh
    # ... other partition config
```

Then reload SLURM:
```bash
sudo scontrol reconfigure
```

### 4. Verify Hooks Are Wired

```bash
# Submit a test job
sbatch --gres=gpu:1 sleep 10

# Check SLURM logs for prolog/epilog execution
tail -f /var/log/slurm/slurmd.log | grep "Theta"

# After job ends, check for report
ls ~/.theta/job_*.json
cat ~/.theta/job_<JOBID>.json
```

## Configuration

### Health API Endpoint

By default, the hooks query `http://localhost:7777` for Theta's health API.

To use a different endpoint:

```bash
export THETA_HEALTH_API_ENDPOINT="http://monitoring-node:7777"
sbatch ...
```

Or set it in SLURM's environment:

```bash
# In slurm.conf
export THETA_HEALTH_API_ENDPOINT="http://monitoring-node:7777"
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `THETA_HEALTH_API_ENDPOINT` | `http://localhost:7777` | Theta health API URL |
| `THETA_JOBSTATS_ENABLED` | (unset) | Set to `1` to enable job tracking (optional gate) |

## Output Format

Job reports are written to `~/.theta/job_<JOBID>.json`:

```json
{
  "jobid": "12345",
  "node": "node-a",
  "start_time": 1234567890.5,
  "end_time": 1234567920.3,
  "duration_sec": 29.8,
  "per_gpu": {
    "0": {
      "gpu_index": 0,
      "r_theta_baseline": 0.058,
      "r_theta_max": 0.075,
      "r_theta_delta_percent": 29.3,
      "energy_wh": 1.25,
      "thermal_stress": true,
      "n_steady_samples": 15
    }
  }
}
```

### Field Definitions

- `r_theta_baseline`: R_theta (°C/W) at workload detection time
- `r_theta_max`: Peak R_theta during job (thermal degradation indicator)
- `r_theta_delta_percent`: Rise from baseline to max, as a percentage
- `energy_wh`: Total energy consumed by this GPU during the job (watt-hours)
- `thermal_stress`: Boolean flag — `true` if R_theta rise > 10% (indicates thermal throttling or paste degradation)
- `n_steady_samples`: Number of steady-state samples tracked (for confidence)

## Reporting

### List recent jobs

```bash
ls -ltr ~/.theta/job_*.json | tail -10
```

### Query a specific job

```bash
theta report --job 12345
# or manually:
cat ~/.theta/job_12345.json | jq .
```

### Bulk analysis

```bash
# Find all jobs with thermal stress
for f in ~/.theta/job_*.json; do
  jq 'select(.per_gpu[].thermal_stress == true) | .jobid' "$f"
done

# Average R_theta delta across all jobs
jq '.per_gpu[].r_theta_delta_percent' ~/.theta/job_*.json | \
  awk '{sum+=$1; n++} END {print "Average delta: " sum/n "%"}'
```

## Troubleshooting

### Hooks not running

Check SLURM logs:
```bash
grep "Prolog\|Epilog" /var/log/slurm/slurmd.log
```

If absent, verify the paths in `slurm.conf` are correct and readable:
```bash
ls -l /opt/theta/jobstats/theta-*.sh
```

### Health API connection errors

Verify Theta daemon is running and API is reachable:
```bash
curl http://localhost:7777/api/v1/metrics | jq .
```

If it fails, check Theta daemon logs:
```bash
systemctl status theta-monitor
journalctl -u theta-monitor -n 20
```

### No reports written

Verify the `.theta` directory exists and is writable:
```bash
ls -ld ~/.theta
touch ~/.theta/test.txt && rm ~/.theta/test.txt
```

If Prometheus metrics conflict appears in logs, clear the registry:
```bash
# This is typically only needed in test environments
python -c "from prometheus_client import REGISTRY; [REGISTRY.unregister(c) for c in list(REGISTRY._collector_to_names)]"
```

## Cal Poly AI Factory Deployment

For the Cal Poly DGX B200 deployment:

1. Install Theta on the login node
2. Install hooks in `/opt/theta/jobstats/`
3. Configure SLURM on the login node
4. Run `theta setup` to calibrate B200 GPUs (handles H100 calibration data)
5. Start daemon: `systemctl start theta-monitor`
6. Verify: `theta status`

Monthly reporting:
```bash
# Aggregate thermal stats for the month
for f in ~/.theta/job_*.json; do
  jq '.jobid, .per_gpu[].r_theta_delta_percent, .per_gpu[].thermal_stress' "$f"
done | sort | uniq -c
```

## References

- [SLURM Prolog/Epilog Documentation](https://slurm.schedmd.com/slurm.conf.html#OPT_Prolog)
- [Theta Agent README](../../README.md)
- [SLURM Integration Plan](../../SLURM_INTEGRATION_PLAN.md)
- [Job Tracker Implementation](../../theta/agent/job_tracker.py)
