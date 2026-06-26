# SLURM Integration Plan — v0.1.9 (Cal Poly DGX B200)

**Deadline:** August 17, 2026 (Cal Poly AI Factory deployment)  
**Current status:** jobreport.py exists for post-hoc analysis; prolog/epilog hooks need implementation

---

## Gap Analysis (from thermalos_gap_analysis.md)

### Problem
- NVIDIA's SLURM+DCGM integration has a critical timing bug: job stats recording starts with the prolog command, not with the actual user workload
- If 10s passes between prolog and job start, the energy and thermal stats for those 10s are attributed to the wrong job
- DCGM's SLURM stats module marks "Board limit slowdowns" and "Reliability throttling" as **Not Supported** — exactly what Theta can detect
- Current jobreport.py is post-hoc analysis; we need **real-time per-job tracking**

### Solution
Implement prolog/epilog hooks that:
1. **Prolog:** Record job start timestamp, GPU assignment, R_theta baseline at job start
2. **Epilog:** Record job end, compute per-job R_theta delta (degradation during job), energy, MFU estimate
3. **Timing fix:** Align R_theta recording to workload start (detected by GPU util spike), not prolog execution
4. **Per-job report:** `~/.theta/job_<SLURM_JOBID>.json` with thermal provenance

---

## Implementation Plan

### Phase 1: Core Tracking (Week 1-2)

#### 1.1 Job Tracker Module (`job_tracker.py`)
- Track per-GPU job context (jobid, node_index, start_time, gpu_index)
- Map SLURM_JOBID → GPU indices
- Store baseline R_theta at job start
- Detect job transitions (workload start via util spike, job end)

```python
@dataclass
class JobContext:
    jobid: str
    start_time: float
    start_util_spike_time: Optional[float]  # when did utilization spike?
    gpu_indices: list[int]
    baseline_r_theta: dict[int, float]      # gpu_index → r_theta at job start
    max_r_theta_delta: dict[int, float]     # gpu_index → max degradation during job
    total_energy_wh: dict[int, float]       # gpu_index → Wh consumed
```

#### 1.2 Prolog Hook (`deploy/jobstats/theta-prolog.sh`)
- Called by SLURM before job starts
- Extract: `$SLURM_JOBID`, `$SLURM_GPUS`, node name
- Query Theta's health API: current R_theta baseline for each GPU
- Write `/tmp/theta_job_$SLURM_JOBID.json` with initial state

```bash
#!/bin/bash
JOBID=$SLURM_JOBID
GPUS=$SLURM_GPUS
NODE=$(hostname)
TIMESTAMP=$(date +%s.%N)

# Query current R_theta for this job's GPUs
curl -s http://localhost:7777/api/v1/metrics | jq ".gpu_metrics[] | select(.gpu_index | inside($GPUS)) | {gpu_index, r_theta, utilization}" > /tmp/theta_job_${JOBID}_start.json

echo "{\"jobid\": \"$JOBID\", \"node\": \"$NODE\", \"gpus\": [$GPUS], \"start_time\": $TIMESTAMP}" > /tmp/theta_job_${JOBID}.json
```

#### 1.3 Epilog Hook (`deploy/jobstats/theta-epilog.sh`)
- Called by SLURM after job ends
- Collect end state from Theta
- Compute delta (R_theta rise, energy consumed)
- Write final report to `~/.theta/job_<JOBID>.json`

### Phase 2: Daemon Integration (Week 2-3)

#### 2.1 Modify `daemon.py`
- Add job tracker initialization to `AgentDaemon`
- On each sample, check for job state changes
- Track max R_theta delta and energy per job
- Detect job start (util spike from < 5% to > 50%)
- Detect job end (SLURM signal or util drop)

```python
async def process_tick(self, samples: list[EnrichedSample]) -> None:
    # ... existing pipeline ...
    
    # NEW: Job tracking
    for sample in samples:
        job_context = self.job_tracker.update(sample)
        if job_context and job_context.has_ended():
            # Write per-job report
            await self.write_job_report(job_context)
```

#### 2.2 Job Report Generation
- When job ends, compute:
  - Per-GPU R_theta delta (max - baseline)
  - Total energy consumed (integral of power)
  - MFU (flop-utilization) estimate if accessible
  - Flags: any R_theta rise > 10% → "thermal_stress"

```python
def write_job_report(job_context: JobContext) -> Path:
    report = {
        "jobid": job_context.jobid,
        "node": job_context.node,
        "gpus": job_context.gpu_indices,
        "start_time": job_context.start_time,
        "end_time": time.time(),
        "per_gpu": {
            str(idx): {
                "r_theta_baseline": job_context.baseline_r_theta[idx],
                "r_theta_max": compute_max(idx),  # from detector baseline
                "r_theta_delta_percent": ((max - baseline) / baseline) * 100,
                "energy_wh": job_context.total_energy_wh[idx],
                "thermal_stress": max > baseline * 1.1,
            }
            for idx in job_context.gpu_indices
        }
    }
    
    path = Path.home() / ".theta" / f"job_{job_context.jobid}.json"
    path.write_text(json.dumps(report, indent=2))
    return path
```

### Phase 3: Deployment Integration (Week 3)

#### 3.1 SLURM Configuration for Cal Poly
Add to `/etc/slurm/slurm.conf`:
```
PrologSlurmctld=/opt/theta/deploy/jobstats/theta-prolog.sh
EpilogSlurmctld=/opt/theta/deploy/jobstats/theta-epilog.sh
```

Or per-partition:
```
PartitionName=gpu Prolog=/opt/theta/deploy/jobstats/theta-prolog.sh
PartitionName=gpu Epilog=/opt/theta/deploy/jobstats/theta-epilog.sh
```

#### 3.2 Systemd Service Integration
- `theta-monitor.service` already runs the daemon
- Add environment var: `THETA_JOBSTATS_ENABLED=1`
- Health API listens on `localhost:7777` for prolog/epilog queries

#### 3.3 Monthly Report Generation
- `theta report --job $JOBID` reads per-job data
- Feed into Lupo's dashboard: per-student thermal provenance

---

## Data Flow

```
Job submitted
    ↓
Prolog executes
    ├─ Read current R_theta baseline from Theta health API
    └─ Store in /tmp/theta_job_$JOBID.json
    ↓
Job runs
    ├─ GPU utilization spikes → Theta detects workload start
    ├─ Daemon tracks max R_theta, total energy per GPU
    └─ Per-GPU R_theta delta accumulated
    ↓
Job ends
    ├─ Epilog executes
    ├─ Compute final R_theta max, energy, thermal stress
    └─ Write ~/.theta/job_$JOBID.json
    ↓
User runs: theta report --job 12345
    ├─ Read ~/.theta/job_12345.json
    ├─ Display: per-GPU R_theta rise, energy, thermal stress flags
    └─ Show in dashboard / Lupo's reports
```

---

## Testing Strategy

### Unit Tests (`tests/test_job_tracker.py`)
- Job context creation
- Util spike detection (5% → 50%)
- R_theta delta computation
- Energy integration

### Integration Tests
- Mock Prometheus jobstats data (like existing jobreport tests)
- Run daemon with mock prolog/epilog
- Verify per-job report generation

### Manual Testing at Cal Poly (Aug 1-17)
- Run on 1 node first, then expand to 4 nodes
- Submit test jobs, verify per-job reports
- Cross-check with SLURM accounting data

---

## Files to Create/Modify

### New Files
- `theta/agent/job_tracker.py` (250 lines)
- `deploy/jobstats/theta-prolog.sh` (50 lines)
- `deploy/jobstats/theta-epilog.sh` (80 lines)
- `tests/test_job_tracker.py` (300 lines)

### Modified Files
- `theta/agent/daemon.py` (+50 lines for job tracking integration)
- `theta/agent/jobreport.py` (+100 lines for job context enrichment)
- `theta/cli.py` (+30 lines for `theta report` enhancement)

### Configuration
- `deploy/systemd/theta-monitor.service` (add THETA_JOBSTATS_ENABLED)
- `deploy/slurm/example-slurm.conf` (document prolog/epilog config)

---

## Timeline

| Date | Milestone | Notes |
|------|-----------|-------|
| Jun 26–30 | Impl job_tracker.py + prolog/epilog | Core tracking logic |
| Jul 1–7 | Daemon integration + per-job reports | Real-time tracking |
| Jul 8–14 | Unit + integration tests | Validation |
| Jul 15–17 | Manual testing at Cal Poly | 1 node → 4 nodes |
| Aug 17 | Live deployment | DGX B200, 32 GPUs |

---

## Success Criteria

- ✅ Per-job JSON reports written to `~/.theta/job_*.json`
- ✅ R_theta delta captured (baseline → max during job)
- ✅ Energy consumption tracked per GPU per job
- ✅ Thermal stress flagged (R_theta rise > 10%)
- ✅ Integration with existing jobstats (no new telemetry required)
- ✅ Timing synchronized: workload-start detection via util spike, not prolog timestamp
- ✅ Zero false positives in normal operation (no job → no report)
- ✅ Reports available in `theta report --job $JOBID`

---

## Notes

- Prolog/epilog hooks must be **fast** (< 100 ms) — they block SLURM scheduling
- Job start detection via util spike is more reliable than prolog timing
- Per-job reports are the **thermal provenance** for every student paper at Cal Poly AI Factory
- This work unblocks the August 17 deployment and feeds the ICPE 2027 paper (multi-fleet validation)
