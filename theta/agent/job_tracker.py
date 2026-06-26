"""
Per-job thermal tracking for SLURM integration.

Tracks per-GPU job context (jobid, start time, R_theta baseline, degradation)
and detects job transitions via utilization spikes. Writes per-job reports to
~/.theta/job_<JOBID>.json when jobs end.

This solves Gap 2 from thermalos_gap_analysis: SLURM's prolog runs at submit time,
not workload start. We detect actual workload start via GPU util spike (5% → 50%).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Detection thresholds
UTIL_IDLE_THRESHOLD = 5.0          # GPU idle if util < 5%
UTIL_WORKLOAD_THRESHOLD = 50.0     # Workload detected if util > 50%
UTIL_SPIKE_WINDOW = 10             # Samples to detect spike (10 × 5s = 50s)
UTIL_SUSTAINED = 3                 # Sustained above threshold to confirm
MIN_STEADY_SAMPLES = 30            # Minimum samples before a GPU is scored
THERMAL_STRESS_THRESHOLD = 1.10    # R_theta rise > 10% = thermal stress


@dataclass
class GpuJobState:
    """Per-GPU state during a job."""
    gpu_index: int
    r_theta_baseline: Optional[float] = None
    r_theta_max: Optional[float] = None
    r_theta_max_timestamp: Optional[float] = None
    power_samples: list[float] = field(default_factory=list)
    temperature_samples: list[float] = field(default_factory=list)
    util_history: list[float] = field(default_factory=list)
    workload_detected: bool = False
    workload_detect_time: Optional[float] = None
    n_steady_samples: int = 0

    def compute_energy_wh(self) -> float:
        """Integrate power over time to get energy (Wh)."""
        if not self.power_samples or len(self.power_samples) < 2:
            return 0.0
        # Assume 5s between samples
        interval_hours = 5.0 / 3600.0
        total_wh = sum(self.power_samples) * interval_hours
        return total_wh

    def is_thermal_stress(self) -> bool:
        """True if R_theta rose > 10% during job."""
        if self.r_theta_baseline is None or self.r_theta_max is None:
            return False
        if self.r_theta_baseline == 0:
            return False
        delta_pct = (self.r_theta_max - self.r_theta_baseline) / self.r_theta_baseline
        return delta_pct > (THERMAL_STRESS_THRESHOLD - 1.0)

    def as_report_dict(self) -> dict:
        """Serialize to job report format."""
        return {
            "gpu_index": self.gpu_index,
            "r_theta_baseline": self.r_theta_baseline,
            "r_theta_max": self.r_theta_max,
            "r_theta_delta_percent": (
                ((self.r_theta_max - self.r_theta_baseline) / self.r_theta_baseline * 100)
                if self.r_theta_baseline and self.r_theta_max else None
            ),
            "energy_wh": self.compute_energy_wh(),
            "thermal_stress": self.is_thermal_stress(),
            "n_steady_samples": self.n_steady_samples,
        }


@dataclass
class JobContext:
    """Job lifecycle tracking."""
    jobid: str
    node: str
    gpu_indices: list[int]
    start_time: float
    gpu_states: dict[int, GpuJobState] = field(default_factory=dict)
    has_ended: bool = False
    end_time: Optional[float] = None

    def update_gpu_sample(
        self,
        gpu_index: int,
        r_theta: float,
        power_w: float,
        temperature_c: float,
        utilization_pct: float,
    ) -> None:
        """Update per-GPU state with a new sample."""
        if gpu_index not in self.gpu_states:
            self.gpu_states[gpu_index] = GpuJobState(gpu_index=gpu_index)

        state = self.gpu_states[gpu_index]
        state.util_history.append(utilization_pct)
        state.power_samples.append(power_w)
        state.temperature_samples.append(temperature_c)

        # Update max R_theta if workload has been detected
        if state.workload_detected:
            if state.r_theta_max is None or r_theta > state.r_theta_max:
                state.r_theta_max = r_theta
                state.r_theta_max_timestamp = time.time()
            state.n_steady_samples += 1

        # Detect workload start: util spike from idle → active
        if not state.workload_detected:
            recent_util = state.util_history[-UTIL_SPIKE_WINDOW:]
            if len(recent_util) >= UTIL_SPIKE_WINDOW:
                # Check if there's a spike: most recent sustained above threshold
                recent_sustained = recent_util[-UTIL_SUSTAINED:]
                if all(u > UTIL_WORKLOAD_THRESHOLD for u in recent_sustained):
                    # Workload confirmed after UTIL_SUSTAINED consecutive samples
                    # above threshold. Baseline is the R_theta at confirmation
                    # time (the workload has ramped to steady power by now, so
                    # this reflects the thermal starting point of the run rather
                    # than a transient spike).
                    state.workload_detected = True
                    state.workload_detect_time = time.time()
                    if state.r_theta_baseline is None:
                        state.r_theta_baseline = r_theta
                        state.r_theta_max = r_theta
                    log.info(
                        "workload_detected",
                        gpu_index=gpu_index,
                        jobid=self.jobid,
                        r_theta_baseline=state.r_theta_baseline,
                    )

    def as_report_dict(self) -> dict:
        """Serialize to job report format."""
        return {
            "jobid": self.jobid,
            "node": self.node,
            "gpu_indices": self.gpu_indices,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": (self.end_time - self.start_time) if self.end_time else None,
            "per_gpu": {
                str(idx): state.as_report_dict()
                for idx, state in self.gpu_states.items()
            },
        }


class JobTracker:
    """Tracks active and completed jobs across all GPUs."""

    def __init__(self, job_report_dir: Optional[Path] = None):
        self.active_jobs: dict[str, JobContext] = {}  # jobid → JobContext
        self.gpu_to_jobid: dict[int, str] = {}        # gpu_index → current jobid
        self.job_report_dir = job_report_dir or (Path.home() / ".theta")
        self.completed_jobs: list[str] = []

    def register_job(
        self,
        jobid: str,
        node: str,
        gpu_indices: list[int],
    ) -> JobContext:
        """Register a new job (called from prolog hook)."""
        ctx = JobContext(
            jobid=jobid,
            node=node,
            gpu_indices=gpu_indices,
            start_time=time.time(),
        )
        self.active_jobs[jobid] = ctx

        # Map GPUs to this job
        for gpu_idx in gpu_indices:
            self.gpu_to_jobid[gpu_idx] = jobid

        log.info("job_registered", jobid=jobid, gpu_indices=gpu_indices)
        return ctx

    def update_gpu_sample(
        self,
        gpu_index: int,
        r_theta: float,
        power_w: float,
        temperature_c: float,
        utilization_pct: float,
    ) -> Optional[JobContext]:
        """
        Update job tracking with a new sample. Returns the JobContext if a job
        is active for this GPU, None otherwise.
        """
        jobid = self.gpu_to_jobid.get(gpu_index)
        if jobid is None:
            return None

        ctx = self.active_jobs.get(jobid)
        if ctx is None or ctx.has_ended:
            return None

        ctx.update_gpu_sample(gpu_index, r_theta, power_w, temperature_c, utilization_pct)
        return ctx

    def end_job(self, jobid: str) -> Optional[JobContext]:
        """Mark a job as ended and return its final context."""
        ctx = self.active_jobs.get(jobid)
        if ctx is None:
            return None

        ctx.has_ended = True
        ctx.end_time = time.time()

        # Unmap GPUs
        for gpu_idx in ctx.gpu_indices:
            self.gpu_to_jobid.pop(gpu_idx, None)

        self.completed_jobs.append(jobid)
        log.info(
            "job_ended",
            jobid=jobid,
            duration_sec=ctx.end_time - ctx.start_time,
        )

        return ctx

    def write_job_report(self, ctx: JobContext) -> Path:
        """Write per-job thermal report to disk."""
        self.job_report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.job_report_dir / f"job_{ctx.jobid}.json"

        report_data = ctx.as_report_dict()
        report_path.write_text(json.dumps(report_data, indent=2))

        log.info("job_report_written", jobid=ctx.jobid, path=str(report_path))
        return report_path

    def get_job(self, jobid: str) -> Optional[JobContext]:
        """Retrieve a job context by ID."""
        return self.active_jobs.get(jobid)

    def get_active_jobid_for_gpu(self, gpu_index: int) -> Optional[str]:
        """Get the active job ID for a GPU, if any."""
        return self.gpu_to_jobid.get(gpu_index)

    def list_active_jobs(self) -> list[str]:
        """List all active job IDs."""
        return [jid for jid, ctx in self.active_jobs.items() if not ctx.has_ended]
