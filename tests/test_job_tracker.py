"""
Tests for SLURM job tracking (job_tracker.py).

Characterization tests that pin the expected behavior:
- Job registration and GPU mapping
- Workload detection via utilization spike (5% → 50%)
- R_theta baseline capture and max tracking
- Energy integration
- Thermal stress flagging
- Per-job report generation
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from theta.agent.job_tracker import JobTracker, JobContext, GpuJobState


class TestJobTrackerBasics:
    """Test job registration and GPU mapping."""

    def test_register_job(self):
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0, 1, 2])

        assert ctx.jobid == "12345"
        assert ctx.node == "node-a"
        assert ctx.gpu_indices == [0, 1, 2]
        assert ctx.start_time > 0

    def test_gpu_to_jobid_mapping(self):
        tracker = JobTracker()
        tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0, 1])
        tracker.register_job(jobid="12346", node="node-b", gpu_indices=[2, 3])

        assert tracker.get_active_jobid_for_gpu(0) == "12345"
        assert tracker.get_active_jobid_for_gpu(1) == "12345"
        assert tracker.get_active_jobid_for_gpu(2) == "12346"
        assert tracker.get_active_jobid_for_gpu(3) == "12346"
        assert tracker.get_active_jobid_for_gpu(4) is None

    def test_list_active_jobs(self):
        tracker = JobTracker()
        tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])
        tracker.register_job(jobid="12346", node="node-b", gpu_indices=[1])

        assert set(tracker.list_active_jobs()) == {"12345", "12346"}

        tracker.end_job("12345")
        assert tracker.list_active_jobs() == ["12346"]


class TestWorkloadDetection:
    """Test utilization spike detection (workload start)."""

    def test_workload_detection_from_idle(self):
        """Workload detected when util spikes from idle to active."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

        # Simulate idle: util < 5% for a while
        for i in range(15):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)
            assert ctx.gpu_states[0].workload_detected is False

        # Spike to active: util > 50% for sustained period
        for i in range(5):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.065, power_w=250, temperature_c=65, utilization_pct=80)

        # After 3 consecutive samples > 50%, workload should be detected
        assert ctx.gpu_states[0].workload_detected is True
        assert ctx.gpu_states[0].r_theta_baseline == pytest.approx(0.065, rel=0.01)

    def test_baseline_set_at_spike(self):
        """R_theta baseline is set when workload is detected (after UTIL_SUSTAINED confirmation)."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

        # Idle phase
        for i in range(15):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)

        # Spike with elevated R_theta (need 3 sustained samples to confirm)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.062, power_w=250, temperature_c=62, utilization_pct=80)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.064, power_w=260, temperature_c=64, utilization_pct=82)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.065, power_w=270, temperature_c=65, utilization_pct=85)

        # Baseline set at 3rd sustained sample (after UTIL_SUSTAINED=3 confirmation)
        assert ctx.gpu_states[0].r_theta_baseline == pytest.approx(0.065, rel=0.01)

    def test_no_workload_if_util_drops(self):
        """Workload not detected if util drops before sustaining."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

        # Idle
        for i in range(15):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)

        # Brief spike but drops before sustaining
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.060, power_w=200, temperature_c=60, utilization_pct=60)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.060, power_w=200, temperature_c=60, utilization_pct=60)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=15, temperature_c=30, utilization_pct=3)  # drop

        # Workload still not detected (didn't sustain)
        assert ctx.gpu_states[0].workload_detected is False


class TestRThetaTracking:
    """Test R_theta baseline and max tracking."""

    def test_rtheta_max_tracking(self):
        """R_theta max is tracked during job."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

        # Idle
        for i in range(15):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)

        # Workload with rising R_theta (need 3 sustained samples for detection)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.062, power_w=250, temperature_c=62, utilization_pct=80)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.070, power_w=280, temperature_c=70, utilization_pct=90)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.075, power_w=300, temperature_c=75, utilization_pct=95)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.072, power_w=290, temperature_c=72, utilization_pct=92)

        state = ctx.gpu_states[0]
        # Baseline set at 3rd sustained sample (0.075)
        assert state.r_theta_baseline == pytest.approx(0.075, rel=0.01)
        # Max continues to be tracked
        assert state.r_theta_max == pytest.approx(0.075, rel=0.01)

    def test_no_tracking_before_workload_detected(self):
        """R_theta not tracked until workload is detected."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

        # Idle with high R_theta (should not affect tracking)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.100, power_w=50, temperature_c=50, utilization_pct=3)

        state = ctx.gpu_states[0]
        assert state.r_theta_baseline is None
        assert state.r_theta_max is None


class TestEnergyIntegration:
    """Test energy (Wh) computation."""

    def test_energy_integration(self):
        """Energy computed as integral of power over time (all samples from job start)."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

        # Idle phase (15 samples at 10W)
        for i in range(15):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)

        # Workload phase (3 sustained + 10 additional at 250W)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.065, power_w=250, temperature_c=65, utilization_pct=85)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.065, power_w=250, temperature_c=65, utilization_pct=85)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.065, power_w=250, temperature_c=65, utilization_pct=85)
        for i in range(10):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.065, power_w=250, temperature_c=65, utilization_pct=85)

        state = ctx.gpu_states[0]
        # All samples: 15×10W + 13×250W = 150 + 3250 = 3400W total
        # 3400W × (5s / 3600s/h) ≈ 4.72 Wh
        expected_wh = (15.0 * 10.0 + 13.0 * 250.0) * (5.0 / 3600.0)
        assert state.compute_energy_wh() == pytest.approx(expected_wh, rel=0.01)


class TestThermalStress:
    """Test thermal stress flagging."""

    def test_thermal_stress_10pct_rise(self):
        """Thermal stress flagged when R_theta rises > 10%."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

        # Idle
        for i in range(15):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)

        # Sustained spike: baseline set after 3 samples, then rises
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.060, power_w=250, temperature_c=60, utilization_pct=85)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.062, power_w=280, temperature_c=62, utilization_pct=90)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.064, power_w=280, temperature_c=64, utilization_pct=90)
        # Now baseline=0.064, continues to track higher values
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.071, power_w=300, temperature_c=71, utilization_pct=95)

        state = ctx.gpu_states[0]
        # Baseline 0.064, max 0.071 = 10.9% rise → thermal stress
        assert state.is_thermal_stress() is True

    def test_no_thermal_stress_under_10pct(self):
        """No thermal stress if R_theta rises < 10%."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

        # Idle
        for i in range(15):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.060, power_w=10, temperature_c=25, utilization_pct=2)

        # Baseline 0.060, peak 0.064 = 6.7% rise → no stress
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.060, power_w=250, temperature_c=62, utilization_pct=85)
        tracker.update_gpu_sample(gpu_index=0, r_theta=0.064, power_w=280, temperature_c=65, utilization_pct=90)

        state = ctx.gpu_states[0]
        assert state.is_thermal_stress() is False


class TestJobCompletion:
    """Test job end and report generation."""

    def test_job_end(self):
        """Job ending unmaps GPUs and marks as ended."""
        tracker = JobTracker()
        tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0, 1])

        assert tracker.get_active_jobid_for_gpu(0) == "12345"

        ctx = tracker.end_job("12345")
        assert ctx.has_ended is True
        assert ctx.end_time > ctx.start_time
        assert tracker.get_active_jobid_for_gpu(0) is None
        assert tracker.get_active_jobid_for_gpu(1) is None

    def test_job_report_generation(self):
        """Per-job report written to ~/.theta/job_<JOBID>.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = JobTracker(job_report_dir=Path(tmpdir))
            ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0])

            # Simulate workload with sustained high util + rising R_theta
            for i in range(15):
                tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)

            # Sustained spike with rising R_theta (baseline set at 3rd sample)
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.060, power_w=250, temperature_c=60, utilization_pct=85)
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.065, power_w=270, temperature_c=65, utilization_pct=90)
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.070, power_w=280, temperature_c=70, utilization_pct=90)
            for i in range(10):
                tracker.update_gpu_sample(gpu_index=0, r_theta=0.070, power_w=280, temperature_c=70, utilization_pct=90)

            # End job and write report
            ctx = tracker.end_job("12345")
            report_path = tracker.write_job_report(ctx)

            # Verify report
            assert report_path.exists()
            report_data = json.loads(report_path.read_text())
            assert report_data["jobid"] == "12345"
            assert report_data["node"] == "node-a"
            assert report_data["gpu_indices"] == [0]
            assert "per_gpu" in report_data
            assert "0" in report_data["per_gpu"]
            # Baseline 0.070, max 0.070 = no rise, but let's verify the report format
            assert report_data["per_gpu"]["0"]["r_theta_baseline"] is not None

    def test_report_dict_serialization(self):
        """JobContext serializes correctly to report format."""
        ctx = JobContext(jobid="12345", node="node-a", gpu_indices=[0], start_time=time.time())
        ctx.gpu_states[0] = GpuJobState(gpu_index=0, r_theta_baseline=0.058, r_theta_max=0.070)

        report = ctx.as_report_dict()
        assert report["jobid"] == "12345"
        assert report["node"] == "node-a"
        assert report["per_gpu"]["0"]["r_theta_baseline"] == 0.058
        assert report["per_gpu"]["0"]["r_theta_max"] == 0.070
        assert report["per_gpu"]["0"]["r_theta_delta_percent"] == pytest.approx(20.69, rel=0.01)


class TestMultiGPUJob:
    """Test jobs spanning multiple GPUs."""

    def test_multi_gpu_tracking(self):
        """Track multiple GPUs independently within one job."""
        tracker = JobTracker()
        ctx = tracker.register_job(jobid="12345", node="node-a", gpu_indices=[0, 1, 2])

        # Simulate workload on GPU 0 only
        for i in range(15):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)
            tracker.update_gpu_sample(gpu_index=1, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)
            tracker.update_gpu_sample(gpu_index=2, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)

        # GPU 0 gets workload
        for i in range(5):
            tracker.update_gpu_sample(gpu_index=0, r_theta=0.070, power_w=280, temperature_c=70, utilization_pct=90)
            tracker.update_gpu_sample(gpu_index=1, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)
            tracker.update_gpu_sample(gpu_index=2, r_theta=0.058, power_w=10, temperature_c=25, utilization_pct=2)

        # Only GPU 0 should detect workload
        assert ctx.gpu_states[0].workload_detected is True
        assert ctx.gpu_states[1].workload_detected is False
        assert ctx.gpu_states[2].workload_detected is False

        # Only GPU 0 should have R_theta max set (GPUs 1 & 2 have no workload, so max is None)
        assert ctx.gpu_states[0].r_theta_max is not None
        assert ctx.gpu_states[1].r_theta_max is None
        assert ctx.gpu_states[2].r_theta_max is None
