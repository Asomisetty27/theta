"""
Shell integration tests for the SLURM prolog/epilog hooks.

These tests actually EXECUTE the bash scripts (not just import Python) because
the hooks run on the SLURM cluster as standalone shell. A pure-Python test
cannot catch shell portability bugs — e.g. `mapfile`/`declare -A` are bash 4+
builtins absent on bash 3.2, and would silently fail at runtime while passing
`bash -n` syntax checks.

The hooks must:
- Run on bash 3.2+ (portability — clusters vary)
- Degrade gracefully when the health API is unreachable (never block a job)
- Always exit 0 (a failing SLURM epilog can mark a node down)
- Produce valid JSON job reports
- Clean up their temp dirs
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HOOK_DIR = Path(__file__).parent.parent / "deploy" / "jobstats"
PROLOG = HOOK_DIR / "theta-prolog.sh"
EPILOG = HOOK_DIR / "theta-epilog.sh"

# Skip the whole module if bash isn't available (e.g. exotic CI image)
bash_available = shutil.which("bash") is not None
pytestmark = pytest.mark.skipif(not bash_available, reason="bash not available")


def _run(script: Path, env: dict) -> subprocess.CompletedProcess:
    """Run a hook script with a given environment, capturing output."""
    full_env = {**os.environ, **env}
    return subprocess.run(
        ["bash", str(script)],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestHookScriptsExist:
    def test_hook_scripts_present_and_executable(self):
        assert PROLOG.exists(), f"prolog missing at {PROLOG}"
        assert EPILOG.exists(), f"epilog missing at {EPILOG}"
        # Executable bit set (deployed via SLURM, must be runnable)
        assert os.access(PROLOG, os.X_OK), "prolog not executable"
        assert os.access(EPILOG, os.X_OK), "epilog not executable"

    def test_hook_scripts_pass_syntax_check(self):
        for script in (PROLOG, EPILOG):
            result = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True
            )
            assert result.returncode == 0, f"{script.name} syntax error: {result.stderr}"


class TestPrologHook:
    def test_prolog_parses_comma_separated_gpus(self, tmp_path):
        """Prolog parses SLURM_GPUS='0,1' into 2 GPUs and writes metadata."""
        jobid = "test_prolog_comma"
        job_tmpdir = Path(f"/tmp/theta_job_{jobid}")
        if job_tmpdir.exists():
            shutil.rmtree(job_tmpdir)

        result = _run(PROLOG, {
            "SLURM_JOBID": jobid,
            "SLURM_GPUS": "0,1",
            # Unreachable endpoint — must degrade gracefully
            "THETA_HEALTH_API_ENDPOINT": "http://localhost:1",
        })

        try:
            assert result.returncode == 0, f"prolog failed: {result.stderr}"
            metadata = (job_tmpdir / "metadata").read_text()
            assert f"jobid={jobid}" in metadata
            assert "gpu_count=2" in metadata
            assert "start_time=" in metadata
        finally:
            if job_tmpdir.exists():
                shutil.rmtree(job_tmpdir)

    def test_prolog_handles_bracket_notation(self, tmp_path):
        """Prolog parses SLURM_GPUS='[0-3]' into 4 GPUs."""
        jobid = "test_prolog_bracket"
        job_tmpdir = Path(f"/tmp/theta_job_{jobid}")
        if job_tmpdir.exists():
            shutil.rmtree(job_tmpdir)

        result = _run(PROLOG, {
            "SLURM_JOBID": jobid,
            "SLURM_GPUS": "[0-3]",
            "THETA_HEALTH_API_ENDPOINT": "http://localhost:1",
        })

        try:
            assert result.returncode == 0, f"prolog failed: {result.stderr}"
            metadata = (job_tmpdir / "metadata").read_text()
            assert "gpu_count=4" in metadata
        finally:
            if job_tmpdir.exists():
                shutil.rmtree(job_tmpdir)

    def test_prolog_handles_alternate_gpu_var_names(self):
        """Prolog falls back to SLURM_JOB_GPUS / CUDA_VISIBLE_DEVICES."""
        jobid = "test_prolog_altvar"
        job_tmpdir = Path(f"/tmp/theta_job_{jobid}")
        if job_tmpdir.exists():
            shutil.rmtree(job_tmpdir)

        # SLURM_GPUS intentionally unset; only CUDA_VISIBLE_DEVICES present
        env = {k: v for k, v in os.environ.items() if k != "SLURM_GPUS"}
        env.update({
            "SLURM_JOBID": jobid,
            "CUDA_VISIBLE_DEVICES": "0,1,2",
            "THETA_HEALTH_API_ENDPOINT": "http://localhost:1",
        })
        result = subprocess.run(
            ["bash", str(PROLOG)], env=env, capture_output=True, text=True, timeout=30
        )

        try:
            assert result.returncode == 0, result.stderr
            metadata = (job_tmpdir / "metadata").read_text()
            assert "gpu_count=3" in metadata
        finally:
            if job_tmpdir.exists():
                shutil.rmtree(job_tmpdir)

    def test_prolog_exits_zero_with_no_gpu_vars(self):
        """Prolog must never block a job even with no GPU env vars at all."""
        jobid = "test_prolog_nogpu"
        job_tmpdir = Path(f"/tmp/theta_job_{jobid}")
        if job_tmpdir.exists():
            shutil.rmtree(job_tmpdir)

        # Strip every GPU-related var
        env = {k: v for k, v in os.environ.items()
               if k not in ("SLURM_GPUS", "SLURM_JOB_GPUS",
                            "GPU_DEVICE_ORDINAL", "CUDA_VISIBLE_DEVICES")}
        env.update({
            "SLURM_JOBID": jobid,
            "THETA_HEALTH_API_ENDPOINT": "http://localhost:1",
        })
        result = subprocess.run(
            ["bash", str(PROLOG)], env=env, capture_output=True, text=True, timeout=30
        )

        try:
            assert result.returncode == 0, result.stderr
            metadata = (job_tmpdir / "metadata").read_text()
            assert "gpu_count=0" in metadata
        finally:
            if job_tmpdir.exists():
                shutil.rmtree(job_tmpdir)

    def test_prolog_exits_zero_when_api_down(self):
        """Prolog must never fail a job because the health API is unreachable."""
        jobid = "test_prolog_apidown"
        job_tmpdir = Path(f"/tmp/theta_job_{jobid}")
        if job_tmpdir.exists():
            shutil.rmtree(job_tmpdir)

        result = _run(PROLOG, {
            "SLURM_JOBID": jobid,
            "SLURM_GPUS": "0",
            "THETA_HEALTH_API_ENDPOINT": "http://localhost:1",
        })

        try:
            assert result.returncode == 0
        finally:
            if job_tmpdir.exists():
                shutil.rmtree(job_tmpdir)


class TestEpilogHook:
    def test_full_prolog_epilog_roundtrip(self):
        """Prolog → epilog produces a valid JSON report and cleans up.

        This is the regression test for the bash-4-only `mapfile` bug: it
        actually executes the epilog, which `bash -n` cannot catch.
        """
        jobid = "test_roundtrip"
        job_tmpdir = Path(f"/tmp/theta_job_{jobid}")
        if job_tmpdir.exists():
            shutil.rmtree(job_tmpdir)

        with tempfile.TemporaryDirectory() as fake_home:
            env = {
                "SLURM_JOBID": jobid,
                "SLURM_GPUS": "0,1",
                "THETA_HEALTH_API_ENDPOINT": "http://localhost:1",
                "HOME": fake_home,  # report goes to $HOME/.theta
            }

            # 1. Prolog
            prolog_result = _run(PROLOG, env)
            assert prolog_result.returncode == 0, prolog_result.stderr
            assert job_tmpdir.exists(), "prolog did not create temp dir"

            # 2. Epilog
            epilog_result = _run(EPILOG, env)
            assert epilog_result.returncode == 0, epilog_result.stderr

            # 3. Report written and valid
            report_path = Path(fake_home) / ".theta" / f"job_{jobid}.json"
            assert report_path.exists(), "epilog did not write report"
            data = json.loads(report_path.read_text())
            assert data["jobid"] == jobid
            assert data["gpu_count"] == 2
            assert data["duration_sec"] >= 0
            assert data["end_time"] >= data["start_time"]

            # 4. Temp dir cleaned up
            assert not job_tmpdir.exists(), "epilog left temp dir behind"

    def test_epilog_exits_zero_with_no_prolog_metadata(self):
        """Epilog must exit 0 even if prolog never ran (job timeout / crash)."""
        jobid = "test_no_metadata"
        job_tmpdir = Path(f"/tmp/theta_job_{jobid}")
        if job_tmpdir.exists():
            shutil.rmtree(job_tmpdir)

        with tempfile.TemporaryDirectory() as fake_home:
            result = _run(EPILOG, {
                "SLURM_JOBID": jobid,
                "HOME": fake_home,
                "THETA_HEALTH_API_ENDPOINT": "http://localhost:1",
            })
            # No metadata → graceful no-op, never fail the node
            assert result.returncode == 0, result.stderr
