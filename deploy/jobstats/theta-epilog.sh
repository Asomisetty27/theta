#!/bin/bash
#
# SLURM Epilog Hook for Theta Job Tracking
#
# Executed by SLURM after job completes. Reads the prolog-written metadata,
# queries Theta's health API for current state, computes per-job thermal
# report (R_theta delta, energy, thermal stress), and writes to
# ~/.theta/job_<JOBID>.json
#
# Environment variables provided by SLURM:
#   SLURM_JOBID — job ID
#   SLURM_NODEID — node index (0-based)
#   SLURM_JOB_EXIT_CODE — exit code
#   SLURM_JOB_NUM_NODES — number of allocated nodes
#

set -euo pipefail

JOBID="${SLURM_JOBID:?job_id_required}"
TIMESTAMP="$(date -u +%s.%N)"
JOB_TMPDIR="/tmp/theta_job_${JOBID}"
THETA_JOB_REPORTS="${HOME}/.theta"

# Health API endpoint (same as prolog)
HEALTH_API_ENDPOINT="${THETA_HEALTH_API_ENDPOINT:-http://localhost:7777}"

# Ensure output directory exists
mkdir -p "$THETA_JOB_REPORTS"

# Read metadata written by prolog
if [[ ! -f "$JOB_TMPDIR/metadata" ]]; then
    echo "[Theta] Job $JOBID epilog — no prolog metadata found (job may have timed out)" >&2
    exit 0
fi

# Parse metadata
mapfile -t metadata < "$JOB_TMPDIR/metadata"
declare -A meta
for line in "${metadata[@]}"; do
    key="${line%=*}"
    value="${line#*=}"
    meta["$key"]="$value"
done

start_time="${meta[start_time]:-0}"
node="${meta[node]:-unknown}"
gpu_count="${meta[gpu_count]:-0}"

# Query final state from Theta health API (best-effort)
final_metrics=""
if command -v curl &>/dev/null; then
    final_metrics=$(curl -s --max-time 2 \
        "${HEALTH_API_ENDPOINT}/api/v1/metrics" 2>/dev/null || echo "{}")
fi

# Compute duration
duration=$(echo "$TIMESTAMP - $start_time" | bc -l 2>/dev/null || echo "0")

# Write job report (JSON format matching job_tracker.py output)
# This is a simplified version; the full report is written by daemon.py's
# _process_job_completions when it detects the job has ended.
report_file="$THETA_JOB_REPORTS/job_${JOBID}.json"
{
    cat <<EOF
{
  "jobid": "$JOBID",
  "node": "$node",
  "start_time": $start_time,
  "end_time": $TIMESTAMP,
  "duration_sec": $duration,
  "gpu_count": $gpu_count,
  "epilog_timestamp": $TIMESTAMP,
  "note": "Per-GPU R_theta and energy tracking is done in-daemon; this epilog file records job timing. Full thermal report written by daemon's _process_job_completions."
}
EOF
} > "$report_file" 2>/dev/null || true

# Clean up temp files
rm -rf "$JOB_TMPDIR" 2>/dev/null || true

# Log for debugging
echo "[Theta] Job $JOBID epilog — duration=${duration}s, wrote report to $report_file" >&2

exit 0
