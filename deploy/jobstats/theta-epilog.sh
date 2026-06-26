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

# Best-effort: an epilog hook must NEVER fail the node over telemetry. We do
# not use `set -e` — every step degrades gracefully and the script always
# exits 0. `set -u` is also avoided so unset SLURM vars fall back to defaults.
set -o pipefail 2>/dev/null || true

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

# Parse metadata — portable to bash 3.2 (no mapfile / associative arrays).
# Metadata is simple "key=value" lines written by the prolog.
start_time="0"
node="unknown"
gpu_count="0"
while IFS='=' read -r key value; do
    case "$key" in
        start_time) start_time="$value" ;;
        node)       node="$value" ;;
        gpu_count)  gpu_count="$value" ;;
    esac
done < "$JOB_TMPDIR/metadata"

# Query final state from Theta health API (best-effort)
final_metrics=""
if command -v curl &>/dev/null; then
    final_metrics=$(curl -s --max-time 2 \
        "${HEALTH_API_ENDPOINT}/api/v1/metrics" 2>/dev/null || echo "{}")
fi

# Compute duration. Use awk (POSIX, always present) with %.3f so the result
# is always valid JSON — `bc` emits ".045" (no leading zero) for sub-second
# values, which is NOT valid JSON and corrupts every consumer of the report.
duration=$(awk "BEGIN { printf \"%.3f\", ${TIMESTAMP:-0} - ${start_time:-0} }" 2>/dev/null)
case "$duration" in
    ''|*[!0-9.-]*) duration="0" ;;   # fall back if awk produced nothing sane
esac

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
