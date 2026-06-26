#!/bin/bash
#
# SLURM Prolog Hook for Theta Job Tracking
#
# Executed by SLURM before job starts. Records job metadata and queries
# current R_theta baseline from Theta's health API for each assigned GPU.
#
# Environment variables provided by SLURM:
#   SLURM_JOBID — job ID
#   SLURM_NODEID — node index (0-based)
#   SLURM_JOB_NODEID — job's view of node index
#   SLURM_GPUS — GPU indices assigned to this job (comma-separated or colon-delimited)
#   SLURM_LOCALID — local task ID
#   SLURM_STEP_ID — step ID (0 for job prolog)
#

set -euo pipefail

JOBID="${SLURM_JOBID:?job_id_required}"
NODE="$(hostname -s)"
TIMESTAMP="$(date -u +%s.%N)"

# Temporary working directory for this job's state
JOB_TMPDIR="/tmp/theta_job_${JOBID}"
mkdir -p "$JOB_TMPDIR"

# Query current R_theta baseline from Theta's health API
# Default endpoint: localhost:7777 (can be overridden via env var)
HEALTH_API_ENDPOINT="${THETA_HEALTH_API_ENDPOINT:-http://localhost:7777}"

# Parse SLURM_GPUS into array
# SLURM_GPUS format varies: could be "0,1,2" or "0:1:2" or "[0-1]"
parse_gpus() {
    local gpu_str="${1:?}"
    # Handle bracket notation like [0-1] → 0,1
    if [[ "$gpu_str" =~ \[([0-9]+)-([0-9]+)\] ]]; then
        local start="${BASH_REMATCH[1]}"
        local end="${BASH_REMATCH[2]}"
        for ((i=start; i<=end; i++)); do
            echo "$i"
        done
    else
        # Replace colons with spaces for word splitting
        echo "$gpu_str" | tr ',:' ' '
    fi
}

# Fetch current R_theta baseline for each GPU from health API
get_baselines() {
    local gpu_array=("$@")

    # Try to fetch metrics from health API
    if ! metrics_json=$(curl -s --max-time 2 \
        "${HEALTH_API_ENDPOINT}/api/v1/metrics" 2>/dev/null); then
        echo "# Failed to contact Theta health API at ${HEALTH_API_ENDPOINT}" >&2
        return 1
    fi

    # Parse R_theta for each GPU
    for gpu_idx in "${gpu_array[@]}"; do
        if command -v jq &>/dev/null; then
            # Use jq if available
            r_theta=$(echo "$metrics_json" | jq -r \
                ".gpu_metrics[] | select(.gpu_index == $gpu_idx) | .rtheta_baseline // .rtheta // empty" 2>/dev/null || echo "null")
        else
            # Fallback: basic grep/sed (brittle but works for simple JSON)
            r_theta=$(grep -o "\"gpu_index\": $gpu_idx[^}]*\"rtheta[^:]*: [^,}]*" <<< "$metrics_json" | \
                      tail -1 | grep -o '[0-9.]*$' || echo "null")
        fi
        echo "$r_theta"
    done
}

# Main
GPU_ARRAY=($(parse_gpus "$SLURM_GPUS"))
echo "jobid=$JOBID" > "$JOB_TMPDIR/metadata"
echo "node=$NODE" >> "$JOB_TMPDIR/metadata"
echo "start_time=$TIMESTAMP" >> "$JOB_TMPDIR/metadata"
echo "gpu_count=${#GPU_ARRAY[@]}" >> "$JOB_TMPDIR/metadata"

# Write baselines to file (best-effort; failures don't block job)
{
    echo "gpu_baselines:"
    for idx in "${!GPU_ARRAY[@]}"; do
        gpu="${GPU_ARRAY[$idx]}"
        echo "  gpu_$gpu:"
        get_baselines "$gpu" | while read -r baseline; do
            echo "    r_theta_baseline: $baseline"
        done
    done
} > "$JOB_TMPDIR/baselines" 2>/dev/null || true

# Log for debugging (stdout goes to slurm log)
echo "[Theta] Job $JOBID prolog — GPUs=${GPU_ARRAY[*]} node=$NODE" >&2

exit 0
