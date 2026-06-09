"""
B200 baseline data collector — runs without root, no daemon required.

Usage:
  pip install nvidia-ml-py
  python collect_b200_baseline.py

  # With known coolant inlet temperature (from BMC / `ipmitool` / facility spec):
  python collect_b200_baseline.py --coolant-c 20.5

  # Run for 2 hours, saving to a specific file:
  python collect_b200_baseline.py --duration 7200 --out b200_baseline.csv

What this collects (one row per GPU per second):
  timestamp, gpu_index, gpu_name, temp_junction_c, power_w, util_pct, pstate,
  clock_sm_mhz, mem_util_pct, throttle_reasons, coolant_inlet_c

Why we need this:
  Theta's B200 thermal profiles are physics estimates, not measurements.
  Two hours of data covering idle + a training run gives us the real R_theta
  operating range and validates whether the classifier thresholds are sensible.
  This is E005 in the ThermalOS Stage 2 validation protocol.

No data leaves this machine unless you explicitly send the CSV file.
The script is read-only — it queries NVML and writes a local file.
"""

import argparse
import csv
import signal
import sys
import time

try:
    import pynvml as nv
except ImportError:
    print("ERROR: pynvml not installed. Run: pip install nvidia-ml-py")
    sys.exit(1)


FIELDS = [
    "timestamp_s",
    "gpu_index",
    "gpu_name",
    "temp_junction_c",
    "power_w",
    "util_pct",
    "mem_util_pct",
    "pstate",
    "clock_sm_mhz",
    "clock_mem_mhz",
    "throttle_reasons",
    "coolant_inlet_c",   # blank if not provided
]

_running = True


def _sigint(sig, frame):
    global _running
    _running = False
    print("\n[interrupted] Flushing and closing CSV…")


def _get_gpu_name(handle) -> str:
    try:
        name = nv.nvmlDeviceGetName(handle)
        return name.decode() if isinstance(name, bytes) else name
    except Exception:
        return "unknown"


def collect(
    out_path: str,
    interval_s: float = 1.0,
    duration_s: float = 7200.0,
    coolant_c: float | None = None,
    gpu_indices: list[int] | None = None,
):
    nv.nvmlInit()
    n_gpus = nv.nvmlDeviceGetCount()
    indices = gpu_indices or list(range(n_gpus))
    handles = {i: nv.nvmlDeviceGetHandleByIndex(i) for i in indices}
    names   = {i: _get_gpu_name(h) for i, h in handles.items()}

    print(f"\n  Collecting from {len(handles)} GPU(s):")
    for i, name in names.items():
        print(f"    GPU {i}: {name}")
    print(f"  Interval:  {interval_s}s")
    print(f"  Duration:  {duration_s / 60:.0f} min")
    print(f"  Output:    {out_path}")
    if coolant_c is not None:
        print(f"  Coolant:   {coolant_c}°C (supplied)")
    else:
        print(f"  Coolant:   not supplied — add BMC reading to CSV manually if available")
    print(f"\n  [Ctrl+C to stop early]\n")

    signal.signal(signal.SIGINT, _sigint)

    deadline = time.time() + duration_s
    rows_written = 0

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        while _running and time.time() < deadline:
            t0 = time.time()
            for idx, handle in handles.items():
                try:
                    temp   = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
                    power  = nv.nvmlDeviceGetPowerUsage(handle) / 1000.0   # mW → W
                    util   = nv.nvmlDeviceGetUtilizationRates(handle)
                    pstate = nv.nvmlDeviceGetPerformanceState(handle)
                    sm_clk = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)
                    mm_clk = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM)
                    try:
                        throttle = nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                    except Exception:
                        throttle = 0
                    row = {
                        "timestamp_s":    round(t0, 3),
                        "gpu_index":      idx,
                        "gpu_name":       names[idx],
                        "temp_junction_c": temp,
                        "power_w":        round(power, 2),
                        "util_pct":       util.gpu,
                        "mem_util_pct":   util.memory,
                        "pstate":         pstate,
                        "clock_sm_mhz":   sm_clk,
                        "clock_mem_mhz":  mm_clk,
                        "throttle_reasons": throttle,
                        "coolant_inlet_c": coolant_c if coolant_c is not None else "",
                    }
                    w.writerow(row)
                    rows_written += 1
                except Exception as e:
                    print(f"  [warn] GPU {idx}: {e}")

            f.flush()
            elapsed = time.time() - t0
            remaining = interval_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

            elapsed_total = time.time() - (deadline - duration_s)
            if int(elapsed_total) % 300 < 1:   # status every ~5 min
                print(f"  {elapsed_total/60:.0f}m elapsed — {rows_written} rows written")

    nv.nvmlShutdown()
    print(f"\n  Done. {rows_written} rows → {out_path}")
    print(f"\n  Send this file to the ThermalOS team for Stage 2 calibration.")
    print(f"  It will NOT be shared further without your consent.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="B200 baseline telemetry collector for ThermalOS Stage 2")
    p.add_argument("--out",       default="b200_baseline.csv", help="Output CSV path")
    p.add_argument("--duration",  type=float, default=7200.0,  help="Collection duration in seconds (default: 7200 = 2h)")
    p.add_argument("--interval",  type=float, default=1.0,     help="Sample interval in seconds (default: 1.0)")
    p.add_argument("--coolant-c", type=float, default=None,    dest="coolant_c",
                   help="Coolant inlet temperature °C (from BMC / facility spec)")
    p.add_argument("--gpus",      type=str,   default=None,
                   help="Comma-separated GPU indices to monitor (default: all)")
    args = p.parse_args()

    gpu_list = None
    if args.gpus:
        gpu_list = [int(x.strip()) for x in args.gpus.split(",")]

    collect(
        out_path   = args.out,
        interval_s = args.interval,
        duration_s = args.duration,
        coolant_c  = args.coolant_c,
        gpu_indices = gpu_list,
    )
