"""
Device capability detection — MIG and vGPU awareness.

Two real-fleet configurations break naive per-device R_θ if ignored:

* **MIG (Multi-Instance GPU).** An A100/H100 can be partitioned into up to 7
  compute instances. But temperature and power are properties of the PHYSICAL
  die — there is one heatsink, one thermal path, shared by every instance. So
  R_θ is meaningful only at the physical-GPU level; computing it "per MIG
  instance" would attribute one die's thermals to N instances and is wrong.
  NVML's device count enumerates PHYSICAL GPUs even with MIG on (instances are a
  separate API), so monitoring physical handles is already correct — this module
  makes that explicit and labels it, rather than leaving it to luck.

* **vGPU (virtualized guest).** Inside a VM, NVML frequently cannot read junction
  temperature or power (the host doesn't expose them to the guest). R_θ = ΔT/P is
  then uncomputable, and emitting a fabricated value would be worse than silence.
  The agent must DETECT this and say "telemetry unavailable here," not guess.

`classify_device` is pure logic over probe results (no NVML import), so the rules
are unit-testable on any machine. `probe_capability` runs the actual NVML calls
defensively (any call may be unsupported on a given driver/SKU) and feeds them in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class DeviceMode(str, Enum):
    PHYSICAL = "physical"   # ordinary full GPU
    MIG      = "mig"        # MIG-enabled — temp/power are physical-die, shared
    VGPU     = "vgpu"       # virtualized guest — telemetry may be unavailable
    UNKNOWN  = "unknown"


@dataclass
class DeviceCapability:
    mode:               DeviceMode
    temp_readable:      bool
    power_readable:     bool
    rtheta_computable:  bool          # need BOTH temp and power
    note:               str

    def as_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "temp_readable": self.temp_readable,
            "power_readable": self.power_readable,
            "rtheta_computable": self.rtheta_computable,
            "note": self.note,
        }


def classify_device(
    *,
    mig_enabled: bool,
    is_vgpu: bool,
    temp_readable: bool,
    power_readable: bool,
) -> DeviceCapability:
    """Pure classification from probe booleans. Order matters: vGPU telemetry
    gaps and MIG sharing are both about what R_θ *means*, not just whether the
    call returns a number."""
    computable = temp_readable and power_readable

    if is_vgpu:
        note = ("vGPU guest — R_θ computable" if computable else
                "vGPU guest — temperature/power not exposed to the guest; "
                "R_θ unavailable, thermal monitoring inactive for this device")
        return DeviceCapability(DeviceMode.VGPU, temp_readable, power_readable,
                                computable, note)

    if mig_enabled:
        note = ("MIG enabled — R_θ is per PHYSICAL GPU (shared across all MIG "
                "instances on this die), not per-instance"
                if computable else
                "MIG enabled but temperature/power unreadable — R_θ unavailable")
        return DeviceCapability(DeviceMode.MIG, temp_readable, power_readable,
                                computable, note)

    if not computable:
        missing = ", ".join(
            x for x, ok in (("temperature", temp_readable), ("power", power_readable)) if not ok
        )
        return DeviceCapability(DeviceMode.PHYSICAL, temp_readable, power_readable,
                                False, f"telemetry incomplete ({missing} unreadable) — "
                                       f"R_θ unavailable for this device")

    return DeviceCapability(DeviceMode.PHYSICAL, True, True, True, "physical GPU — full telemetry")


def _readable(call: Callable) -> bool:
    """True iff the NVML call returns a finite, usable value (not an exception,
    None, or a sentinel like 0 power which guests often report)."""
    try:
        v = call()
    except Exception:
        return False
    return v is not None


def probe_capability(pynvml, handle) -> DeviceCapability:
    """Run the NVML probes for one device handle, defensively. Each call is
    optional — driver/SKU/permission differences mean any may be unsupported."""
    # MIG mode: nvmlDeviceGetMigMode returns (current, pending); current==1 → on.
    mig_enabled = False
    try:
        current, _pending = pynvml.nvmlDeviceGetMigMode(handle)
        mig_enabled = (current == 1)
    except Exception:
        mig_enabled = False

    # vGPU: a virtualization-mode query; HOST/NONE are bare metal, others virtual.
    is_vgpu = False
    try:
        mode = pynvml.nvmlDeviceGetVirtualizationMode(handle)
        # NVML_GPU_VIRTUALIZATION_MODE_NONE=0, PASSTHROUGH=1, VGPU=2, HOST_VGPU=3, HOST_VSGA=4
        is_vgpu = mode in (2, 3, 4)
    except Exception:
        is_vgpu = False

    temp_ok = _readable(lambda: pynvml.nvmlDeviceGetTemperature(
        handle, pynvml.NVML_TEMPERATURE_GPU))
    # Power: 0 from a guest is "not exposed"; treat <=0 as unreadable.
    def _power():
        p = pynvml.nvmlDeviceGetPowerUsage(handle)
        return p if (p is not None and p > 0) else None
    power_ok = _readable(_power)

    return classify_device(mig_enabled=mig_enabled, is_vgpu=is_vgpu,
                           temp_readable=temp_ok, power_readable=power_ok)
