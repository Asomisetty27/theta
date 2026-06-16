"""
Tests for MIG/vGPU device capability detection (theta/agent/device_caps.py).

The classification logic is pure, so the correctness rules are testable without
MIG/vGPU hardware: R_θ is per-physical-die under MIG, uncomputable when a vGPU
guest can't read temp/power, and only "computable" when both are readable.
"""
from theta.agent.device_caps import (
    classify_device, probe_capability, DeviceMode, DeviceCapability,
)


def test_plain_physical_gpu_is_computable():
    c = classify_device(mig_enabled=False, is_vgpu=False,
                        temp_readable=True, power_readable=True)
    assert c.mode is DeviceMode.PHYSICAL
    assert c.rtheta_computable is True


def test_mig_is_per_physical_die():
    c = classify_device(mig_enabled=True, is_vgpu=False,
                        temp_readable=True, power_readable=True)
    assert c.mode is DeviceMode.MIG
    assert c.rtheta_computable is True              # temp/power are physical → valid
    assert "physical" in c.note.lower()             # labeled as per-die, not per-instance


def test_vgpu_without_telemetry_is_not_computable():
    c = classify_device(mig_enabled=False, is_vgpu=True,
                        temp_readable=False, power_readable=False)
    assert c.mode is DeviceMode.VGPU
    assert c.rtheta_computable is False
    assert "unavailable" in c.note.lower()


def test_vgpu_with_passthrough_telemetry_is_computable():
    # Some vGPU/passthrough setups DO expose telemetry — then R_θ is fine.
    c = classify_device(mig_enabled=False, is_vgpu=True,
                        temp_readable=True, power_readable=True)
    assert c.mode is DeviceMode.VGPU
    assert c.rtheta_computable is True


def test_physical_missing_power_is_not_computable():
    c = classify_device(mig_enabled=False, is_vgpu=False,
                        temp_readable=True, power_readable=False)
    assert c.rtheta_computable is False
    assert "power" in c.note


# ── probe against a fake pynvml ───────────────────────────────────────────────

class _FakeNVML:
    NVML_TEMPERATURE_GPU = 0
    def __init__(self, mig=0, virt=0, temp=65, power=300_000):
        self._mig, self._virt, self._temp, self._power = mig, virt, temp, power
    def nvmlDeviceGetMigMode(self, h): return (self._mig, self._mig)
    def nvmlDeviceGetVirtualizationMode(self, h): return self._virt
    def nvmlDeviceGetTemperature(self, h, kind):
        if self._temp is None: raise RuntimeError("not supported")
        return self._temp
    def nvmlDeviceGetPowerUsage(self, h):
        if self._power is None: raise RuntimeError("not supported")
        return self._power


def test_probe_physical():
    c = probe_capability(_FakeNVML(), object())
    assert c.mode is DeviceMode.PHYSICAL and c.rtheta_computable


def test_probe_mig():
    c = probe_capability(_FakeNVML(mig=1), object())
    assert c.mode is DeviceMode.MIG and c.rtheta_computable


def test_probe_vgpu_guest_no_telemetry():
    # virtualization mode 2 (VGPU), power not exposed (0 → unreadable).
    c = probe_capability(_FakeNVML(virt=2, power=0), object())
    assert c.mode is DeviceMode.VGPU
    assert c.rtheta_computable is False


def test_probe_handles_missing_mig_api():
    class NoMig(_FakeNVML):
        def nvmlDeviceGetMigMode(self, h): raise RuntimeError("unsupported")
        def nvmlDeviceGetVirtualizationMode(self, h): raise RuntimeError("unsupported")
    c = probe_capability(NoMig(), object())
    assert c.mode is DeviceMode.PHYSICAL        # degrades gracefully
    assert c.rtheta_computable is True
