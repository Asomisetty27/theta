"""
Hardware abstraction layer (HAL) for telemetry collectors.

Audit finding addressed: collector.py is hardcoded to pynvml/NVIDIA, with
no abstraction. AMD MI300 / Intel Gaudi / future TPU support requires
rewriting the collector each time, or worse: silently falling back to
demo mode and pretending everything is fine.

This module defines a `TelemetryCollector` protocol that any vendor can
implement, plus a `select_collector()` factory that auto-detects which
backend(s) are available on the host. The existing NVMLCollector now
implements this protocol (no behavior change for NVIDIA users), and a
stub ROCmCollector is provided as the AMD path — currently raising
NotImplementedError with a clear migration message, but architected so a
real implementation can drop in without touching the daemon.

The point of building this NOW, before the AMD implementation exists, is
that the daemon and downstream modules can be written to the protocol
today — so when ROCm support arrives, integration is a one-line factory
change, not a refactor of every module that touches RawSample.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .metrics import RawSample


@runtime_checkable
class TelemetryCollector(Protocol):
    """Protocol every per-vendor collector must satisfy.

    All methods are async — even the trivially-synchronous ones — so the
    daemon can `await` uniformly regardless of whether the underlying
    library blocks (pynvml does), is awaitable (some Intel/ROCm tools do),
    or runs in a subprocess. Synchronous implementations can wrap with
    `asyncio.to_thread`.

    The lifecycle is intentionally async-context-manager shaped because
    real hardware libraries need explicit init/shutdown (pynvml's nvmlInit
    grabs a process-wide lock; ROCm's rocm_smi_lib requires explicit
    init+shutdown calls; subprocess-based collectors need pipe cleanup).
    """

    vendor: str   # e.g. "nvidia", "amd", "intel", "demo"

    async def __aenter__(self) -> "TelemetryCollector": ...
    async def __aexit__(self, *_) -> None: ...

    async def collect_all(self) -> list[RawSample]:
        """One sample per monitored GPU, concurrent where possible."""
        ...

    @property
    def gpu_count(self) -> int:
        """Number of GPUs this collector is monitoring."""
        ...

    @property
    def gpu_names(self) -> list[str]:
        """Friendly model names indexed by GPU slot (for hw_profiles lookup)."""
        ...


# ──────────────────────────────────────────────────────────────────────────
# Backend availability probes
# ──────────────────────────────────────────────────────────────────────────

def _nvml_available() -> bool:
    """Is pynvml importable AND able to talk to a driver?"""
    try:
        import pynvml
        try:
            pynvml.nvmlInit()
            try:
                pynvml.nvmlDeviceGetCount()
                return True
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except pynvml.NVMLError:
            return False
    except ImportError:
        return False


def _rocm_available() -> bool:
    """Is amdsmi importable AND able to talk to a driver?"""
    try:
        import amdsmi
    except ImportError:
        return False
    try:
        amdsmi.amdsmi_init()
        try:
            return len(amdsmi.amdsmi_get_processor_handles()) > 0
        finally:
            try:
                amdsmi.amdsmi_shut_down()
            except Exception:
                pass
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────
# AMD ROCm collector — real implementation in rocm_collector.py
# ──────────────────────────────────────────────────────────────────────────
# (Imported lazily inside select_collector so NVIDIA-only hosts never import
# amdsmi. Implemented against amdsmi; hardware-validation on real MI300 silicon
# is the open item — see rocm_collector.py module docstring.)


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────

def select_collector(config, *, prefer: str | None = None):
    """Auto-detect and return the best available collector.

    Detection order (unless `prefer` overrides):
      1. NVIDIA (pynvml) — production-grade, fully implemented
      2. AMD (rocm_smi) — stub for now, falls through to demo if unimplemented
      3. Demo mode — synthetic samples, used in CI and for site-only deploys

    `prefer` can be "nvidia" | "amd" | "demo" to force a specific backend.
    Useful for testing the AMD code path on an NVIDIA host (will raise
    NotImplementedError loudly, which is correct).
    """
    # Lazy imports to avoid pulling pynvml on AMD-only hosts and vice versa
    from .collector import NVMLCollector

    def _rocm():
        from .rocm_collector import ROCmCollector
        return ROCmCollector(config)

    if prefer == "nvidia":
        return NVMLCollector(config)
    if prefer == "amd":
        return _rocm()
    if prefer == "demo":
        # NVMLCollector's demo mode is the canonical fake-data source
        coll = NVMLCollector(config)
        coll._demo_mode = True  # type: ignore[attr-defined]
        return coll

    # Auto-detect
    if _nvml_available():
        return NVMLCollector(config)
    if _rocm_available():
        return _rocm()
    # Fall back to demo mode
    return NVMLCollector(config)
