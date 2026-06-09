"""
Per-GPU R_θ profile learner.

Accumulates steady-state load samples and signals when enough clean data
exists to upgrade a hardware profile from 'extrapolated' to 'measured'
confidence — closing the loop between deployment experience and the
hw_profiles.py priors.

Integration:
    learner = ProfileLearner()
    # In _process_sample, on each stable load window:
    learner.update(gpu, name, window.rtheta_mean, power_w, window.is_stable)
    # Every 100 ticks, check:
    for gpu in gpu_indices:
        m = learner.ready_to_upgrade(gpu)
        if m:
            log.info("profile_upgrade_ready", **m.as_log_dict())
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

LOAD_POWER_W  = 200.0   # W   — minimum power to count as "under load"
UPGRADE_N     = 500     # samples before suggesting upgrade
UPGRADE_STD   = 0.005   # C/W — max std for a "clean" measurement
WARN_MULT     = 1.20    # warn threshold = mean × 1.20
CRIT_MULT     = 1.40    # critical threshold = mean × 1.40


@dataclass
class ProfileMeasurement:
    gpu_index:       int
    gpu_name:        str
    n_samples:       int
    rtheta_mean:     float
    rtheta_std:      float
    warn_threshold:  float
    crit_threshold:  float

    def as_log_dict(self) -> dict:
        return {
            "gpu":          self.gpu_index,
            "name":         self.gpu_name,
            "n":            self.n_samples,
            "rtheta_mean":  self.rtheta_mean,
            "rtheta_std":   self.rtheta_std,
            "warn":         self.warn_threshold,
            "critical":     self.crit_threshold,
        }

    def hw_profiles_suggestion(self) -> str:
        lines = [
            f"# Profile upgrade ready for GPU {self.gpu_index} ({self.gpu_name})",
            f"# Measured from {self.n_samples} steady-state load windows.",
            f"# Update hw_profiles.py entry for this GPU family:",
            f"    rtheta_expected_under_load = {self.rtheta_mean},",
            f"    rtheta_expected_idle       = {self.rtheta_mean},  # flat — liquid-cooled",
            f"    rtheta_load_threshold      = {self.warn_threshold},",
            f"    rtheta_idle_threshold      = {self.warn_threshold},",
            f"    confidence                 = 'measured',  # was: extrapolated",
            f"# Then run: theta calibrate --gpu {self.gpu_index}",
        ]
        return "\n".join(lines)


class ProfileLearner:
    """
    Tracks per-GPU R_θ load samples and fires a one-time upgrade signal
    once a GPU has accumulated enough clean steady-state data.
    """

    def __init__(self) -> None:
        self._load_samples: dict[int, deque] = defaultdict(lambda: deque(maxlen=1500))
        self._gpu_names: dict[int, str] = {}
        self._alerted: set[int] = set()

    def update(
        self,
        gpu_index: int,
        gpu_name: str,
        rtheta_mean: float,
        power_w: float,
        is_stable: bool,
    ) -> None:
        if not is_stable or power_w < LOAD_POWER_W:
            return
        if math.isnan(rtheta_mean) or rtheta_mean <= 0:
            return
        self._gpu_names[gpu_index] = gpu_name
        self._load_samples[gpu_index].append(rtheta_mean)

    def ready_to_upgrade(self, gpu_index: int) -> ProfileMeasurement | None:
        """
        Returns a ProfileMeasurement if this GPU has enough clean load data
        to suggest a hw_profiles.py confidence upgrade to 'measured'.
        Fires at most once per GPU per daemon lifetime.
        """
        if gpu_index in self._alerted:
            return None
        samples = list(self._load_samples[gpu_index])
        if len(samples) < UPGRADE_N:
            return None
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)
        std = math.sqrt(variance)
        if std > UPGRADE_STD:
            return None  # too noisy — wait for more stable operating period
        self._alerted.add(gpu_index)
        return ProfileMeasurement(
            gpu_index      = gpu_index,
            gpu_name       = self._gpu_names.get(gpu_index, "unknown"),
            n_samples      = len(samples),
            rtheta_mean    = round(mean, 4),
            rtheta_std     = round(std, 5),
            warn_threshold = round(mean * WARN_MULT, 4),
            crit_threshold = round(mean * CRIT_MULT, 4),
        )

    def sample_count(self, gpu_index: int) -> int:
        return len(self._load_samples[gpu_index])
