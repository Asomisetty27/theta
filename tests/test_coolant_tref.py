"""
Tests for live BMC coolant-inlet T_ref wiring (the B200 liquid-cooled fix).

Liquid-cooled GPUs (t_ref_strategy='coolant_inlet', e.g. DGX H100/B200) can't
learn T_ref from an idle window — T_j_idle ≈ coolant temp, so ΔT falls below the
noise floor and idle R_θ is invalid. They must use the coolant inlet as T_ref.

Before this fix the daemon collected the Redfish inlet temp but never fed it into
T_ref, so those GPUs fell back to the *static* profile expected_ambient_c — any
real coolant drift (spec 20 °C vs actual 25 °C) silently biased every R_θ on the
node. These tests pin that the live reading now drives T_ref for liquid-cooled
GPUs, and that air-cooled GPUs are left on their idle-window baseline.
"""

import pytest
from prometheus_client import REGISTRY

from theta.agent.baseline import BaselineManager
from theta.agent.daemon import ThetaAgent, AgentConfig

B200 = "NVIDIA B200 SXM6"      # coolant_inlet, expected_ambient 20 °C
T4 = "Tesla T4"                # idle_window, expected_ambient 22 °C


@pytest.fixture
def clean_prometheus():
    for c in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(c)
        except Exception:
            pass
    yield
    for c in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(c)
        except Exception:
            pass


class TestBaselineCoolantOverride:
    def test_live_coolant_overrides_static_profile_prior(self, tmp_path):
        """A live BMC reading replaces the static expected_ambient_c prior."""
        bm = BaselineManager(_file=tmp_path / "baselines.json")
        # Cold start: seed the static profile prior (B200 → 20 °C)
        bm.seed_from_profile(0, B200)
        assert bm.get_t_ref(0, B200) == pytest.approx(20.0)

        # Live coolant actually runs at 25 °C → T_ref should track it
        bm.set_external_ambient(0, 25.0, source="coolant_inlet_bmc")
        assert bm.get_t_ref(0, B200) == pytest.approx(25.0)

    def test_coolant_tref_bias_correction_is_material(self):
        """The 5 °C correction materially changes R_θ (this is the whole point)."""
        from theta.agent.metrics import compute_rtheta
        t_j, power = 74.0, 900.0   # B200 under load

        r_static, _ = compute_rtheta(t_j, 20.0, power)   # stale spec T_ref
        r_live, _ = compute_rtheta(t_j, 25.0, power)      # live coolant T_ref
        # 5 °C of T_ref error is a large fraction of the junction-to-coolant ΔT
        assert abs(r_static - r_live) / r_live > 0.08

    def test_external_ambient_does_not_clobber_locked_baseline(self, tmp_path):
        """A hard-locked (measured) baseline is never overwritten by a BMC prior."""
        bm = BaselineManager(_file=tmp_path / "baselines.json")
        # Simulate an air-cooled GPU that locked a real idle window
        bm.set_external_ambient(0, 22.0)             # provisional
        # Force a non-provisional lock via the idle-window path proxy:
        b = bm.get_baseline(0)
        b.provisional = False                         # pretend it's measured
        bm.set_external_ambient(0, 30.0)             # should be ignored
        assert bm.get_t_ref(0) == pytest.approx(22.0)


class TestDaemonCoolantWiring:
    def test_apply_coolant_tref_updates_only_liquid_cooled(self, clean_prometheus, tmp_path):
        """_apply_coolant_tref overrides T_ref for B200 but not for T4."""
        agent = ThetaAgent(AgentConfig())
        agent._baseline = BaselineManager(_file=tmp_path / "baselines.json")
        # Mixed node: gpu0 = liquid-cooled B200, gpu1 = air-cooled T4
        agent._collector_gpu_names = {0: B200, 1: T4}
        agent._baseline.seed_from_profile(0, B200)   # 20 °C prior
        agent._baseline.seed_from_profile(1, T4)     # 22 °C prior

        # Live coolant inlet reads 26 °C
        agent._apply_coolant_tref(26.0, ts=1000.0)

        # B200 tracks the live coolant; T4 keeps its idle-window prior
        assert agent._baseline.get_t_ref(0, B200) == pytest.approx(26.0)
        assert agent._baseline.get_t_ref(1, T4) == pytest.approx(22.0)

    def test_apply_coolant_tref_marks_source(self, clean_prometheus, tmp_path):
        """The override is tagged so it's distinguishable in baseline state."""
        agent = ThetaAgent(AgentConfig())
        agent._baseline = BaselineManager(_file=tmp_path / "baselines.json")
        agent._collector_gpu_names = {0: B200}
        agent._baseline.seed_from_profile(0, B200)
        agent._apply_coolant_tref(24.0, ts=1000.0)

        b = agent._baseline.get_baseline(0)
        assert b.source == "coolant_inlet_bmc"
        assert b.provisional is True   # still upgradeable, widens alert bands
