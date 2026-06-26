"""
Tests for the incident store — the flywheel substrate.

These pin the properties that make it a measurement device rather than a log:
  - the full lifecycle (open → action → recovery → label) persists and reloads,
  - a confirmed label is the ONLY thing that reaches confirmed_cause,
  - accuracy is reported only over labeled incidents and never fabricated,
  - the daemon integration helper opens/closes incidents at the right moments.
"""

import json

from theta.agent.incident_store import (
    IncidentStore, update_from_diagnosis,
    STAGE_OPEN, STAGE_ACTIONED, STAGE_RESOLVED, STAGE_CONFIRMED,
)


def _store(tmp_path):
    # Deterministic ids so assertions don't depend on uuid randomness.
    counter = {"n": 0}
    def ids():
        counter["n"] += 1
        return f"inc{counter['n']}"
    return IncidentStore(tmp_path / "incidents.jsonl", id_factory=ids)


def _open(store, **over):
    kw = dict(
        gpu_uuid="A100#3", gpu_index=3, node="dgx-01",
        cause="tim_degradation", confidence=0.74, tier="probable",
        headline="GPU 3: TIM aging", features_before={"rtheta_k_sigma": 4.8},
        now=100.0,
    )
    kw.update(over)
    return store.open_incident(**kw)


def test_full_lifecycle_and_stages(tmp_path):
    store = _store(tmp_path)
    inc = _open(store)
    assert inc.stage == STAGE_OPEN
    assert inc.effective_tier == "probable"

    store.record_action(inc.id, "reseated_cold_plate", now=200.0)
    assert store.get(inc.id).stage == STAGE_ACTIONED

    store.record_recovery(inc.id, {"rtheta_k_sigma": 0.6}, now=300.0)
    got = store.get(inc.id)
    assert got.stage == STAGE_RESOLVED
    assert got.resolved is True
    assert got.features_after == {"rtheta_k_sigma": 0.6}
    # Resolved but unlabeled does NOT reach confirmed_cause.
    assert got.effective_tier == "probable"

    store.label(inc.id, "contact_pressure", now=400.0, notes="repaste fixed it")
    got = store.get(inc.id)
    assert got.stage == STAGE_CONFIRMED
    assert got.effective_tier == "confirmed_cause"   # label is the only path here
    assert got.notes == "repaste fixed it"


def test_label_drives_correctness_and_accuracy(tmp_path):
    store = _store(tmp_path)
    a = _open(store, cause="tim_degradation")
    b = _open(store, gpu_uuid="A100#4", gpu_index=4, cause="dust_accumulation")

    store.record_recovery(a.id, {}, now=200.0)
    store.record_recovery(b.id, {}, now=200.0)

    # No labels yet → accuracy must be unearned, never a fabricated number.
    assert store.accuracy() == {"labeled": 0, "correct": 0, "rate": None}

    store.label(a.id, "tim_degradation", now=300.0)   # Theta was right
    store.label(b.id, "airflow_blockage", now=300.0)  # Theta was wrong

    assert store.get(a.id).prediction_was_correct is True
    assert store.get(b.id).prediction_was_correct is False
    acc = store.accuracy()
    assert acc == {"labeled": 2, "correct": 1, "rate": 0.5}


def test_persistence_reloads_across_instances(tmp_path):
    path = tmp_path / "incidents.jsonl"
    s1 = IncidentStore(path, id_factory=lambda: "fixed1")
    s1.open_incident(
        gpu_uuid="H100#0", gpu_index=0, node="n1", cause="fabric_link",
        confidence=0.7, tier="high", headline="link errors",
        features_before={}, now=10.0,
    )
    s1.record_recovery("fixed1", {}, now=20.0)

    # A fresh instance reads the same file back.
    s2 = IncidentStore(path)
    inc = s2.get("fixed1")
    assert inc is not None and inc.cause == "fabric_link" and inc.resolved

    # File is valid JSONL, one object per line.
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "fixed1"


def test_corrupt_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "incidents.jsonl"
    path.write_text('{"id": "ok", "gpu_uuid": "x", "gpu_index": 0, "node": "n", '
                    '"opened_at": 1.0, "cause": "nominal", "confidence": 0.0, '
                    '"tier": "unconfirmed", "headline": "h"}\n'
                    'THIS IS NOT JSON\n')
    store = IncidentStore(path)              # must not raise
    assert store.get("ok") is not None
    assert len(store.all()) == 1


def test_open_for_gpu_returns_only_unresolved(tmp_path):
    store = _store(tmp_path)
    a = _open(store)
    assert store.open_for_gpu("A100#3").id == a.id
    store.record_recovery(a.id, {}, now=200.0)
    assert store.open_for_gpu("A100#3") is None


# ── daemon integration helper ────────────────────────────────────────────────

def _causal(cause, tier, urgency, conf=0.7):
    return {
        "headline": f"GPU x: {cause}",
        "urgency": urgency,
        "tier": tier,
        "hypothesis": {"cause": cause, "confidence": conf, "one_line": ""},
    }


def test_update_opens_then_closes_one_incident(tmp_path):
    store = _store(tmp_path)
    args = dict(gpu_uuid="A100#3", gpu_index=3, node="dgx-01",
                features={"rtheta_k_sigma": 4.0})

    # Healthy → nothing tracked.
    assert update_from_diagnosis(
        store, causal_dict=_causal("nominal", "unconfirmed", "info"), now=1.0, **args
    ) is None
    assert store.all() == []

    # Crosses the bar → opens exactly one incident.
    inc = update_from_diagnosis(
        store, causal_dict=_causal("tim_degradation", "probable", "act_soon"), now=2.0, **args
    )
    assert inc is not None
    # Still failing next cycle → does NOT open a duplicate.
    again = update_from_diagnosis(
        store, causal_dict=_causal("tim_degradation", "probable", "act_soon"), now=3.0, **args
    )
    assert again is None
    assert len(store.all()) == 1

    # Returns to health → the open incident is resolved.
    closed = update_from_diagnosis(
        store, causal_dict=_causal("nominal", "unconfirmed", "info"),
        features={"rtheta_k_sigma": 0.5}, gpu_uuid="A100#3", gpu_index=3, node="dgx-01", now=4.0,
    )
    assert closed is not None and closed.resolved
    assert closed.features_after == {"rtheta_k_sigma": 0.5}


def test_update_ignores_subthreshold_diagnoses(tmp_path):
    store = _store(tmp_path)
    # Unconfirmed + low urgency should not litter the store.
    update_from_diagnosis(
        store, gpu_uuid="g", gpu_index=0, node="n",
        causal_dict=_causal("dust_accumulation", "unconfirmed", "info"),
        features={}, now=1.0,
    )
    assert store.all() == []
