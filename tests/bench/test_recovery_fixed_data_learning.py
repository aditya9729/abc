"""Focused CPU boundaries with explicit data/transform doubles; no models or teacher data."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from abc_bench import fixed_data_learning as fixed
from abc_bench import realtime_update_probe as probe
from abc_bench import recovery_fixed_data_learning as recovery


def save_json(path, value):
    path.write_text(json.dumps(value))
    return probe.file_pin(path)


@pytest.fixture
def admission_config(tmp_path):
    norm = {"path": "/explicit-fixture/norm", "bytes": 3, "sha256": "fake"}
    record = {
        "schema": "sadhana.recovery_expo_admission/1",
        "owner": "/root",
        "status": "accepted_complete_recovery_episode_for_ingestion",
        "artifact_accepted": True,
        "training_authorized": False,
        "reward_view": recovery.ORIGINAL,
        "normalization_manifest": norm,
        "task_config": norm,
        "origin_group": "synthetic-fixture-only",
    }
    config = {
        "phase": recovery.PHASE,
        "fixed_data": {
            "ingestion_admission": save_json(tmp_path / "admission.json", record),
            "condition": "plus_accepted_correction",
        },
    }
    return config, {"normalization_manifest": norm, "task_config": norm}, record


def test_closed_dispatch_preserves_legacy():
    assert probe.fixed_phase_module({"phase": "fixed_data_10"}) is fixed
    assert probe.fixed_phase_module({"phase": recovery.PHASE}) is recovery
    for value in (None, "os", "arbitrary.module", "fixed_data_11"):
        with pytest.raises(ValueError, match="Unknown"):
            probe.fixed_phase_module({"phase": value})
    assert fixed.PHASE == "fixed_data_10"
    assert (
        fixed.SPEC_SHA256
        == "9e42561f6fe20549fe126017bef0ee142552839ce22c324a72317323d52c45df"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "teacher_refused"),
        ("schema", "sadhana.recovery_expo_admission/3"),
        ("schema", None),
        ("artifact_accepted", False),
        ("owner", "not-root"),
        ("training_authorized", True),
        ("reward_view", "strict_teacher_sparse_terminal_v1"),
        ("normalization_manifest", {}),
        ("task_config", {}),
    ],
)
def test_missing_partial_or_wrong_view_refused_before_adapter(
    admission_config, tmp_path, field, value
):
    config, spec, record = admission_config
    record[field] = value
    config["fixed_data"]["ingestion_admission"] = save_json(
        tmp_path / "bad.json", record
    )
    with pytest.raises(ValueError):
        recovery.admission(config, spec)


def test_admission_requires_bytes_and_nonnull_pin(admission_config, tmp_path):
    config, spec, record = admission_config
    assert recovery.admission(config, spec) == record
    path = tmp_path / "admission.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        recovery.admission(config, spec)
    config["fixed_data"]["ingestion_admission"] = None
    with pytest.raises(ValueError, match="real pinned"):
        recovery.admission(config, spec)


def qualified_release_fixture(config, spec, tmp_path, monkeypatch):
    payloads = {}
    for name in (recovery.ADAPTER, recovery.CONNECTED_ADAPTER):
        path = tmp_path / (name.replace(".", "/") + ".py")
        path.parent.mkdir(exist_ok=True)
        path.write_text("# Explicit payload double, never imported: " + name + "\n")
        payloads[name] = probe.file_pin(path)
    proof = save_json(tmp_path / "proof.json", {"fixture": "synthetic CPU receipt"})
    identity = save_json(
        tmp_path / "identity.json",
        {
            "source_wheel_installed_match": True,
            "files": list(payloads.values()),
        },
    )
    release = {
        "status": "installed_CPU_fixtures_qualified_only",
        "installed_tests_passed": 55,
        "installed_identity": identity,
        **{
            key: proof
            for key in (
                "source_manifest",
                "parent_review",
                "derivation_AST_equality",
                "wheel",
                "test_result",
                "test_xml",
            )
        },
    }
    spec.update(
        phase=recovery.PHASE,
        update_calls=10,
        measure_at=[0, 1, 10],
        adapter_cpu_cases=55,
        adapter_release=save_json(tmp_path / "release.json", release),
    )
    spec_pin = save_json(tmp_path / "spec.json", spec)
    monkeypatch.setattr(recovery, "SPEC_SHA256", spec_pin["sha256"])
    installed = payloads[recovery.ADAPTER]
    config["fixed_data"].update(spec=spec_pin, python=installed)
    config["installed_modules"] = {
        recovery.MODULE: installed,
        fixed.MODULE: installed,
        **payloads,
    }
    return payloads, release


def test_spec_pins_release_payload_and_separates_conditions(
    admission_config, tmp_path, monkeypatch
):
    config, spec, _ = admission_config
    payloads, _ = qualified_release_fixture(config, spec, tmp_path, monkeypatch)
    for condition in recovery.CONDITIONS:
        config["fixed_data"]["condition"] = condition
        assert recovery.load_spec(config) == spec
    config["fixed_data"]["condition"] = "unbounded"
    with pytest.raises(ValueError, match="condition"):
        recovery.load_spec(config)
    config["fixed_data"]["condition"] = recovery.CONDITIONS[0]
    config["installed_modules"][recovery.ADAPTER] = {
        **payloads[recovery.ADAPTER],
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="payload"):
        recovery.load_spec(config)


class FakeReplay:
    """Only storage used by this caller check; not a substitute replay implementation."""

    def __init__(self):
        self._replan_steps = 8
        self.size = recovery.BASE_ROWS
        self.dataset_dict = {
            "is_success": np.r_[np.ones(7069, bool), np.zeros(9899, bool)]
        }

    def __len__(self):
        return self.size


def new_processor():
    def unstarted():
        yield "explicit fixture batch"

    return SimpleNamespace(
        replay_buffer=FakeReplay(),
        offline_replay_buffer=[],
        batch_size=64,
        utd_ratio=20,
        offline_ratio=0,
        actor_success_only=True,
        use_dagger_hil_sampling=False,
        offline_iterator=None,
        hil_iterator=None,
        replay_iterator=unstarted(),
    )


def transforms():
    # Deliberately nonidentity transforms make raw/requested action misuse visible.
    def normalize(raw):
        assert set(raw) == {"state", "head_left", "prompt", "actions"}
        result = dict(raw)
        result["state"] = raw["state"] * 3 - 2
        result["actions"] = raw["actions"] * 2 + 1
        return result

    def pad(raw):
        return {
            **raw,
            "actions": np.pad(raw["actions"], ((0, 0), (0, 18))),
            "state": np.pad(raw["state"], (0, 18)),
        }

    return SimpleNamespace(input_transforms=normalize, input_model_transforms=pad)


class FakeCorrection:
    """144 synthetic transitions, including128 failed-prefix rows; not acceptance evidence."""

    def __init__(self, pin, record):
        self.admission_pin = pin
        self.admission = record
        self.clipped_prefix_controls = 10
        self.first_historical_terminal = 144
        self.full_controls = 160

    def __len__(self):
        return 144

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {
            "observations": {
                "state": np.arange(14, dtype=np.float32),
                "head_left": np.zeros((1, 1, 3), np.uint8),
                "prompt": "fixture",
            },
            "actions": np.full(14, index / 1000, np.float32),
        }


@pytest.fixture
def prepared_seam(admission_config, monkeypatch):
    config, spec, record = admission_config
    processor = new_processor()
    actor = transforms()
    events = []

    def load(pin):
        events.append("authenticated-complete-dataset")
        return FakeCorrection(pin, record)

    def append(buffer, correction, *, expected_rows):
        assert expected_rows == recovery.BASE_ROWS and not hasattr(buffer, "rng")
        events.append("append-after-restore-before-sampling")
        n = expected_rows + len(correction)
        buffer.dataset_dict["is_success"] = np.r_[
            buffer.dataset_dict["is_success"], np.arange(len(correction)) >= 128
        ]
        buffer.dataset_dict["actions"] = np.zeros((n, 1, 32), np.float32)
        buffer.dataset_dict["state"] = np.zeros((n, 32), np.float32)
        for i in range(len(correction)):
            row = correction[i]
            raw = copy.deepcopy(row["observations"])
            raw["actions"] = np.tile(row["actions"][None], (16, 1))
            norm = actor.input_model_transforms(actor.input_transforms(raw))
            buffer.dataset_dict["actions"][expected_rows + i, 0] = norm["actions"][0]
            buffer.dataset_dict["state"][expected_rows + i] = norm["state"]
        buffer.size = n
        return {"fixture_only": True, "rows_appended": len(correction)}

    adapter = SimpleNamespace(RecoveryTransitions=load, append_after_restore=append)
    monkeypatch.setattr(probe, "checked_import", lambda name, cfg: adapter)
    return config, spec, processor, SimpleNamespace(actor=actor), events, adapter


@pytest.mark.parametrize("condition", recovery.CONDITIONS)
def test_prepare_exact_old_vs_appended_pool_without_sampling(prepared_seam, condition):
    config, spec, processor, agent, events, _ = prepared_seam
    config["fixed_data"]["condition"] = condition
    report = recovery.prepare_replay(agent, processor, config, spec)
    pools = {r["source"]: r for r in report["source_pools"]}
    assert events[0] == "authenticated-complete-dataset"
    assert pools["historical_demo"]["actor_sampleable_starts"] == 7069
    assert pools["historical_online_timeout"]["actor_sampleable_starts"] == 0
    assert report["clipped_policy_prefix_controls"] == 10
    assert report["consumed_source_coverage"] is None
    assert report["sample_indices_exposed_by_original_sampler"] is False
    if condition == "plus_accepted_correction":
        assert len(events) == 2 and report["rows_after"] == 17112
        assert pools["correction_policy_prefix"]["actor_sampleable_starts"] == 0
        assert pools["correction_teacher_suffix"]["actor_sampleable_starts"] == 8
        assert report["normalization_check"]["rows_checked"] == 144
    else:
        assert len(events) == 1 and report["rows_after"] == 16968
        assert report["normalization_check"] is None and report["append"] is None
    assert not hasattr(processor.replay_buffer, "rng")


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 32),
        ("utd_ratio", 1),
        ("offline_ratio", 0.5),
        ("actor_success_only", False),
        ("use_dagger_hil_sampling", True),
        ("hil_iterator", object()),
        ("offline_iterator", object()),
    ],
)
def test_routing_refused_before_dataset_load(prepared_seam, field, value):
    config, spec, processor, agent, events, _ = prepared_seam
    setattr(processor, field, value)
    with pytest.raises(ValueError):
        recovery.prepare_replay(agent, processor, config, spec)
    assert not events


@pytest.mark.parametrize("mutation", ["sampled", "iterator", "wrong_rows"])
def test_bad_append_order_refused(prepared_seam, mutation):
    config, spec, processor, agent, events, _ = prepared_seam
    if mutation == "sampled":
        processor.replay_buffer.rng = "already sampled"
    elif mutation == "iterator":
        next(processor.replay_iterator)
    else:
        processor.replay_buffer.size -= 1
    with pytest.raises(ValueError):
        recovery.prepare_replay(agent, processor, config, spec)
    assert not events


@pytest.mark.parametrize(
    "mutation", ["raw_actions", "privileged", "prefix_positive", "append_failure"]
)
def test_transform_label_and_partial_append_refusals(prepared_seam, mutation):
    config, spec, processor, agent, _, adapter = prepared_seam
    original = adapter.append_after_restore

    def bad_append(buffer, correction, *, expected_rows):
        result = original(buffer, correction, expected_rows=expected_rows)
        if mutation == "raw_actions":
            buffer.dataset_dict["actions"][expected_rows, 0, 0] += 1
        elif mutation == "prefix_positive":
            buffer.dataset_dict["is_success"][expected_rows] = True
        elif mutation == "privileged":
            base = correction.__class__.__getitem__

            class Privileged(FakeCorrection):
                def __getitem__(self, index):
                    row = base(self, index)
                    row["observations"]["source_control"] = index
                    return row

            correction.__class__ = Privileged
        else:
            raise OSError("explicit partial append failure")
        return result

    adapter.append_after_restore = bad_append
    with pytest.raises((ValueError, OSError)):
        recovery.prepare_replay(agent, processor, config, spec)
    assert not hasattr(processor.replay_buffer, "rng")


def test_new_entry_reuses_shared_run_and_lifetime_holder(monkeypatch):
    observed = {}
    args = (
        ["one-use-agent"],
        "processor",
        "old-dataset",
        "task",
        {},
        "output",
        "opts",
        "params",
        0,
    )

    def common(*received, **kwargs):
        observed.update(args=received, kwargs=kwargs)
        return "explicit common-run double"

    monkeypatch.setattr(
        probe, "checked_import", lambda name, cfg: SimpleNamespace(run=common)
    )
    assert recovery.run(*args) == "explicit common-run double"
    assert observed["args"] == args
    assert observed["kwargs"] == {"recovery": recovery}


def test_final_authentication_rechecks_complete_frame_inventory(
    admission_config, tmp_path, monkeypatch
):
    config, spec, record = admission_config
    payload = tmp_path / "frame.fixture"
    payload.write_bytes(b"explicit immutable frame fixture")
    frame_pin = probe.file_pin(payload)
    inventory = save_json(tmp_path / "inventory.json", {"files": [frame_pin]})
    for key in (
        "prefix_acceptance",
        "prefix_result",
        "teacher_result",
        "producer_source_manifest",
        "producer_release",
        "origin_policy_evidence",
        "strict_contract_source",
        "task_config",
        "normalization_manifest",
    ):
        record[key] = inventory
    spec.update(task_config=inventory, normalization_manifest=inventory)
    record["qualification_evidence"] = [inventory]
    config["fixed_data"]["ingestion_admission"] = save_json(
        tmp_path / "complete.json", record
    )
    monkeypatch.setattr(recovery, "load_spec", lambda cfg: spec)
    monkeypatch.setattr(fixed, "authenticate_data", lambda cfg, spec: {"files": []})
    result = recovery.authenticate_data(config)
    assert result["recovery_inputs"]["authenticated_file_references"] == 12
    payload.write_bytes(b"changed after append")
    with pytest.raises(ValueError, match="identity differs"):
        recovery.authenticate_data(config)


@pytest.mark.parametrize("condition", recovery.CONDITIONS)
def test_connected_dispatch_keeps_original_append_and_routing(
    prepared_seam, admission_config, tmp_path, monkeypatch, condition
):
    config, spec, processor, agent, events, adapter = prepared_seam
    _, _, record = admission_config
    record["schema"] = recovery.ADMISSION_SCHEMAS[1]
    config["fixed_data"].update(
        condition=condition,
        ingestion_admission=save_json(tmp_path / "connected-admission.json", record),
    )
    calls = []

    def connected_load(pin):
        calls.append(pin)
        return adapter.RecoveryTransitions(pin)

    connected = SimpleNamespace(ConnectedRecoveryTransitions=connected_load)

    def checked_import(name, cfg):
        assert cfg is config
        assert name in (recovery.ADAPTER, recovery.CONNECTED_ADAPTER)
        return adapter if name == recovery.ADAPTER else connected

    monkeypatch.setattr(probe, "checked_import", checked_import)
    result = recovery.prepare_replay(agent, processor, config, spec)
    assert calls == [config["fixed_data"]["ingestion_admission"]]
    assert events[0] == "authenticated-complete-dataset"
    assert result["rows_after"] == (
        17112 if condition == "plus_accepted_correction" else 16968
    )
    assert result["consumed_source_coverage"] is None
    assert result["strict_terminal_reward_view_executed"] is False
    if condition == "plus_accepted_correction":
        assert events[-1] == "append-after-restore-before-sampling"
        pools = {row["source"]: row for row in result["source_pools"]}
        assert pools["correction_policy_prefix"]["actor_sampleable_starts"] == 0
        assert pools["correction_teacher_suffix"]["actor_sampleable_starts"] == 8


@pytest.mark.parametrize("condition", recovery.CONDITIONS)
def test_connected_closure_checked_before_and_after_without_relabelling(
    admission_config, tmp_path, monkeypatch, condition
):
    config, spec, record = admission_config
    record["schema"] = recovery.ADMISSION_SCHEMAS[1]
    config["fixed_data"].update(
        condition=condition,
        ingestion_admission=save_json(tmp_path / "admission2.json", record),
    )
    inputs = [config["fixed_data"]["ingestion_admission"]]
    for name in (
        "policy_capture",
        "failed_E8",
        "accepted_verification",
        "current_frames",
        "full_replay",
    ):
        inputs.append(save_json(tmp_path / (name + ".json"), {"fixture": name}))
    calls = []

    def closure(pin):
        calls.append(pin)
        return copy.deepcopy(inputs)

    monkeypatch.setattr(recovery, "load_spec", lambda cfg: spec)
    monkeypatch.setattr(fixed, "authenticate_data", lambda cfg, spec: {"files": []})
    monkeypatch.setattr(
        probe,
        "checked_import",
        lambda name, cfg: (
            SimpleNamespace(connected_input_pins=closure)
            if name == recovery.CONNECTED_ADAPTER
            else None
        ),
    )
    first = recovery.authenticate_data(config)["recovery_inputs"]
    second = recovery.authenticate_data(config)["recovery_inputs"]
    assert first == second and first["input_files"] == inputs
    assert first["admission_schema"] == recovery.ADMISSION_SCHEMAS[1]
    assert calls == [config["fixed_data"]["ingestion_admission"]] * 2
    (tmp_path / "failed_E8.json").write_text("changed historical provenance")
    with pytest.raises(ValueError, match="identity differs"):
        recovery.authenticate_data(config)


@pytest.mark.parametrize(
    "change",
    [
        "missing_connected",
        "changed_connected",
        "missing_identity",
        "duplicate_identity",
        "changed_cpu_receipt",
    ],
)
def test_actual_release_closure_refusals(
    admission_config, tmp_path, monkeypatch, change
):
    config, spec, _ = admission_config
    payloads, release = qualified_release_fixture(config, spec, tmp_path, monkeypatch)
    if change == "missing_connected":
        del config["installed_modules"][recovery.CONNECTED_ADAPTER]
    elif change == "changed_connected":
        config["installed_modules"][recovery.CONNECTED_ADAPTER] = {
            **payloads[recovery.CONNECTED_ADAPTER],
            "sha256": "0" * 64,
        }
    elif change == "changed_cpu_receipt":
        (tmp_path / "proof.json").write_text("changed qualified receipt")
    else:
        records = [payloads[recovery.ADAPTER]]
        if change == "duplicate_identity":
            records += [payloads[recovery.CONNECTED_ADAPTER]] * 2
        release["installed_identity"] = save_json(
            tmp_path / "identity.json",
            {
                "source_wheel_installed_match": True,
                "files": records,
            },
        )
        spec["adapter_release"] = save_json(tmp_path / "release.json", release)
        config["fixed_data"]["spec"] = save_json(tmp_path / "spec.json", spec)
        monkeypatch.setattr(
            recovery, "SPEC_SHA256", config["fixed_data"]["spec"]["sha256"]
        )
    with pytest.raises(ValueError):
        recovery.load_spec(config)
