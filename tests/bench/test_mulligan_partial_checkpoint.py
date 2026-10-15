"""CPU admission fixtures with synthetic files, never original model execution."""

import json
from pathlib import Path

import pytest

from abc_bench import mulligan_r1lite_diagnostic as diagnostic
from abc_bench import mulligan_r1lite_eval as evaluation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return evaluation.pin(path)


@pytest.fixture
def partial(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "WORKSPACE", tmp_path)
    run_id = "author-subset-explicit-cpu-fixture"
    run = tmp_path / run_id
    root = run / "training/checkpoints/idql_fixture/checkpoints/checkpoint_10000"
    root.mkdir(parents=True)
    write(root / "metadata.json", {"step": 10000, "policy_type": "idql"})
    write(root / "stats.json", {"fixture": "No model or normalizer loaded"})
    (root / "policy.pt").write_bytes(b"Explicit synthetic payload, never unpickled")
    marker = write(tmp_path / "evidence.json", {"fixture": True})
    manifest = write(
        tmp_path / "inspector-manifest.json", {"files": [], "inputs": [marker]}
    )
    monkeypatch.setattr(
        evaluation, "PERIODIC_INSPECTOR_MANIFEST", Path(manifest["path"])
    )
    monkeypatch.setattr(evaluation, "PERIODIC_INSPECTOR_SHA256", manifest["sha256"])
    bindings = {k: marker for k in ("author_manifest", "data", "task")}
    canonical = {
        "run_id": run_id,
        "status": "failed",
        "task": evaluation.TASK,
        "embodiment": evaluation.EMBODIMENT,
    }
    worker = {
        "status": "failed",
        "mode": evaluation.training.MODE,
        "stage": "idql",
        "task": evaluation.TASK,
        "embodiment": evaluation.EMBODIMENT,
    }
    cleanup = {
        "owner": "/root",
        "run_id": run_id,
        "all_owned_processes_stopped": True,
        "lease_released": True,
        "charges_preserved": True,
        "canonical": write(tmp_path / "runs" / f"{run_id}.json", canonical),
        "worker": write(run / "receipt.json", worker),
    }
    stop = {
        "owner": "/root",
        "run_id": run_id,
        "training_steps": 10000,
        "checkpoint_root": str(root),
        "status": "root_stopped_after_committed_periodic_checkpoint",
    }
    receipt = {
        "schema": "sadhana.mulligan_r1_periodic_checkpoint_acceptance/1",
        "owner": "/root",
        "status": "accepted_r1_idql_periodic_checkpoint",
        "artifact_accepted": True,
        "scope": "frozen_evaluation_only",
        "task": evaluation.TASK,
        "embodiment": evaluation.EMBODIMENT,
        "policy_type": "idql",
        "parent": None,
        "training_steps": 10000,
        "checkpoint_root": str(root),
        "checkpoint_files": [],
        "bindings": bindings,
        "producer_run_id": run_id,
        "producer_status": "failed",
        "evidence": [marker],
    }
    inspection = {
        "schema": "sadhana.mulligan_partial_idql_inspection/1",
        "status": "passed_partial_idql_checkpoint_inspection_pending_root_acceptance",
        "source_manifest": manifest,
        "scope": "frozen_evaluation_only",
        "artifact_accepted": False,
        "rolling_step": 10000,
        "rolling_completed": False,
        "periodic_equals_rolling_at_selected_step": True,
        "optimizer_at_selected_step_available": True,
        "original_rolling_rng_sampler_present": True,
        "finite_model_leaves": 1,
        "exact_resume_executed": False,
        "fixed_input_original_loader_probe": "equal_finite_action_q_v_for_seven_calls",
        "restore_probe_device": "cpu",
        "cuda_initialized": False,
        "optimizer_counters": {
            k: {"step": 10000, "parameter_states": 1}
            for k in ("actor", "critic", "value")
        },
    }
    config = {"method": "idql", "parent_acceptance": None}

    def seal(*, refresh_inventory=True, synchronize_inspection=True):
        receipt["producer_cleanup"] = write(tmp_path / "cleanup.json", cleanup)
        receipt["stop_evidence"] = write(tmp_path / "stop.json", stop)
        if refresh_inventory:
            receipt["checkpoint_files"] = [
                evaluation.pin(p) for p in sorted(root.iterdir())
            ]
        if synchronize_inspection:
            for key in (
                "producer_run_id",
                "bindings",
                "training_steps",
                "policy_type",
                "checkpoint_root",
                "checkpoint_files",
                "producer_cleanup",
                "stop_evidence",
            ):
                inspection[key] = receipt[key]
        receipt["inspection"] = write(tmp_path / "inspection.json", inspection)
        config["checkpoint_acceptance"] = write(tmp_path / "acceptance.json", receipt)

    seal()
    return {
        "config": config,
        "inputs": bindings,
        "receipt": receipt,
        "inspection": inspection,
        "cleanup": cleanup,
        "stop": stop,
        "root": root,
        "seal": seal,
    }


def test_partial_idql_admits_actual_step_without_final_model(partial):
    p = partial
    result = evaluation.accepted_checkpoint(p["config"], p["inputs"])
    assert result["training_steps"] == 10000
    assert Path(result["checkpoint_root"]).name == "checkpoint_10000"
    assert result["producer_status"] == "failed"


@pytest.mark.parametrize("step", [True, 0, -1, 10001, 150000, 150001, 10000.0])
def test_partial_step_is_strict_and_bounded(partial, step):
    partial["receipt"]["training_steps"] = step
    partial["seal"]()
    with pytest.raises(ValueError):
        evaluation.accepted_checkpoint(partial["config"], partial["inputs"])


@pytest.mark.parametrize(
    "damage",
    [
        "root_owner",
        "unaccepted",
        "resume_scope",
        "divl",
        "parent",
        "lineage",
        "metadata_step",
        "metadata_type",
        "directory_name",
        "changed_payload",
        "extra_file",
        "inspection_pending",
        "inspection_binding",
        "inspection_manifest",
        "finite_model",
        "restore_probe",
        "cuda",
        "resume_claim",
        "rolling_advanced",
        "rolling_completed",
        "model_rolling_mismatch",
        "optimizer_counter",
        "cleanup_missing",
        "live_children",
        "open_lease",
        "lost_charges",
        "cleanup_run",
        "stop_run",
        "stop_step",
        "stop_root",
        "canonical_running",
        "wrong_worker",
        "worker_running",
    ],
)
def test_partial_refuses_incomplete_or_mismatched_evidence(partial, damage):
    p = partial
    receipt, inspection, cleanup, stop = (
        p[k] for k in ("receipt", "inspection", "cleanup", "stop")
    )
    refresh, sync = True, True
    if damage == "root_owner":
        receipt["owner"] = "unaccepted-fixture"
    elif damage == "unaccepted":
        receipt["artifact_accepted"] = False
    elif damage == "resume_scope":
        receipt["scope"] = "training_resume"
    elif damage == "divl":
        p["config"]["method"] = receipt["policy_type"] = "divl"
    elif damage == "parent":
        receipt["parent"] = p["config"]["checkpoint_acceptance"]
    elif damage == "lineage":
        receipt["bindings"] = {**receipt["bindings"], "task": {}}
    elif damage.startswith("metadata_"):
        write(
            p["root"] / "metadata.json",
            {
                "step": 9999 if damage == "metadata_step" else 10000,
                "policy_type": "divl" if damage == "metadata_type" else "idql",
            },
        )
    elif damage == "directory_name":
        receipt["checkpoint_root"] = str(p["root"].parent)
    elif damage == "changed_payload":
        (p["root"] / "policy.pt").write_bytes(b"Changed after inspected inventory")
        refresh = False
    elif damage == "extra_file":
        (p["root"] / "extra").write_bytes(b"Not an original three-file checkpoint")
    elif damage == "inspection_pending":
        inspection["status"] = "pending"
    elif damage == "inspection_binding":
        inspection["training_steps"] = 5000
        sync = False
    elif damage == "inspection_manifest":
        inspection["source_manifest"] = {
            **inspection["source_manifest"],
            "sha256": "0" * 64,
        }
    elif damage == "finite_model":
        inspection["finite_model_leaves"] = 0
    elif damage == "restore_probe":
        inspection["fixed_input_original_loader_probe"] = "unexecuted"
    elif damage == "cuda":
        inspection["cuda_initialized"] = True
    elif damage == "resume_claim":
        inspection["exact_resume_executed"] = True
    elif damage == "rolling_advanced":
        inspection["rolling_step"] = 15000
    elif damage == "rolling_completed":
        inspection["rolling_completed"] = True
    elif damage == "model_rolling_mismatch":
        inspection["periodic_equals_rolling_at_selected_step"] = False
    elif damage == "optimizer_counter":
        inspection["optimizer_counters"]["critic"]["step"] = 9999
    elif damage == "cleanup_missing":
        cleanup.pop("all_owned_processes_stopped")
    elif damage == "live_children":
        cleanup["all_owned_processes_stopped"] = False
    elif damage == "open_lease":
        cleanup["lease_released"] = False
    elif damage == "lost_charges":
        cleanup["charges_preserved"] = False
    elif damage == "cleanup_run":
        cleanup["run_id"] = "unrelated"
    elif damage == "stop_run":
        stop["run_id"] = "unrelated"
    elif damage == "stop_step":
        stop["training_steps"] = 5000
    elif damage == "stop_root":
        stop["checkpoint_root"] = str(p["root"].parent)
    else:
        key = "canonical" if damage == "canonical_running" else "worker"
        path = Path(cleanup[key]["path"])
        value = json.loads(path.read_text())
        value["stage" if damage == "wrong_worker" else "status"] = (
            "divl" if damage == "wrong_worker" else "running"
        )
        cleanup[key] = write(path, value)
    p["seal"](refresh_inventory=refresh, synchronize_inspection=sync)
    with pytest.raises(ValueError):
        evaluation.accepted_checkpoint(p["config"], p["inputs"])


def artifact_only(p, rolling_step):
    p["receipt"]["scope"] = "frozen_policy_artifact_only"
    p["inspection"].update(
        scope="frozen_policy_artifact_only",
        rolling_step=rolling_step,
        optimizer_at_selected_step_available=False,
        optimizer_counters={},
        periodic_equals_rolling_at_selected_step=None,
        original_rolling_rng_sampler_present=None,
    )
    p["stop"].update(
        training_steps=rolling_step,
        checkpoint_root=str(p["root"].parent / f"checkpoint_{rolling_step}"),
    )
    p["seal"]()


@pytest.mark.parametrize("rolling_step", [10000, 20000, 140000])
def test_policy_only_can_select_old_policy_without_optimizer_at_x(
    partial, rolling_step
):
    artifact_only(partial, rolling_step)
    result = evaluation.accepted_checkpoint(partial["config"], partial["inputs"])
    assert result["training_steps"] == 10000
    assert result["scope"] == "frozen_policy_artifact_only"
    assert partial["inspection"]["optimizer_counters"] == {}


@pytest.mark.parametrize(
    "damage",
    [
        "optimizer_available",
        "optimizer_values",
        "model_equality",
        "rng_present",
        "missing_equality",
        "missing_rng",
        "rolling_before_x",
        "completed_ceiling",
        "nonperiodic_stop",
        "stop_at_x",
        "scope_mismatch",
    ],
)
def test_policy_only_rejects_attribution_or_stop_mismatch(partial, damage):
    p = partial
    artifact_only(p, 20000)
    inspection = p["inspection"]
    if damage == "optimizer_available":
        inspection["optimizer_at_selected_step_available"] = True
    elif damage == "optimizer_values":
        inspection["optimizer_counters"] = {"critic": {"step": 20000}}
    elif damage == "model_equality":
        inspection["periodic_equals_rolling_at_selected_step"] = True
    elif damage == "rng_present":
        inspection["original_rolling_rng_sampler_present"] = True
    elif damage == "missing_equality":
        del inspection["periodic_equals_rolling_at_selected_step"]
    elif damage == "missing_rng":
        del inspection["original_rolling_rng_sampler_present"]
    elif damage == "rolling_before_x":
        inspection["rolling_step"] = 5000
    elif damage == "completed_ceiling":
        inspection["rolling_step"] = 150000
    elif damage == "nonperiodic_stop":
        inspection["rolling_step"] = 25000
    elif damage == "stop_at_x":
        p["stop"].update(training_steps=10000, checkpoint_root=str(p["root"]))
    elif damage == "scope_mismatch":
        inspection["scope"] = "frozen_evaluation_only"
    p["seal"]()
    with pytest.raises(ValueError):
        evaluation.accepted_checkpoint(p["config"], p["inputs"])


def test_partial_idql_cannot_become_divl_parent(partial, tmp_path):
    p = partial
    parent = p["config"]["checkpoint_acceptance"]
    root = tmp_path / "divl/final_model"
    root.mkdir(parents=True)
    for name in ("metadata.json", "stats.json", "policy.pt"):
        (root / name).write_text("Explicit original final-branch fixture")
    child = {
        "status": "accepted_r1_divl_checkpoint",
        "task": evaluation.TASK,
        "embodiment": evaluation.EMBODIMENT,
        "training_steps": 150000,
        "policy_type": "divl",
        "bindings": p["inputs"],
        "checkpoint_root": str(root),
        "checkpoint_files": [evaluation.pin(x) for x in sorted(root.iterdir())],
        "evidence": p["receipt"]["evidence"],
        "parent": parent,
    }
    config = {
        "method": "divl",
        "parent_acceptance": parent,
        "checkpoint_acceptance": write(tmp_path / "child.json", child),
    }
    with pytest.raises(ValueError, match="150000-step IDQL parent"):
        evaluation.accepted_checkpoint(config, p["inputs"])
