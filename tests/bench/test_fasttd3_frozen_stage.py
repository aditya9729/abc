"""Installed metadata and lifecycle contracts with explicit CPU doubles only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from abc_bench import fasttd3_frozen_stage as stage


def write(path, value):
    path.write_text(json.dumps(value))
    return stage.artifact(path)


@pytest.fixture
def inputs(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"explicit opaque checkpoint double, not a trained actor")
    checkpoint_pin = stage.artifact(checkpoint)
    producer_host = write(
        tmp_path / "training-host.json", {"profile": "scratch_training128"}
    )
    worker = {
        "status": "completed",
        "phase": "original_fasttd3_abc_scratch_training4000",
        "metrics": stage.TRAIN_METRICS.copy(),
        "child_exit_codes": {"host": 0, "learner": 0},
        "host_config": producer_host,
        "input_config": write(
            tmp_path / "training-config.json", {"host_config": producer_host}
        ),
        "native_receipt": write(
            tmp_path / "training-native.json",
            {
                "status": "completed",
                "vector_steps": 4000,
                "transitions": 512000,
                "config": {"profile": "scratch_training128"},
                "worker_cleanup": [
                    {
                        "rank": rank,
                        "pid": 1900000000 + rank,
                        "alive": False,
                        "exitcode": 0,
                    }
                    for rank in range(128)
                ],
            },
        ),
        "learner_receipt": write(
            tmp_path / "training-learner.json",
            {
                "status": "completed",
                "controls": 4000,
                "actual_updates": {"critic": 7978, "actor": 3989},
                "checkpoint": checkpoint_pin,
            },
        ),
        "checkpoint": checkpoint_pin,
        "export_check": write(
            tmp_path / "export.json",
            {
                "global_step": 4000,
                "state_fingerprints": {
                    key: {"sha256": "a" * 64}
                    for key in (
                        "actor_state_dict",
                        "qnet_state_dict",
                        "qnet_target_state_dict",
                        "obs_normalizer_state",
                        "critic_obs_normalizer_state",
                    )
                },
            },
        ),
    }
    worker_pin = write(tmp_path / "training-worker.json", worker)
    parent_pin = write(
        tmp_path / "training-parent.json",
        {
            "status": "completed",
            "task": "put_plastic_bottles_in_bin",
            "embodiment": "native_yam",
            "worker_exit_code": 0,
            "worker_receipt_sha256": worker_pin["sha256"],
            "metrics": stage.TRAIN_METRICS.copy(),
        },
    )
    # Metadata fixture only. CapturedScene's full native gate is not replaced.
    episode_pin = write(
        tmp_path / "capture.json",
        {
            "episode_id": "test-val",
            "split": "validation",
            "seed": 9100,
        },
    )
    cohort_pin = write(
        tmp_path / "cohort.json",
        {
            "schema": "sadhana.abc.demonstration-cohort/v1",
            "splits": {"train": [], "validation": ["test-val"], "development": []},
            "episodes": [
                {
                    "manifest": episode_pin,
                    "worker_receipt": worker_pin,
                    "parent_receipt": parent_pin,
                }
            ],
        },
    )
    host = {
        "schema_version": 2,
        "stage": "evaluation",
        "host": {
            "num_envs": 1,
            "initial_seed": 9100,
            "allowed_seeds": [9100],
            "max_episode_steps": 3540,
            "save_video": True,
            "action_profile": "original_float32_no_host_clip_v1",
            "reset_profile": {
                "schema": "sadhana.abc.captured-reset/v1",
                "method": "captured_demonstration_mjb_integration_v1",
                "cohort": cohort_pin,
                "episode_id": "test-val",
            },
        },
        "timeouts": {"host_wall_s": 1800, "rpc_s": 180, "max_requests": 4000},
    }
    host_pin = write(tmp_path / "host.json", host)
    distribution = stage.importlib.metadata.distribution("abc-minimal")
    sources = {
        "abc": [
            {
                **stage.artifact(Path(distribution.locate_file(relative))),
                "path": relative,
            }
            for relative in sorted(stage.ABC_PAYLOADS)
        ],
        "fast_adapter": [
            {"path": relative, "bytes": 1, "sha256": "0" * 64}
            for relative in sorted(stage.FAST_PAYLOADS)
        ],
    }
    config = stage.prepare_config(
        Path(parent_pin["path"]),
        Path(worker_pin["path"]),
        Path(host_pin["path"]),
        sources,
    )
    path = tmp_path / "config.json"
    write(path, config)
    return path, config, host, worker


def proof_and_description(tmp_path):
    proof = {
        "schema": "sadhana.abc.captured-reset-verification/v1",
        "status": "completed",
        "reset_verified": True,
        "seed": 9100,
        "episode_id": "test-val",
        "actual_controls": 0,
        "physics_steps": 0,
        "sampler_calls": 0,
        "checks": {
            when: {
                key: {"equal": True}
                for key in (
                    "chronology",
                    "model",
                    "integration",
                    "qpos",
                    "qvel",
                    "state14",
                    "time",
                    "judge",
                )
            }
            for when in ("before_rgb", "after_rgb")
        },
        "camera_checks": {
            camera: {
                "equal": True,
                "actual_sha256": "b" * 64,
                "expected_sha256": "b" * 64,
            }
            for camera in ("left", "right", "top")
        },
    }
    proof_pin = write(tmp_path / "explicit-proof.json", proof)
    description = {
        "num_envs": 1,
        "num_obs": 216,
        "num_actions": 14,
        "max_episode_steps": 3540,
        "profile": "captured_validation",
        "autoreset": False,
        "privileged_state": True,
        "action_profile": "normalized14_affine_clip_v1",
        "embodiment": "native_yam",
        "reset_policy": {
            "reset_verified": True,
            "profile": "captured_demonstration_mjb_integration_v1",
            "heldout_seed": 9100,
            "allowed_seeds": [9100],
            "episode_id": "test-val",
            "verification": proof_pin,
        },
    }
    return proof, description


def result_doubles(tmp_path, output, config, *, success=False):
    output.mkdir(exist_ok=True)
    (output / "native_host").mkdir(exist_ok=True)
    (output / "learner").mkdir(exist_ok=True)
    _proof, description = proof_and_description(tmp_path)
    video = output / "native_host/episode.mp4"
    video.write_bytes(b"explicit opaque video double, not native camera proof")
    reset = description["reset_policy"]["verification"]
    judge = {
        "num_active_bottles": 5,
        "num_bottles_in_bin": 5 if success else 0,
        "success": success,
    }
    native = {
        "completed": True,
        "seed": 9100,
        "success": success,
        "final_native_task_eval": judge,
        "reset_verification": reset,
        "video": stage.artifact(video),
    }
    episode = {
        **native,
        "controls": 20 if success else 3540,
        "terminated": success,
        "truncated": not success,
    }
    host = {
        "status": "completed",
        "description": description,
        "actual_optimizer_updates": 0,
        "episodes": [native],
        "controls": episode["controls"],
        "reset_proofs": [reset],
    }
    learner = {
        "status": "completed",
        "checkpoint": config["checkpoint"],
        "evaluation_seeds": [9100],
        "episodes": [episode],
        "actor_updates": 0,
        "critic_updates": 0,
        "policy_state_before": "a" * 64,
        "policy_state_after": "a" * 64,
        "rng_state_before": {"torch_cpu": "b" * 64, "torch_policy_device": "c" * 64},
        "rng_state_after": {"torch_cpu": "b" * 64, "torch_policy_device": "c" * 64},
    }
    write(output / "native_host/receipt.json", host)
    write(output / "learner/receipt.json", learner)
    return host, learner


def test_preparation_and_metadata_config_bind_terminal_producer(inputs):
    path, config, host, worker = inputs
    assert stage.load_config(path) == (config, host, worker)


@pytest.mark.parametrize(
    "fault", ["partial", "parent_hash", "collector", "cleanup", "checkpoint"]
)
def test_producer_rejects_incomplete_or_changed_training(inputs, fault):
    _path, config, _host, worker = inputs
    parent_path = Path(config["training_parent"]["path"])
    parent = json.loads(parent_path.read_text())
    if fault == "partial":
        parent["metrics"]["vector_steps"] = 3999
    elif fault == "parent_hash":
        parent["worker_receipt_sha256"] = "0" * 64
    elif fault in ("collector", "cleanup"):
        native_path = Path(worker["native_receipt"]["path"])
        native = json.loads(native_path.read_text())
        if fault == "collector":
            native["transitions"] = 511872
        else:
            native["worker_cleanup"][0]["alive"] = True
        worker["native_receipt"] = write(native_path, native)
        config["training_worker"] = write(
            Path(config["training_worker"]["path"]), worker
        )
        parent["worker_receipt_sha256"] = config["training_worker"]["sha256"]
    else:
        Path(config["checkpoint"]["path"]).write_bytes(b"changed checkpoint bytes")
    config["training_parent"] = write(parent_path, parent)
    with pytest.raises(ValueError):
        stage.training_producer(config["training_parent"], config["training_worker"])


@pytest.mark.parametrize("fault", ["physics", "camera", "stepped", "identity"])
def test_reset_proof_rejects_wrong_initial_native_evidence(inputs, tmp_path, fault):
    proof, description = proof_and_description(tmp_path)
    if fault == "physics":
        proof["checks"]["after_rgb"]["integration"]["equal"] = False
    elif fault == "camera":
        proof["camera_checks"]["right"]["actual_sha256"] = "d" * 64
    elif fault == "stepped":
        proof["actual_controls"] = 1
    else:
        proof["seed"] = 9101
    description["reset_policy"]["verification"] = write(
        tmp_path / "explicit-proof.json", proof
    )
    with pytest.raises(ValueError):
        stage.check_description(description, inputs[2])


@pytest.mark.parametrize("success", [False, True])
def test_complete_episode_is_scored_from_native_judge_and_video(
    inputs, tmp_path, success
):
    output = tmp_path / "run"
    result_doubles(tmp_path, output, inputs[1], success=success)
    result = stage.evaluate_result(inputs[1], inputs[2], output)
    assert result["metrics"]["successes"] == int(success)
    assert result["metrics"]["bottles_in_bin"] == (5 if success else 0)
    assert result["metrics"]["actor_updates"] == 0


@pytest.mark.parametrize(
    "fault", ["actor", "rng", "missing_rng", "partial", "judge", "reset", "video"]
)
def test_frozen_result_rejects_mutation_partial_episode_or_missing_artifact(
    inputs, tmp_path, fault
):
    output = tmp_path / "run"
    host, learner = result_doubles(tmp_path, output, inputs[1])
    if fault == "actor":
        learner["policy_state_after"] = "d" * 64
    elif fault == "rng":
        learner["rng_state_after"]["torch_cpu"] = "d" * 64
    elif fault == "missing_rng":
        learner.pop("rng_state_before")
        learner.pop("rng_state_after")
    elif fault == "partial":
        learner["episodes"][0]["controls"] = 3539
        host["controls"] = 3539
    elif fault == "judge":
        learner["episodes"][0]["final_native_task_eval"] = {"num_active_bottles": 4}
    elif fault == "reset":
        host["reset_proofs"] = []
    else:
        Path(host["episodes"][0]["video"]["path"]).unlink()
    write(output / "native_host/receipt.json", host)
    write(output / "learner/receipt.json", learner)
    with pytest.raises((RuntimeError, ValueError)):
        stage.evaluate_result(inputs[1], inputs[2], output)


def test_lease_refusal_precedes_mutation(inputs, tmp_path, monkeypatch):
    def reject():
        raise RuntimeError("No canonical lease")

    monkeypatch.setattr(stage, "require_subset_lease", reject)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="No canonical lease"):
        stage.run(inputs[0], output)
    assert not output.exists()


@pytest.mark.parametrize("learner_failure", [False, True])
def test_owned_children_exit_and_socket_cleanup_are_bound(
    inputs, tmp_path, monkeypatch, learner_failure
):
    processes, shutdowns = [], []
    monkeypatch.setattr(stage, "ROOT", tmp_path)
    (tmp_path / "builds").mkdir()
    monkeypatch.setattr(stage, "require_subset_lease", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-explicit-test-double")
    monkeypatch.setattr(stage.os, "getpgid", lambda _pid: stage.os.getpgrp())

    def preflight(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        write(Path(command[5]), {"explicit_cpu_double": True})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(stage.subprocess, "run", preflight)

    class ChildDouble:
        def __init__(self, command, **kwargs):
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-explicit-test-double"
            assert "start_new_session" not in kwargs
            self.pid, self.returncode = 2000000000 + len(processes), None
            self.role = (
                "host" if command[5] == "abc_bench.fasttd3_captured_host" else "learner"
            )
            if self.role == "learner":
                assert (
                    command[command.index("--checkpoint") + 1]
                    == inputs[1]["checkpoint"]["path"]
                )
                result_doubles(
                    tmp_path,
                    Path(command[command.index("--output") + 1]).parent,
                    inputs[1],
                )
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 1 if self.role == "learner" and learner_failure else 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(stage.subprocess, "Popen", ChildDouble)
    _proof, description = proof_and_description(tmp_path)
    monkeypatch.setattr(stage, "wait_ready", lambda *_args: description)

    def shutdown(_path, operation, *, timeout):
        assert operation == "shutdown" and timeout == 30
        shutdowns.append(operation)
        processes[0].returncode = 0
        return {"status": "closed"}

    monkeypatch.setattr(stage, "rpc", shutdown)
    output = tmp_path / "run"
    if learner_failure:
        with pytest.raises(RuntimeError, match="consumer failed"):
            stage.run(inputs[0], output)
    else:
        assert stage.run(inputs[0], output)["status"] == "completed"
    result = json.loads((output / "receipt.json").read_text())
    assert result["status"] == ("failed" if learner_failure else "completed")
    assert all(child.poll() is not None for child in processes)
    assert shutdowns == ([] if learner_failure else ["shutdown"])
    assert list((tmp_path / "builds").iterdir()) == []
