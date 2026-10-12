"""Explicit file/receipt doubles; no simulator, model, or real task score."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from abc_bench.upstream_report import publish


def pin(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return pin(path)


def seal(run: Path, worker: dict) -> None:
    """A test coordinator authenticates the exact fixture worker and job bytes."""
    job = write(run / "job.json", {"fixture": "explicit report CPU double"})
    receipt = write(run / "receipt.json", worker)
    write(
        run.parent / "runs" / f"{run.name}.json",
        {
            "schema_version": 1,
            "run_id": run.name,
            "status": "completed",
            "task": "put_plastic_bottles_in_bin",
            "embodiment": "native_yam",
            "worker_exit_code": 0,
            "resolved_job_sha256": job["sha256"],
            "worker_receipt_sha256": receipt["sha256"],
            "metrics": worker.get("metrics", {}),
        },
    )


def video(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"explicit video identity fixture: " + path.name.encode())
    return pin(path)


@pytest.fixture
def cohort(tmp_path):
    bench = tmp_path / "bench"
    baseline, train, evaluation, sft = (
        bench / name for name in ("baseline", "train", "evaluation", "sft")
    )
    randomization = {
        "seed": 20260511,
        "metadata": {"bottle_count": 5, "bottle_variants": ["bottle_18"] * 5},
        "object_states": {"bin_joint": {"pos": [0.1, 0.2, 0.3], "quat": [1, 0, 0, 0]}},
        "scale_states": {"bin_joint": 1.1},
    }
    base_video = video(baseline / "baseline.mp4")
    world = {
        "world_seed": 20260511,
        "success": True,
        "final_success": True,
        "steps": 819,
        "video_path": base_video["path"],
        "final_task_eval": {"num_bottles_in_bin": 5},
        "randomization": randomization,
    }
    summary = write(
        baseline / "summary.json",
        {
            "task": "put_plastic_bottles_in_bin",
            "num_worlds": 1,
            "num_success": 1,
            "worlds": [world],
            "config": {"rtc": True},
        },
    )
    seal(
        baseline,
        {
            "schema_version": 1,
            "status": "completed",
            "algorithm": "original ABC-VLA",
            "phase": "original_recipe_one_world_qualification",
            "original_summary": summary,
            "original_videos": [base_video],
        },
    )
    checkpoint = video(train / "export.pt")
    train_episodes = [
        {
            "seed": seed,
            "controls": controls,
            "completed": completed,
            "max_native_reward": 0.0,
            "end_reason": reason,
            "video": video(train / f"train-{seed}.mp4"),
            "randomization": {
                "metadata": {"bottle_count": 5, "bottle_variants": ["bottle_0"] * 5},
                "scale_states": {},
            },
        }
        for seed, controls, completed, reason in [
            (9030, 3540, True, "timeout"),
            (9031, 460, False, "host_closed"),
        ]
    ]
    train_learner = {
        "schema_version": 1,
        "condition": "original_fasttd3_scratch_privileged_state_abc",
        "status": "completed",
        "environment": {
            "privileged_state": True,
            "reset_options": {
                "randomization": {
                    "randomize_variants": False,
                    "randomize_scales": False,
                }
            },
        },
        "checkpoint": checkpoint,
        "controls": 4000,
        "actual_updates": {"critic": 7978, "actor": 3989},
    }
    train_host = {
        "schema_version": 1,
        "status": "completed",
        "metrics": {
            "controls": 4000,
            "completed_episodes": 1,
            "normalized_action_clipping_count": 18,
        },
        "episodes": train_episodes,
    }
    frozen_episode = {
        "seed": 20260511,
        "controls": 3540,
        "completed": True,
        "success": False,
        "scene_match": "matched_baseline",
        "randomization": copy.deepcopy(randomization),
        "end_reason": "timeout",
        "max_native_reward": 0.0,
        "final_native_task_eval": {
            "success": False,
            "num_active_bottles": 5,
            "num_bottles_in_bin": 0,
        },
        "video": video(evaluation / "evaluation.mp4"),
    }
    eval_learner = {
        "schema_version": 1,
        "condition": "original_fasttd3_scratch_privileged_state_abc",
        "status": "completed",
        "environment": {"privileged_state": True},
        "checkpoint": checkpoint,
        "evaluation_seeds": [20260511],
        "episodes": [{**copy.deepcopy(frozen_episode), "return": 0.0}],
        "successes": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "policy_state_before": "a" * 64,
        "policy_state_after": "a" * 64,
    }
    eval_host = {
        "schema_version": 1,
        "status": "completed",
        "metrics": {
            "controls": 3540,
            "completed_episodes": 1,
            "normalized_action_clipping_count": 0,
        },
        "episodes": [copy.deepcopy(frozen_episode)],
    }
    data = {
        "baseline": baseline,
        "train": train,
        "evaluation": evaluation,
        "sft": sft,
        "summary": summary,
        "train_learner": train_learner,
        "train_host": train_host,
        "eval_learner": eval_learner,
        "eval_host": eval_host,
        "output": tmp_path / "public",
    }
    bind_fasttd3(data, "train")
    bind_fasttd3(data, "evaluate")
    sft_worker = {
        "schema_version": 1,
        "status": "completed",
        "phase": "original_pi05_sft_qualification",
        "actual_steps": 1,
        "early_stopping_reason": "tiny_batch_loss_target",
        "elapsed_s": 224.0,
        "metrics": {
            "initial_fixed_loss": 0.111,
            "final_fixed_loss": 0.060,
            "final_gradient_norm": 12.59,
            "optimization_passed": True,
            "tiny_batch_loss_target_met": True,
        },
    }
    # SFT has a nested worker directory but the canonical coordinator owns its parent.
    seal(sft, sft_worker)
    (sft / "expoft").mkdir()
    (sft / "receipt.json").rename(sft / "expoft" / "receipt.json")
    return data


def bind_fasttd3(data: dict, stage: str) -> None:
    prefix = "train" if stage == "train" else "eval"
    run = data["train"] if stage == "train" else data["evaluation"]
    learner, host = data[f"{prefix}_learner"], data[f"{prefix}_host"]
    host["original_sources_assets"] = write(
        run / "assets.json", {"fixture": "sealed historical inventory"}
    )
    metrics = (
        {"controls": 4000, "actor_updates": 3989, "critic_updates": 7978}
        if stage == "train"
        else {
            "num_worlds": 1,
            "successes": 0,
            "matched_worlds": 1,
            "actor_updates": 0,
            "critic_updates": 0,
        }
    )
    worker = {
        "schema_version": 1,
        "status": "completed",
        "algorithm": "original FastTD3 from scratch",
        "phase": stage,
        "metrics": metrics,
        "elapsed_seconds": 435.0,
        "learner_receipt": write(run / "learner.json", learner),
        "host_receipt": write(run / "host.json", host),
        "resolved_config": write(
            run / "config.json",
            {
                "stage": stage,
                "host": {
                    "max_episode_steps": 3540,
                    "reference": {
                        "summary": data["summary"]["path"],
                        "sha256": data["summary"]["sha256"],
                        "world_index": 0,
                    },
                },
            },
        ),
        "host_description": write(run / "description.json", {"privileged_state": True}),
        "learner_module_snapshot": write(
            run / "modules.json",
            {
                "fixture": "sealed historical inventory",
                "modules": [
                    {
                        "path": "/historical/sadhana_fasttd3_abc/learner.py",
                        "sha256": data.get(
                            "eval_module_sha",
                            "fe3881d2ac5e3371b83bed7da26840dd281222e11696ac262363d1973478ff41",
                        ),
                    }
                ],
            },
        ),
    }
    if stage == "evaluate":
        worker["evaluation_checkpoint"] = learner["checkpoint"]
    seal(run, worker)


def report(cohort, **overrides):
    options = {
        "fasttd3_train": cohort["train"],
        "fasttd3_eval": cohort["evaluation"],
        "pi05_sft": cohort["sft"],
    }
    options.update(overrides)
    return publish(cohort["baseline"], cohort["output"], **options)


def test_existing_baseline_api_keeps_missing_results_pending(cohort):
    result = publish(cohort["baseline"], cohort["output"])
    assert (result["successes"], result["episodes"], result["controls"]) == (1, 1, 819)
    assert result["methods"]["fasttd3"]["evaluation"]["successes"] is None
    assert result["methods"]["pi05"]["sft"]["status"] == "pending"


def test_unicode_html_is_utf8_under_a_real_ascii_locale(cohort):
    script = """
import codecs
import locale
import sys
from pathlib import Path
from abc_bench.upstream_report import HTML, publish

assert sys.flags.utf8_mode == 0
assert codecs.lookup(locale.getpreferredencoding(False)).name == "ascii"
output = Path(sys.argv[2])
publish(Path(sys.argv[1]), output)
payload = (output / "index.html").read_bytes()
assert payload == HTML.encode("utf-8")
assert any(value > 127 for value in payload)
print("ASCII locale confirmed; actual Unicode HTML matches UTF-8 bytes")
"""
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            "utf8=0",
            "-c",
            script,
            str(cohort["baseline"]),
            str(cohort["output"]),
        ],
        env={**os.environ, "LC_ALL": "C", "LANG": "C", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "ASCII locale confirmed" in child.stdout


def test_complete_matched_frozen_eval_is_the_only_fasttd3_score(cohort):
    result = report(cohort)
    fast = result["methods"]["fasttd3"]
    assert fast["training"]["successes"] is None
    assert fast["training"]["controls"] == 4000
    assert fast["training"]["critic_updates"] == 7978
    assert len(fast["training"]["episodes"]) == 2
    assert fast["evaluation"]["successes"] == 0
    assert fast["evaluation"]["episodes"] == 1
    assert fast["evaluation"]["evaluation_precision"]["autocast_enabled"] is False
    assert (
        "FP32 legacy diagnostic" in fast["evaluation"]["evaluation_precision"]["label"]
    )
    assert fast["evaluation"]["scene_distribution_changes"] == [
        "bottle shapes",
        "object scales",
    ]
    assert (cohort["output"] / fast["evaluation"]["video"]).is_file()
    assert result["methods"]["pi05"]["sft"]["actual_steps"] == 1
    assert result["methods"]["pi05"]["sft"]["successes"] is None


def recorded_precision() -> dict:
    """The new producer schema, with explicit observed dtypes and backend flags."""
    return {
        "autocast_enabled": True,
        "autocast_device_type": "cuda",
        "configured_amp": True,
        "configured_cuda": True,
        "amp_dtype": "bf16",
        "autocast_dtype": "torch.bfloat16",
        "policy_parameter_dtype": ["torch.float32"],
        "policy_action_dtype": ["torch.bfloat16"],
        "environment_action_dtype": "torch.float32",
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": True,
    }


def test_recorded_amp_precision_is_shown_without_changing_the_task_score(cohort):
    cohort["eval_learner"]["evaluation_precision"] = recorded_precision()
    bind_fasttd3(cohort, "evaluate")
    result = report(cohort)
    evaluation = result["methods"]["fasttd3"]["evaluation"]
    assert evaluation["evaluation_precision"]["label"] == "BF16 autocast"
    assert evaluation["evaluation_precision"]["policy_action_dtype"] == [
        "torch.bfloat16"
    ]
    assert evaluation["successes"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("autocast_enabled", "true"),
        ("amp_dtype", "fp32"),
        ("policy_action_dtype", []),
        ("configured_amp", False),
    ],
)
def test_invalid_precision_cannot_be_published_as_author_evaluation(
    cohort, field, value
):
    profile = recorded_precision()
    profile[field] = value
    cohort["eval_learner"]["evaluation_precision"] = profile
    bind_fasttd3(cohort, "evaluate")
    with pytest.raises(ValueError, match="precision|autocast"):
        report(cohort)
    assert not (cohort["output"] / "results.json").exists()


def test_missing_precision_for_unknown_legacy_source_is_not_assumed_fp32(cohort):
    cohort["eval_module_sha"] = "b" * 64
    bind_fasttd3(cohort, "evaluate")
    with pytest.raises(ValueError, match="precision is unrecorded"):
        report(cohort)


def test_running_eval_has_no_score(cohort):
    path = cohort["evaluation"].parent / "runs" / "evaluation.json"
    receipt = json.loads(path.read_text())
    receipt["status"] = "running"
    write(path, receipt)
    result = report(cohort)
    assert result["methods"]["fasttd3"]["evaluation"]["status"] == "pending"
    assert result["methods"]["fasttd3"]["evaluation"]["successes"] is None


@pytest.mark.parametrize(
    "change", ["partial", "scene", "frozen", "updates", "short_horizon", "checkpoint"]
)
def test_unqualified_eval_cannot_publish_a_score(cohort, change):
    learner, host = cohort["eval_learner"], cohort["eval_host"]
    if change == "partial":
        learner["episodes"][0]["completed"] = host["episodes"][0]["completed"] = False
    elif change == "scene":
        for value in (learner["episodes"][0], host["episodes"][0]):
            value["randomization"]["scale_states"]["bin_joint"] = 1.2
    elif change == "frozen":
        learner["policy_state_after"] = "b" * 64
    elif change == "updates":
        learner["actor_updates"] = 1
    elif change == "short_horizon":
        learner["episodes"][0]["controls"] = host["episodes"][0]["controls"] = 1000
    else:
        learner["checkpoint"] = video(cohort["evaluation"] / "different-export.pt")
    bind_fasttd3(cohort, "evaluate")
    with pytest.raises(ValueError):
        report(cohort)
    assert not (cohort["output"] / "results.json").exists()


def test_complete_eval_requires_the_bound_train_export(cohort):
    with pytest.raises(ValueError, match="TRAIN export"):
        report(cohort, fasttd3_train=None)


@pytest.mark.parametrize(
    "target", ["worker", "learner", "video", "eval_video", "checkpoint", "job"]
)
def test_mutated_artifacts_leave_existing_live_report_intact(cohort, target):
    report(cohort)
    result_before = (cohort["output"] / "results.json").read_bytes()
    html_before = (cohort["output"] / "index.html").read_bytes()
    path = {
        "worker": cohort["train"] / "receipt.json",
        "learner": cohort["train"] / "learner.json",
        "video": Path(cohort["train_host"]["episodes"][0]["video"]["path"]),
        "eval_video": Path(cohort["eval_host"]["episodes"][0]["video"]["path"]),
        "checkpoint": Path(cohort["train_learner"]["checkpoint"]["path"]),
        "job": cohort["train"] / "job.json",
    }[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        report(cohort)
    assert (cohort["output"] / "results.json").read_bytes() == result_before
    assert (cohort["output"] / "index.html").read_bytes() == html_before


def test_mutated_sft_receipt_is_not_a_completed_diagnostic(cohort):
    path = cohort["sft"] / "expoft" / "receipt.json"
    value = json.loads(path.read_text())
    value["actual_steps"] = 99
    write(path, value)
    with pytest.raises(ValueError, match="coordinator pin"):
        report(cohort)
