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
        learner["episodes"][0]["controls"] = host["episodes"][-1]["controls"] = 1000
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


def tree_pin(root: Path, target: Path, kind: str, producer: Path) -> dict:
    records = [
        {**pin(path), "path": str(path.relative_to(root))}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    producer_pin = pin(producer)
    manifest = write(
        target,
        {
            "schema": "expo-abc-checkpoint-tree/v1",
            "kind": kind,
            "root": str(root),
            "files": records,
            "total_bytes": sum(row["bytes"] for row in records),
            "tree_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "producer_receipt": {key: producer_pin[key] for key in ("path", "sha256")},
        },
    )
    return {key: manifest[key] for key in ("path", "sha256")}


def bind_native(data: dict, name: str) -> None:
    run, worker, host = data[name], data[name + "_worker"], data[name + "_host"]
    episode = worker["episodes"][0]
    controls, completed = episode["controls"], episode["completed_native_episode"]
    worker["actual_controls"] = controls
    worker["metrics"] = {
        "full_task_success_count": int(episode["full_task_success"]),
        "completed_native_episode_count": int(completed),
        "max_native_reward": episode["max_native_reward"],
        "command_clip_count": episode["command_clip_count"],
        "max_ever_placed_bottles": episode["ever_placed_count"],
        "sampler_wall_latency": {
            "count": (controls + 7) // 8,
            "warm_count": (controls + 7) // 8 - 1,
            "first_cold_s": 3.0,
            "all_mean_s": 1.6,
            "warm_median_s": 0.2,
            "warm_p95_s": 0.2,
        },
    }
    host["metrics"] = {
        "controls": controls,
        "completed_episodes": int(completed),
        "action_clipping_count": episode["command_clip_count"],
    }
    worker["host_lifecycle"]["host_receipt"] = copy.deepcopy(host)
    host_pin = write(run / "expoft/host/receipt.json", host)
    worker["host_lifecycle"]["receipt"] = {
        key: host_pin[key] for key in ("path", "sha256")
    }
    seal(run, worker)
    (run / "expoft").mkdir(exist_ok=True)
    (run / "receipt.json").replace(run / "expoft/receipt.json")


@pytest.fixture
def native_cohort(cohort):
    """Explicit native producer-schema doubles; these do not run a model or simulator."""
    data = cohort
    bench = data["baseline"].parent
    source = write(
        bench / "source.json",
        {
            "schema": "abc-author-success-prefix/v1",
            "split": "train",
            "task": "put_plastic_bottles_in_bin",
            "source_judge": write(bench / "source-judge.json", {"fixture": True}),
            "source_labels": write(bench / "source-labels.json", {"fixture": True}),
            "raw_inputs": {"states": video(bench / "source-raw.bin")},
        },
    )
    norm = write(bench / "stats/norm_stats.json", {"fixture": "TRAIN statistics"})
    stats = write(
        bench / "stats/diagnostic_manifest.json",
        {
            "schema": "expo-abc-yam14-diagnostic/v1",
            "split": "train",
            "source_manifest": {key: source[key] for key in ("path", "sha256")},
            "stats": {key: norm[key] for key in ("path", "sha256")},
        },
    )
    sft_root = bench / "sft-checkpoint"
    saved_norm = sft_root / "assets/abc_yam14_absolute_one_demo_v1/norm_stats.json"
    saved_norm.parent.mkdir(parents=True)
    saved_norm.write_bytes(Path(norm["path"]).read_bytes())
    video(sft_root / "params/weights.bin")
    sft_worker = json.loads((data["sft"] / "expoft/receipt.json").read_text())
    sft_worker["checkpoint_artifact"] = str(sft_root)
    sft_worker["input_pins"] = {
        "source_manifest": {key: source[key] for key in ("path", "sha256")},
        "stats_manifest": {key: stats[key] for key in ("path", "sha256")},
    }
    seal(data["sft"], sft_worker)
    (data["sft"] / "receipt.json").replace(data["sft"] / "expoft/receipt.json")
    sft_pin = tree_pin(
        sft_root,
        bench / "sft-tree.json",
        "original_pi05_sft",
        data["sft"] / "expoft/receipt.json",
    )
    inputs = {
        "checkpoint_pin": sft_pin,
        "source_manifest": {key: source[key] for key in ("path", "sha256")},
        "stats_manifest": {key: stats[key] for key in ("path", "sha256")},
    }
    data["native_source"], data["native_stats"], data["sft_tree_root"] = (
        source,
        stats,
        sft_root,
    )
    baseline_world = json.loads(Path(data["summary"]["path"]).read_text())["worlds"][0]
    counters = {
        key: 0 for key in ("critic", "batch_encoder", "edit_actor", "temp", "actor")
    }
    groups = (
        "critic",
        "batch_encoder",
        "edit_actor",
        "temp",
        "target_critic",
        "actor",
        "target_actor",
    )
    hashes = {
        key: {
            "sha256": "a" * 64,
            "bytes": 4,
            "leaves": 1,
            "algorithm": "SHA256 ordered leaf path/shape/dtype and exact C-order parameter bytes",
        }
        for key in groups
    }
    for name, phase in (
        ("pi05_eval", "original_pi05_frozen_eval"),
        ("expo_train", "original_expo_update_diagnostic"),
        ("expo_eval", "original_expo_frozen_eval"),
    ):
        train = name == "expo_train"
        run = data[name] = bench / name
        host_dir = run / "expoft/host"
        seed, controls = (9030, 32) if train else (20260511, 3540)
        host_config = write(
            run / "host-config.json",
            {
                "stage": "train" if train else "evaluation",
                "host": {
                    "max_episode_steps": 3540,
                    "reference": None
                    if train
                    else {"sha256": data["summary"]["sha256"], "world_index": 0},
                },
            },
        )
        clip = video(host_dir / "episode.mp4")
        judge = {
            "success": False,
            "num_active_bottles": 5,
            "num_bottles_in_bin": 0,
            "bottle_in_bin_mask": [False] * 5,
        }
        host_episode = {
            "seed": seed,
            "controls": controls,
            "completed": not train,
            "success": False,
            "action_clipping_count": 0,
            "max_native_reward": 0.0,
            "scene_match": "fixed_five_train" if train else "matched_baseline",
            "randomization": copy.deepcopy(baseline_world["randomization"]),
            "final_native_task_eval": judge,
            "end_reason": "host_closed" if train else "timeout",
            "video": clip,
        }
        worker_episode = {
            "seed": seed,
            "controls": controls,
            "completed_native_episode": not train,
            "terminated": False,
            "truncated": not train,
            "full_task_success": False,
            "max_native_reward": 0.0,
            "command_clip_count": 0,
            "ever_placed_mask": [False] * 5,
            "ever_placed_count": 0,
            "stop_reason": "diagnostic_update_limit" if train else "native_time_limit",
            "final_native_task_eval": copy.deepcopy(judge),
        }
        reset_only = {
            **copy.deepcopy(host_episode),
            "controls": 0,
            "completed": False,
            "end_reason": "explicit_reset",
            "video": video(host_dir / "reset-only.mp4"),
        }
        data[name + "_host"] = {
            "schema_version": 1,
            "status": "completed",
            "stage": "train" if train else "evaluation",
            "blockers": [],
            "config": host_config,
            "episodes": [reset_only, host_episode],
            "original_sources_assets": write(
                run / "assets.json", {"fixture": "sealed historical native assets"}
            ),
        }
        parameters = {
            "source_manifest": source["path"],
            "assets_dir": str(bench / "stats"),
            "checkpoint_pin": sft_pin,
            "stats_manifest_sha256": stats["sha256"],
            "host_config": host_config["path"],
            "seeds": [seed],
            "max_controls": controls,
        }
        after = (
            {key: 1 for key in counters}
            if train or name == "expo_eval"
            else dict(counters)
        )
        before = dict(after) if name == "expo_eval" else dict(counters)
        worker = data[name + "_worker"] = {
            "schema_version": 1,
            "status": "completed",
            "phase": phase,
            "resolved_parameters": parameters,
            "input_pins": copy.deepcopy(inputs),
            "producer_sft_steps": 1,
            "source_pins": {
                "expo_ft": "21fe3d3b7d913c80836496817965932f49e0aedf",
                "openpi": "46407a41183b037313a383ff679683f2773b766d",
            },
            "implementation_sha256": {
                file: "b" * 64
                for file in (
                    "native.py",
                    "original_runtime.py",
                    "artifacts.py",
                    "host.py",
                    "data.py",
                    "config.py",
                    "replay.py",
                    "stage.py",
                )
            },
            "actual_steps": int(train),
            "elapsed_s": 20.0,
            "early_stopping_reason": worker_episode["stop_reason"],
            "optimizer_counters_before": before,
            "optimizer_counters_after": after,
            "sampler_rng_before": [31, 32] if name == "expo_eval" else [11, 12],
            "sampler_rng_after": [91, 92] if name == "expo_eval" else [31, 32],
            "actual_parameter_hashes_before": copy.deepcopy(hashes),
            "actual_parameter_hashes_after": copy.deepcopy(hashes),
            "episodes": [worker_episode],
            "updates": [],
            "diagnostic_changes": {},
            "sampling_recipe": {
                "base_candidates": 8,
                "edit_candidates": 0 if name == "pi05_eval" else 8,
                "action_horizon": 16,
                "executed_prefix": 8,
            },
            "host_lifecycle": {
                "exit_code": 0,
                "videos": [reset_only["video"], clip],
                "host_config": {key: host_config[key] for key in ("path", "sha256")},
            },
        }
        if train:
            parameters.update(batch_size=2, utd_ratio=1, max_updates=1)
            worker["diagnostic_changes"] = {
                "batch_size": 2,
                "utd_ratio": 1,
                "author_release_batch_size": 64,
                "author_release_utd_ratio": 20,
            }
            worker["updates"] = [
                {
                    "actual_update": 1,
                    "optimizer_counters_before": before,
                    "optimizer_counters_after": after,
                    "original_metrics": {"critic_loss": 0.5},
                    "parameter_probe_after": {key: [-1.0, 2.0, 3.0] for key in groups},
                }
            ]
            checkpoint_root = run / "expoft/checkpoints/1"
            video(checkpoint_root / "weights.bin")
            worker["checkpoint_artifact"] = str(checkpoint_root)
        elif name == "expo_eval":
            parameters["expo_checkpoint_pin"] = data["expo_checkpoint_pin"]
            worker["expo_checkpoint_pin"] = data["expo_checkpoint_pin"]
            worker["restored_parameter_probe"] = {
                key: [-1.0, 2.0, 3.0] for key in groups
            }
        bind_native(data, name)
        if train:
            data["expo_checkpoint_pin"] = tree_pin(
                checkpoint_root,
                run / "expoft/checkpoint-pin.json",
                "original_expo_learner",
                run / "expoft/receipt.json",
            )
            producer = pin(run / "expoft/receipt.json")
            write(
                run / "expoft/checkpoint-export.json",
                {
                    "schema_version": 1,
                    "checkpoint_pin": data["expo_checkpoint_pin"],
                    "producer_receipt": {
                        key: producer[key] for key in ("path", "sha256")
                    },
                },
            )
    return data


def native_report(data: dict, **overrides):
    return report(
        data,
        **{
            **{key: data[key] for key in ("pi05_eval", "expo_train", "expo_eval")},
            **overrides,
        },
    )


def test_native_pi05_and_expo_scores_require_complete_matched_frozen_evidence(
    native_cohort,
):
    result = native_report(native_cohort)
    pi = result["methods"]["pi05"]
    assert (pi["evaluation"]["successes"], pi["evaluation"]["episodes"]) == (0, 1)
    assert (
        pi["expoft_evaluation"]["successes"],
        pi["expoft_evaluation"]["controls"],
    ) == (0, 3540)
    train = pi["expoft_training"]
    assert train["successes"] is None
    assert (
        train["batch_size"],
        train["utd_ratio"],
        train["controls"],
        train["actual_steps"],
    ) == (2, 1, 32, 1)
    assert (native_cohort["output"] / pi["evaluation"]["video"]).is_file()


@pytest.mark.parametrize("status", ["running", "failed"])
def test_native_incomplete_job_does_not_consume_partial_worker_or_video(
    native_cohort, status
):
    run = native_cohort["pi05_eval"]
    coordinator_path = run.parent / "runs" / (run.name + ".json")
    coordinator = json.loads(coordinator_path.read_text())
    coordinator["status"] = status
    write(coordinator_path, coordinator)
    (run / "expoft/receipt.json").write_bytes(b"incomplete JSON")
    Path(native_cohort["pi05_eval_host"]["episodes"][-1]["video"]["path"]).write_bytes(
        b"incomplete video"
    )
    result = native_report(native_cohort)
    assert result["methods"]["pi05"]["evaluation"]["successes"] is None


def test_native_partial_rollout_has_verified_clip_and_no_task_score(native_cohort):
    worker, host = native_cohort["pi05_eval_worker"], native_cohort["pi05_eval_host"]
    episode, actual = worker["episodes"][0], host["episodes"][-1]
    episode.update(
        controls=123,
        completed_native_episode=False,
        truncated=False,
        stop_reason="control_limit",
    )
    actual.update(controls=123, completed=False, end_reason="host_closed")
    bind_native(native_cohort, "pi05_eval")
    result = native_report(native_cohort)["methods"]["pi05"]["evaluation"]
    assert result["status"] == "partial"
    assert result["successes"] is None and result["episodes"] is None
    assert result["controls"] == 123
    assert (native_cohort["output"] / result["episode_rollouts"][0]["video"]).is_file()


@pytest.mark.parametrize(
    "change",
    [
        "scene",
        "parameters",
        "missing_parameter_group",
        "optimizer",
        "missing_counter_group",
        "updates",
        "short_horizon",
        "phase",
    ],
)
def test_invalid_native_frozen_evidence_leaves_live_report_intact(
    native_cohort, change
):
    native_report(native_cohort)
    live = {
        name: (native_cohort["output"] / name).read_bytes()
        for name in ("results.json", "index.html")
    }
    worker, host = native_cohort["pi05_eval_worker"], native_cohort["pi05_eval_host"]
    if change == "scene":
        host["episodes"][-1]["randomization"]["scale_states"]["bin_joint"] = 2.0
    elif change == "parameters":
        worker["actual_parameter_hashes_after"]["actor"]["sha256"] = "c" * 64
    elif change == "missing_parameter_group":
        del worker["actual_parameter_hashes_before"]["target_actor"]
        del worker["actual_parameter_hashes_after"]["target_actor"]
    elif change == "optimizer":
        worker["optimizer_counters_after"]["actor"] = 1
    elif change == "missing_counter_group":
        del worker["optimizer_counters_before"]["temp"]
        del worker["optimizer_counters_after"]["temp"]
    elif change == "updates":
        worker["actual_steps"] = 1
    elif change == "short_horizon":
        worker["episodes"][0]["controls"] = host["episodes"][-1]["controls"] = 1000
    else:
        worker["phase"] = "invented_frozen_policy"
    bind_native(native_cohort, "pi05_eval")
    with pytest.raises(ValueError):
        native_report(native_cohort)
    for name, before in live.items():
        assert (native_cohort["output"] / name).read_bytes() == before


@pytest.mark.parametrize(
    "change",
    ["video", "host_receipt", "raw_source", "stats", "sft_file", "sft_extra_file"],
)
def test_native_artifact_mutation_is_rejected(native_cohort, change):
    if change == "sft_extra_file":
        (native_cohort["sft_tree_root"] / "extra.bin").write_bytes(b"unrecorded")
    else:
        path = {
            "video": Path(
                native_cohort["pi05_eval_host"]["episodes"][-1]["video"]["path"]
            ),
            "host_receipt": native_cohort["pi05_eval"] / "expoft/host/receipt.json",
            "raw_source": native_cohort["baseline"].parent / "source-raw.bin",
            "stats": native_cohort["baseline"].parent / "stats/norm_stats.json",
            "sft_file": native_cohort["sft_tree_root"] / "params/weights.bin",
        }[change]
        path.write_bytes(path.read_bytes() + b"mutated")
    with pytest.raises(ValueError):
        native_report(native_cohort)
    assert not (native_cohort["output"] / "results.json").exists()


def test_frozen_expo_requires_the_selected_training_export(native_cohort):
    with pytest.raises(ValueError, match="TRAIN export"):
        native_report(native_cohort, expo_train=None)


@pytest.mark.parametrize("change", ["counters", "rng", "probe"])
def test_frozen_expo_rejects_a_different_restored_training_state(native_cohort, change):
    worker = native_cohort["expo_eval_worker"]
    if change == "counters":
        worker["optimizer_counters_before"] = {
            key: 0 for key in worker["optimizer_counters_before"]
        }
        worker["optimizer_counters_after"] = dict(worker["optimizer_counters_before"])
    elif change == "rng":
        worker["sampler_rng_before"] = [99, 100]
    else:
        worker["restored_parameter_probe"]["actor"] = [9.0, 9.0, 9.0]
    bind_native(native_cohort, "expo_eval")
    with pytest.raises(ValueError, match="restored TRAIN state"):
        native_report(native_cohort)


def test_sft_tree_cannot_substitute_another_checkpoint_root(native_cohort):
    tree_path = Path(
        native_cohort["pi05_eval_worker"]["input_pins"]["checkpoint_pin"]["path"]
    )
    tree = json.loads(tree_path.read_text())
    alternate = native_cohort["sft_tree_root"].with_name("substituted-checkpoint")
    for record in tree["files"]:
        destination = alternate / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            (native_cohort["sft_tree_root"] / record["path"]).read_bytes()
        )
    tree["root"] = str(alternate)
    substituted = write(tree_path, tree)
    worker = native_cohort["pi05_eval_worker"]
    worker["input_pins"]["checkpoint_pin"] = worker["resolved_parameters"][
        "checkpoint_pin"
    ] = {key: substituted[key] for key in ("path", "sha256")}
    bind_native(native_cohort, "pi05_eval")
    with pytest.raises(ValueError, match="checkpoint root"):
        native_report(native_cohort)


@pytest.mark.parametrize("change", ["source", "stats"])
def test_native_stage_cannot_substitute_sft_lineage(native_cohort, change):
    worker = native_cohort["pi05_eval_worker"]
    inputs = worker["input_pins"]
    stats = json.loads(Path(inputs["stats_manifest"]["path"]).read_text())
    if change == "source":
        source = json.loads(Path(inputs["source_manifest"]["path"]).read_text())
        source["fixture_source_identifier"] = "another TRAIN lineage"
        alternate = write(
            native_cohort["baseline"].parent / "alternate-source.json", source
        )
        inputs["source_manifest"] = {key: alternate[key] for key in ("path", "sha256")}
        stats["source_manifest"] = inputs["source_manifest"]
        worker["resolved_parameters"]["source_manifest"] = alternate["path"]
    alternate_stats = write(
        native_cohort["baseline"].parent / "alternate-stats.json", stats
    )
    inputs["stats_manifest"] = {key: alternate_stats[key] for key in ("path", "sha256")}
    worker["resolved_parameters"]["stats_manifest_sha256"] = alternate_stats["sha256"]
    bind_native(native_cohort, "pi05_eval")
    with pytest.raises(ValueError, match="SFT/source/stats"):
        native_report(native_cohort)


@pytest.mark.parametrize("change", ["utd_bool", "update_counter_bool", "export_root"])
def test_invalid_expo_training_recipe_or_export_is_rejected(native_cohort, change):
    worker = native_cohort["expo_train_worker"]
    if change == "utd_bool":
        worker["resolved_parameters"]["utd_ratio"] = True
    elif change == "update_counter_bool":
        worker["updates"][0]["optimizer_counters_after"] = {
            **worker["updates"][0]["optimizer_counters_after"],
            "actor": True,
        }
    else:
        worker["checkpoint_artifact"] = str(
            native_cohort["expo_train"] / "other-export"
        )
    bind_native(native_cohort, "expo_train")
    with pytest.raises(ValueError):
        native_report(native_cohort)


def test_native_original_success_can_finish_before_the_time_limit(native_cohort):
    worker, host = native_cohort["pi05_eval_worker"], native_cohort["pi05_eval_host"]
    episode, actual = worker["episodes"][0], host["episodes"][-1]
    judge = {
        "success": True,
        "num_active_bottles": 5,
        "num_bottles_in_bin": 5,
        "bottle_in_bin_mask": [True] * 5,
    }
    episode.update(
        controls=819,
        terminated=True,
        truncated=False,
        full_task_success=True,
        stop_reason="task_success",
        ever_placed_mask=[True] * 5,
        ever_placed_count=5,
        max_native_reward=1.0,
        final_native_task_eval=judge,
    )
    actual.update(
        controls=819,
        success=True,
        end_reason="success",
        max_native_reward=1.0,
        final_native_task_eval=copy.deepcopy(judge),
    )
    bind_native(native_cohort, "pi05_eval")
    result = native_report(native_cohort)["methods"]["pi05"]["evaluation"]
    assert (result["successes"], result["controls"]) == (1, 819)
