"""Synthetic CPU publication boundaries. These fixtures are not robot or video results."""

import fcntl
import json
import tempfile
import unittest
from pathlib import Path

from abc_bench.report_r1lite_expo import pin, publish


class MatchedPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.marker = self.root / "explicit-cpu-fixture.txt"
        self.marker.write_text(
            "Synthetic CPU evidence; no native execution or accepted experiment."
        )
        self.marker_pin = pin(self.marker)
        self.viewer = self.root / "viewer"
        self.viewer.mkdir()
        self.history = {}
        for i in range(40):
            path = self.viewer / (f"{i:016x}.mp4")
            path.write_bytes(f"Historical synthetic fixture {i}".encode())
            self.history[path.name] = path.read_bytes()
        self.before = {
            "schema": "sadhana.original-methods.progress/v2",
            "r1lite": {
                "successes": 8,
                "legacy": "preserve RTC field and failed shutdown",
            },
            "r1lite_initialized_full_expo": {
                "successes": 0,
                "legacy": "unrecorded PRNG",
            },
            "unrelated_method": {"score": 9},
            "packaged_viewer_release": {"fixture": True},
        }
        self.write(self.viewer / "results.json", self.before)
        (self.viewer / "index.html").write_text("<html>Prior CPU fixture</html>")

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return pin(path)

    def cohort(self, kind, damage=None):
        folder = self.root / kind
        folder.mkdir(exist_ok=True)
        trained = kind == "trained"
        step, successes = (9909, 1) if trained else (0, 0)
        run_id = "explicit-cpu-eval-" + kind
        checkpoint = str(self.root / "original-checkpoint/9909")
        scope = {
            "acceptance_scope": "frozen_evaluation_only",
            "frozen_evaluation_only": True,
            "raw_history_complete": False,
            "training_metrics_available": False,
        }
        counters = {
            key: {"step": n}
            for key, n in {
                "actor_train_state": 30,
                "edit_actor": 30,
                "temp": 30,
                "critic": 600,
                "filter_critic": 600,
                "batch_encoder": 600,
            }.items()
        }
        if damage == "counter":
            counters["critic"]["step"] = 599
        host = {
            **scope,
            "status": "host_checkpoint_replay_verified_history_incomplete",
            "all_inspected_files_unchanged": True,
            "checkpoint": checkpoint,
            "expected_update_calls": 30,
            "model": {"counters": counters},
        }
        if damage == "history_scope":
            host["training_metrics_available"] = True
        host_pin = self.write(folder / "HOST.json", host)
        producer = {
            **scope,
            "status": "root_accepted_original_online_checkpoint",
            "owner": "/root",
            "artifact_accepted": True,
            "run_id": "explicit-cpu-producer",
            "checkpoint": checkpoint,
            "checkpoint_step": 9909,
            "host_inspection": host_pin,
        }
        producer_pin = self.write(folder / "PRODUCER.json", producer)
        fixed = {
            "--seed": "42",
            "--num_episodes": "20",
            "--replan_steps": "8",
            "--delay": "5",
            "--only_base_actions": "False",
            "--save_video": "True",
            "--sim_latency": "0",
            "--config.N": "32",
            "--config.n_edit_samples": "32",
            "--config.filter_N": "32",
            "--config.filter_n_edit": "1",
            "--config.edit_scale": "0.1",
            "--checkpoint_step": str(step),
            "--checkpoint_dir": str(folder / "checkpoints"),
            "--client_port": "8198" if trained else "8197",
            "--config.pi05_weight_loader_path": "/fixture/RTC2000",
        }
        if damage == "recipe":
            fixed["--config.N"] = "1"
        config_pin = self.write(
            folder / "EVAL_CONFIG.json",
            {
                "program": "eval_droid_policy.py",
                "arguments": [f"{k}={v}" for k, v in fixed.items()],
            },
        )
        settings = {
            "episode_seeds": [9001, *range(9100, 9120)],
            "record_video": True,
            "event_log_path": str(folder / "native.jsonl"),
            "fixture_task": "same",
        }
        if damage == "comparison":
            settings["fixture_task"] = "different"
        settings_pin = self.write(folder / "EVAL_SETTINGS.json", settings)
        job_pin = self.write(folder / "EVAL_JOB.json", {"explicit_fixture": True})
        jax = {
            "JAX_DEFAULT_PRNG_IMPL": "threefry2x32",
            "JAX_THREEFRY_PARTITIONABLE": "true",
            "JAX_THREEFRY_GPU_KERNEL_LOWERING": "false",
            "JAX_ENABLE_X64": "false",
            "JAX_PLATFORMS": "cuda",
            "JAX_PLATFORM_NAME": "cuda",
        }
        intent = {
            "producer_run_id": "explicit-cpu-producer",
            "cohort": kind,
            "checkpoint_step": step,
            "accepted_producer_checkpoint": checkpoint,
            "config": config_pin,
            "job": job_pin,
            "settings": settings_pin,
            "root_artifact_acceptance": producer_pin,
            "host_inspection": host_pin,
            "declared_parent_jax": jax,
        }
        intent_pin = self.write(folder / "EVAL_EXEC_INTENT.json", intent)
        worker = {
            "status": "completed",
            "author_exit_code": 0,
            "client_exit_code": 0,
            "owned_children_stopped": True,
            "cleanup_errors": [],
            "author_shutdown": "already_stopped",
            "client_shutdown": "already_stopped",
        }
        if damage == "worker":
            worker["author_exit_code"] = 1
        worker_pin = self.write(folder / "worker.json", worker)
        canonical = {
            "run_id": run_id,
            "status": "completed",
            "worker_exit_code": 0,
            "worker_receipt_sha256": worker_pin["sha256"],
        }
        if damage == "canonical":
            canonical["status"] = "failed"
        canonical_pin = self.write(folder / "canonical.json", canonical)
        execution = {
            "canonical": canonical_pin,
            "worker": worker_pin,
            "natural_shutdown": True,
            "declared_and_actual_environment": {
                **jax,
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                "CUDA_VISIBLE_DEVICES": "GPU-CPU-FIXTURE",
            },
        }
        if damage == "prng":
            execution["declared_and_actual_environment"][
                "JAX_THREEFRY_PARTITIONABLE"
            ] = "false"
        episodes, media, scored = [], [], []
        for index, seed in enumerate([9001, *range(9100, 9120)]):
            controls = 900 if index else 0
            success = index > 0 and index <= successes
            video = folder / f"video-{seed}.mp4"
            video.write_bytes(
                f"Not playable: explicit CPU metadata fixture {kind} {seed}".encode()
            )
            sidecar_pin = self.write(
                folder / f"video-{seed}.json",
                {
                    "seed": seed,
                    "native_controls": controls,
                    "frames": controls + 1,
                    "fps": 15,
                    "complete_native_episode": index > 0,
                },
            )
            episodes.append(
                {
                    "seed": seed,
                    "scored": index > 0,
                    "errors": [],
                    "video_path": str(video),
                    "sidecar_path": sidecar_pin["path"],
                    "native_controls": controls,
                    "native_ending": "success" if success else "timeout",
                }
            )
            media.append(
                {
                    "seed": seed,
                    "scored": index > 0,
                    "video": pin(video),
                    "sidecar": sidecar_pin,
                    "full_decode_exit_code": 0,
                    "probe": {
                        "avg_frame_rate": "15/1",
                        "width": 640,
                        "height": 360,
                        "nb_read_frames": str(controls + 1),
                    },
                }
            )
            if index:
                scored.append(
                    {
                        "episode": index,
                        "seed": seed,
                        "success": success,
                        "return": float(success),
                        "native_return": float(success),
                        "len": controls + 1,
                        "native_controls": controls,
                    }
                )
        if damage == "media_pin":
            media[-1]["video"]["sha256"] = "0" * 64
        if damage == "media_frames":
            media[-1]["probe"]["nb_read_frames"] = "1"
        if damage == "media_count":
            media.pop()
        if damage == "startup":
            episodes[0]["seed"] = 9002
        observations = {
            "episodes": scored,
            "successes": successes,
            "failures": 20 - successes,
            "incomplete": 0,
            "full_agent_restore_marker": [str(step)] if trained else [],
            "actor_target_weight_loader_reads": []
            if trained
            else ["/fixture/RTC2000"] * 2,
            "async_full_expo_marker": "RealTimeEXPOFTLearner eval: using async main-actor inference delay=5 env-steps",
        }
        if damage == "restore":
            observations["full_agent_restore_marker"] = [] if trained else ["9909"]
        evidence = {
            "schema": "sadhana.r1lite_frozen_full_expo_dev20_acceptance/1",
            "status": "read_only_evidence_accepted",
            "run_id": run_id,
            "cohort": kind,
            "checkpoint_step": step,
            "producer_run_id": "explicit-cpu-producer",
            "accepted_producer_checkpoint": checkpoint,
            "evidence_errors": [],
            "checkpoint_files_unchanged": True,
            "processes_after_checks": {"live_owned": []},
            "author_native_comparison": observations,
            "accepted_scalar_observations": observations,
            "execution": execution,
            "frozen_scope": {
                "source_calls_update_or_save": [],
                "evaluation_checkpoint_step": step,
                "full_agent_restored": trained,
                "evaluator": self.marker_pin,
            },
            "native": {
                "status": "complete_native_evidence",
                "reset_order": [
                    [i + 1, s] for i, s in enumerate([9001, *range(9100, 9120)])
                ],
                "schedule_valid": True,
                "errors": [],
                "episodes": episodes,
            },
            "media": media,
            "telemetry": {"explicit_fixture": True},
            "launch_pins": [config_pin, job_pin, settings_pin],
            "producer_artifact_pins": [producer_pin, host_pin],
            "artifact_pins": [self.marker_pin, intent_pin],
        }
        for name in (
            "source_pins",
            "operations_pins",
            "installed_supervisor_pins",
            "immutable_input_pins",
            "author_source_pins",
        ):
            evidence[name] = [self.marker_pin]
        evidence_pin = self.write(folder / "READ_ONLY.json", evidence)
        accepted = {
            "schema": "sadhana.r1lite_frozen_full_expo_root_acceptance/1",
            "status": "accepted_frozen_full_expo_dev20",
            "owner": "/root",
            "cohort": kind,
            "run_id": run_id,
            "benchmark_accepted": True,
            "read_only_acceptance": evidence_pin,
            "observations": observations,
            "visual_review": {
                "status": "viewed_by_root",
                "reviewer": "/root",
                "files": [self.marker_pin],
                "seeds": [9100],
                "selection_reason": "Explicit CPU schema fixture only",
            },
        }
        if damage == "root_cohort":
            accepted["cohort"] = "wrong"
        if damage == "observations":
            accepted["observations"] = {**observations, "successes": 19}
        root_pin = self.write(folder / "ROOT.json", accepted)
        if damage == "source_pin":
            self.marker.write_text("Changed explicit CPU fixture")
        return Path(root_pin["path"])

    def publish_cohort(self, kind, damage=None):
        name = (
            "trained_full_expo"
            if kind == "trained"
            else "matched_initialized_full_expo"
        )
        acceptance = self.cohort(kind, damage)
        return publish(
            acceptance, self.viewer, self.root / f"publication-{kind}.json", name
        )

    def assert_history_preserved(self, after):
        for key, value in self.before.items():
            if key != "packaged_viewer_release":
                self.assertEqual(after[key], value)
        for name, data in self.history.items():
            self.assertEqual((self.viewer / name).read_bytes(), data)

    def test_each_new_cohort_preserves_historical_fields_and_all40_files(self):
        for kind in ("initialized", "trained"):
            with self.subTest(kind=kind):
                result = self.publish_cohort(kind)
                after = json.loads((self.viewer / "results.json").read_text())
                self.assert_history_preserved(after)
                self.assertEqual(len(result["videos"]), 20)
                for item in result["videos"]:
                    self.assertEqual(pin(Path(item["path"])), item)
        after = json.loads((self.viewer / "results.json").read_text())
        before, trained = (
            after["r1lite_matched_initialized_full_expo"],
            after["r1lite_trained_full_expo"],
        )
        self.assertEqual(before["learning_updates"], 0)
        self.assertEqual(trained["learning_updates"], 30)
        self.assertEqual(
            before["comparison_binding_sha256"], trained["comparison_binding_sha256"]
        )
        self.assertFalse(
            trained["producer_history_scope"]["training_metrics_available"]
        )

    def test_trained_can_publish_first_without_inventing_before_score(self):
        self.publish_cohort("trained")
        after = json.loads((self.viewer / "results.json").read_text())
        self.assertNotIn("r1lite_matched_initialized_full_expo", after)
        self.assert_history_preserved(after)

    def test_semantic_or_pin_damage_refuses_before_any_publication(self):
        for damage in (
            "canonical",
            "worker",
            "prng",
            "counter",
            "history_scope",
            "recipe",
            "media_pin",
            "media_frames",
            "media_count",
            "startup",
            "restore",
            "root_cohort",
            "observations",
        ):
            with self.subTest(damage=damage):
                before = (self.viewer / "results.json").read_bytes()
                index = (self.viewer / "index.html").read_bytes()
                with self.assertRaises(ValueError):
                    self.publish_cohort("trained", damage)
                self.assertEqual((self.viewer / "results.json").read_bytes(), before)
                self.assertEqual((self.viewer / "index.html").read_bytes(), index)
                self.assertFalse((self.root / "publication-trained.json").exists())
                self.assertEqual(len(list(self.viewer.glob("*.mp4"))), 40)

    def test_changed_source_is_refused(self):
        with self.assertRaises(ValueError):
            self.publish_cohort("trained", "source_pin")
        self.assertEqual(len(list(self.viewer.glob("*.mp4"))), 40)

    def test_comparison_contract_mismatch_preserves_existing_accepted_cohorts(self):
        self.publish_cohort("initialized")
        before = (self.viewer / "results.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "comparison contract"):
            self.publish_cohort("trained", "comparison")
        self.assertEqual((self.viewer / "results.json").read_bytes(), before)
        self.assertFalse((self.root / "publication-trained.json").exists())

    def test_lock_contention_does_not_write(self):
        acceptance = self.cohort("trained")
        before = (self.viewer / "results.json").read_bytes()
        with (self.viewer / ".original-methods-snapshot.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                publish(
                    acceptance,
                    self.viewer,
                    self.root / "publication-trained.json",
                    "trained_full_expo",
                )
        self.assertEqual((self.viewer / "results.json").read_bytes(), before)
        self.assertEqual(len(list(self.viewer.glob("*.mp4"))), 40)


if __name__ == "__main__":
    unittest.main()
