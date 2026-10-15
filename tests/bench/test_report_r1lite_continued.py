"""CPU metadata fixtures; synthetic future files are not experiment evidence."""

import importlib.util
import json
import unittest
from pathlib import Path

from abc_bench.report_r1lite_expo import pin, publish

spec = importlib.util.spec_from_file_location(
    "historical_matched_fixture",
    Path(__file__).with_name("test_report_r1lite_matched.py"),
)
historical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(historical)
FIXTURE = Path(__file__).parents[1] / "fixtures" / "accepted_four_cohort_results.json"
FIELDS = (
    "r1lite",
    "r1lite_initialized_full_expo",
    "r1lite_matched_initialized_full_expo",
    "r1lite_trained_full_expo",
)


class ContinuedPublisherTests(unittest.TestCase):
    def setUp(self):
        self.f = historical.MatchedPublisherTests(methodName="runTest")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.before = json.loads(FIXTURE.read_text())
        self.f.write(self.f.viewer / "results.json", self.before)
        self.old_media = {}
        for key in FIELDS:
            for row in self.before[key]["episodes"]:
                path = self.f.viewer / row["video"]
                # Actual historical names/data; explicit synthetic media stand-ins.
                path.write_bytes(
                    ("CPU historical media placeholder " + path.name).encode()
                )
                self.old_media[path.name] = path.read_bytes()

    def future(self, defect=None):
        root_path = self.f.cohort("trained")
        accepted = json.loads(root_path.read_text())
        evidence_path = Path(accepted["read_only_acceptance"]["path"])
        evidence = json.loads(evidence_path.read_text())
        folder = root_path.parent
        intent_path = folder / "EVAL_EXEC_INTENT.json"
        intent = json.loads(intent_path.read_text())
        host_path = Path(intent["host_inspection"]["path"])
        parent = json.loads(host_path.read_text())
        parent_pin = self.f.write(folder / "PARENT.json", parent)
        # Distinct hypothetical endpoint proves that 10808/59 is not hardcoded.
        step, calls, new_calls = 10509, 50, 20
        checkpoint = folder / "checkpoints" / str(step)
        checkpoint.mkdir(parents=True)
        payload = checkpoint / "agent.bin"
        payload.write_bytes(b"Explicit CPU checkpoint fixture, not a model")
        counters = {
            key: {"step": count}
            for key, count in {
                "actor_train_state": calls,
                "edit_actor": calls,
                "temp": calls,
                "critic": calls * 20,
                "filter_critic": calls * 20,
                "batch_encoder": calls * 20,
            }.items()
        }
        final = {
            "training/" + key: 0.25
            for key in (
                "actor_loss",
                "actor_grad_norm",
                "critic_loss",
                "filter_critic_loss",
                "edit_actor_loss",
                "temperature",
                "temperature_loss",
            )
        }
        final.update(
            {"training/actor_state_step": calls - 1, "training/next_q_nan_ratio": 0}
        )
        host = {
            "status": "resumed_checkpoint_replay_complete_history_verified",
            "run_id": "explicit-cpu-new-producer",
            "checkpoint": str(checkpoint),
            "artifact_accepted": False,
            "lifecycle_accepted": False,
            "all_inspected_files_unchanged": True,
            "parent_artifacts_unchanged": True,
            "parent_report": parent_pin,
            "schedule": {
                "checkpoint_step": step,
                "total_update_calls": calls,
                "new_update_calls": new_calls,
            },
            "history": {
                "raw_history_complete": True,
                "final_boundary_row_present": True,
                "last_step": step,
                "final_row": final,
            },
            "model": {"counters": counters},
            "checkpoint_files": [pin(payload)],
            **{
                key: self.f.marker_pin
                for key in (
                    "history_file",
                    "native_events",
                    "author_log",
                    "source_manifest",
                )
            },
        }
        if defect == "incomplete_history":
            host["history"]["raw_history_complete"] = False
        if defect == "missing_final":
            host["history"]["final_boundary_row_present"] = False
        if defect == "nonfinite":
            final["training/critic_loss"] = float("inf")
        if defect == "missing_metric":
            final.pop("training/actor_loss")
        if defect == "lineage":
            host["schedule"]["new_update_calls"] = new_calls + 1
        if defect == "counter":
            counters["critic"]["step"] -= 1
        if defect == "inventory":
            host["checkpoint_files"] = []
        if defect == "extra_checkpoint":
            (checkpoint / "extra.bin").write_bytes(b"extra")
        if defect == "missing_history_file":
            host.pop("history_file")
        host_pin = self.f.write(host_path, host)
        producer = {
            "status": "root_accepted_original_online_checkpoint",
            "owner": "/root",
            "artifact_accepted": defect != "not_accepted",
            "run_id": host["run_id"],
            "checkpoint": str(checkpoint),
            "checkpoint_step": step,
            "host_inspection": host_pin,
        }
        producer_pin = self.f.write(folder / "PRODUCER.json", producer)
        config_path = Path(intent["config"]["path"])
        config = json.loads(config_path.read_text())
        config["arguments"] = [
            f"--checkpoint_step={step}"
            if a.startswith("--checkpoint_step=")
            else f"--checkpoint_dir={checkpoint.parent}"
            if a.startswith("--checkpoint_dir=")
            else a
            for a in config["arguments"]
        ]
        config_pin = self.f.write(config_path, config)
        intent.update(
            checkpoint_step=step,
            producer_run_id=host["run_id"],
            accepted_producer_checkpoint=str(checkpoint),
            config=config_pin,
            root_artifact_acceptance=producer_pin,
            host_inspection=host_pin,
        )
        intent_pin = self.f.write(intent_path, intent)
        evidence.update(
            checkpoint_step=step,
            producer_run_id=host["run_id"],
            accepted_producer_checkpoint=str(checkpoint),
            launch_pins=[config_pin, intent["job"], intent["settings"]],
            producer_artifact_pins=[producer_pin, host_pin],
            artifact_pins=[self.f.marker_pin, intent_pin],
        )
        evidence["frozen_scope"]["evaluation_checkpoint_step"] = step
        evidence["accepted_scalar_observations"]["full_agent_restore_marker"] = [
            str(step)
        ]
        evidence["author_native_comparison"] = evidence["accepted_scalar_observations"]
        if defect == "missing_media":
            evidence["media"].pop()
        if defect == "restore":
            evidence["accepted_scalar_observations"]["full_agent_restore_marker"] = []
        accepted["observations"] = evidence["accepted_scalar_observations"]
        accepted["read_only_acceptance"] = self.f.write(evidence_path, evidence)
        self.f.write(root_path, accepted)
        return root_path

    def test_append_preserves_actual_historical_data_and_all_media(self):
        result = publish(
            self.future(),
            self.f.viewer,
            self.f.root / "publication.json",
            "continued_full_expo",
        )
        after = json.loads((self.f.viewer / "results.json").read_text())
        for key, value in self.before.items():
            if key != "packaged_viewer_release":
                self.assertEqual(after[key], value)
        for name, data in self.old_media.items():
            self.assertEqual((self.f.viewer / name).read_bytes(), data)
        new = after["r1lite_continued_full_expo"]
        self.assertEqual(
            (new["evaluation_checkpoint_step"], new["learning_updates"]), (10509, 50)
        )
        self.assertEqual(new["producer_update_counters"]["critic"], 1000)
        self.assertFalse(new["producer_history_scope"]["parent_history_complete"])
        self.assertEqual(len(result["videos"]), 20)
        self.assertEqual(
            after["packaged_viewer_release"]["previous"],
            self.before["packaged_viewer_release"],
        )
        self.assertEqual(
            Path(result["previous_results"]["path"]).read_bytes(),
            json.dumps(self.before).encode(),
        )

    def test_duplicate_slot_is_refused_without_replacing_results(self):
        accepted = self.future()
        publish(
            accepted, self.f.viewer, self.f.root / "first.json", "continued_full_expo"
        )
        before = (self.f.viewer / "results.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "append-only"):
            publish(
                accepted,
                self.f.viewer,
                self.f.root / "duplicate.json",
                "continued_full_expo",
            )
        self.assertEqual((self.f.viewer / "results.json").read_bytes(), before)
        self.assertFalse((self.f.root / "duplicate.json").exists())

    def test_missing_or_bad_future_evidence_is_refused_before_publication(self):
        defects = (
            "incomplete_history",
            "missing_final",
            "nonfinite",
            "missing_metric",
            "lineage",
            "counter",
            "inventory",
            "extra_checkpoint",
            "missing_history_file",
            "not_accepted",
            "missing_media",
            "restore",
        )
        for defect in defects:
            with self.subTest(defect=defect):
                # Independent temporary fixture for each failed publication.
                case = ContinuedPublisherTests(methodName="runTest")
                case.setUp()
                try:
                    before = (case.f.viewer / "results.json").read_bytes()
                    index = (case.f.viewer / "index.html").read_bytes()
                    media = set(case.f.viewer.glob("*.mp4"))
                    with self.assertRaises((ValueError, KeyError)):
                        publish(
                            case.future(defect),
                            case.f.viewer,
                            case.f.root / "refused.json",
                            "continued_full_expo",
                        )
                    self.assertEqual(
                        (case.f.viewer / "results.json").read_bytes(), before
                    )
                    self.assertEqual((case.f.viewer / "index.html").read_bytes(), index)
                    self.assertEqual(set(case.f.viewer.glob("*.mp4")), media)
                    self.assertFalse((case.f.root / "refused.json").exists())
                finally:
                    case.doCleanups()
