"""Explicit synthetic CPU evidence/media; no experiment or decoding claims."""

import copy
import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from abc_bench import report_r1lite_expo as report

HISTORY = Path(__file__).parents[1] / "fixtures/accepted_five_cohort_results.json"


class Case:
    def __init__(self, root):
        self.root = root
        self.viewer = root / "viewer"
        self.viewer.mkdir()
        self.before = json.loads(HISTORY.read_text())
        self.write(self.viewer / "results.json", self.before)
        (self.viewer / "index.html").write_text("Explicit old CPU template")
        self.old_media = {}
        for key, value in self.before.items():
            if key.startswith("r1lite") and isinstance(value, dict):
                for row in value.get("episodes", []):
                    name = row["video"]
                    payload = ("Explicit historical media fixture " + name).encode()
                    (self.viewer / name).write_bytes(payload)
                    self.old_media[name] = payload
        marker = self.write(root / "marker.json", {"explicit_cpu_fixture": True})
        self.campaign = self.write(root / "campaign.json", {"files": [marker]})
        self.operator = self.write(
            root / "analyzer-manifest.json", {"explicit_cpu_fixture": True}
        )
        self.roots, self.analyses, self.root_paths, self.analysis_paths = {}, {}, {}, {}
        self.comparison = {
            "schema": "sadhana.expo_train_comparison_root_acceptance/1",
            "status": "accepted_three_mode_train_diagnostic",
            "owner": "/root",
            "artifact_accepted": True,
            "split": "TRAIN",
            "initialization_seed": 42,
            "online_updates": 0,
            "configuration_manifest": self.campaign,
            "conditions": {},
        }
        for mode_index, condition in enumerate(report.TRAIN_CONDITIONS):
            folder = root / condition
            folder.mkdir()
            run_id = "explicit-cpu-" + condition
            worker = self.write(
                folder / "worker.json",
                {
                    "status": "completed",
                    "author_exit_code": 0,
                    "client_exit_code": 0,
                    "owned_children_stopped": True,
                    "cleanup_errors": [],
                    "author_shutdown": "already_stopped",
                    "client_shutdown": "already_stopped",
                },
            )
            canonical = self.write(
                folder / "canonical.json",
                {
                    "status": "completed",
                    "run_id": run_id,
                    "worker_exit_code": 0,
                    "worker_receipt_sha256": worker["sha256"],
                },
            )
            videos, media, scored = [], [], []
            for index, seed in enumerate([20999, *report.TRAIN_SEEDS]):
                success = index > 0 and index <= mode_index
                controls = 0 if not index else 7 if success else 900
                video = folder / f"{seed}.mp4"
                video.write_bytes(f"Explicit CPU video {condition} {seed}".encode())
                sidecar = self.write(
                    video.with_suffix(".json"),
                    {
                        "seed": seed,
                        "native_controls": controls,
                        "frames": controls + 1,
                        "fps": 15,
                        "complete_native_episode": index > 0,
                        "end_reason": "terminal" if index else "reset",
                    },
                )
                item = {
                    "episode": index + 1,
                    "seed": seed,
                    "scored": index > 0,
                    "video": report.pin(video),
                    "sidecar": sidecar,
                    "declared_frames": controls + 1,
                }
                videos.append(item)
                media.append(
                    {
                        **item,
                        "full_decode_exit_code": 0,
                        "probe": {
                            "nb_read_frames": str(controls + 1),
                            "width": 640,
                            "height": 360,
                            "avg_frame_rate": "15/1",
                        },
                    }
                )
                if index:
                    scored.append(
                        {
                            "seed": seed,
                            "native_controls": controls,
                            "success": success,
                            "failure": False,
                            "truncated": not success,
                            "author": {
                                "success": success,
                                "return": float(success),
                                "len": controls + 1,
                            },
                        }
                    )
            analysis = {
                "schema": "sadhana.expo_multistart_read_only_analysis/1",
                "condition": condition,
                "run_id": run_id,
                "status": "completed_trace_and_media_inventory_checks_passed",
                "artifact_accepted": False,
                "configuration_manifest": self.campaign,
                "operator_manifest": self.operator,
                "analysis_source": marker,
                "episodes": 4,
                "startup": {"seed": 20999, "native_controls": 0},
                "canonical_status": "completed",
                "worker_status": "completed",
                "author_exit_code": 0,
                "client_exit_code": 0,
                "input_pins": [canonical, worker, marker],
                "scored": scored,
                "successes": mode_index,
                "videos": videos,
                "media_decode_performed": False,
                "frames_viewed": [],
            }
            accepted = {
                "schema": "sadhana.expo_train_condition_root_acceptance/1",
                "status": "accepted_four_train_start_diagnostic",
                "owner": "/root",
                "artifact_accepted": True,
                "condition": condition,
                "run_id": run_id,
                "execution": {
                    "canonical": canonical,
                    "worker": worker,
                    "natural_shutdown": True,
                },
                "initialization": {
                    "seed": 42,
                    "actor_weight_checkpoint_label": 2000,
                    "online_updates": 0,
                    "full_agent_restored": False,
                    "evidence": marker,
                },
                "immutability": {
                    "rtc2000_unchanged": True,
                    "runtime_source_task_unchanged": True,
                    "evidence": marker,
                },
                "resource_cleanup": marker,
                "visual_review": {
                    "status": "viewed_by_root",
                    "reviewer": "/root",
                    "files": [marker],
                    "seeds": [21000],
                    "selection_reason": "Explicit CPU fixture, no actual viewing",
                },
                "media": media,
            }
            self.analyses[condition], self.roots[condition] = analysis, accepted
            self.root_paths[condition] = folder / "ROOT.json"
            self.analysis_paths[condition] = folder / "ANALYSIS.json"
        self.acceptance = root / "COMPARISON.json"
        self.receipt = root / "PUBLICATION.json"
        self.flush()

    def write(self, path, value):
        path.write_text(json.dumps(value, allow_nan=False))
        return report.pin(path)

    def flush(self):
        for condition in report.TRAIN_CONDITIONS:
            self.roots[condition]["read_only_analysis"] = self.write(
                self.analysis_paths[condition], self.analyses[condition]
            )
            self.comparison["conditions"][condition] = self.write(
                self.root_paths[condition], self.roots[condition]
            )
        self.write(self.acceptance, self.comparison)

    def publish(self):
        # Only campaign/analyzer seal constants are explicit CPU doubles.
        with patch.multiple(
            report,
            TRAIN_CAMPAIGN_SHA256=self.campaign["sha256"],
            TRAIN_ANALYZER_SHA256=self.operator["sha256"],
        ):
            return report.publish(
                self.acceptance, self.viewer, self.receipt, "train_diagnostics"
            )


class TrainPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def case(self, name="case"):
        root = self.root / name
        root.mkdir()
        return Case(root)

    def test_append_preserves_actual_five_cohorts_and_all100_media(self):
        case = self.case()
        result = case.publish()
        after = json.loads((case.viewer / "results.json").read_text())
        self.assertEqual(
            {
                k: v
                for k, v in after.items()
                if k not in ("packaged_viewer_release", "r1lite_expo_train_diagnostics")
            },
            {k: v for k, v in case.before.items() if k != "packaged_viewer_release"},
        )
        self.assertEqual(len(case.old_media), 100)
        for name, data in case.old_media.items():
            self.assertEqual((case.viewer / name).read_bytes(), data)
        self.assertEqual(len(result["videos"]), 12)
        value = after["r1lite_expo_train_diagnostics"]
        self.assertEqual(
            [v["successes"] for v in value["conditions"].values()], [0, 1, 2]
        )
        self.assertNotIn("successes", value)
        for condition, data in value["conditions"].items():
            self.assertEqual(
                [v["seed"] for v in data["episodes"]], [21000, 21001, 21002, 21003]
            )
            startup = case.roots[condition]["media"][0]["video"]["sha256"][:16] + ".mp4"
            self.assertFalse((case.viewer / startup).exists())
        self.assertEqual(
            after["packaged_viewer_release"]["previous"],
            case.before["packaged_viewer_release"],
        )
        self.assertEqual(
            after["packaged_viewer_release"]["template"],
            report.pin(case.viewer / "index.html"),
        )

    def test_duplicate_is_refused_without_replacing_history(self):
        case = self.case()
        case.publish()
        before = (case.viewer / "results.json").read_bytes()
        case.receipt = case.root / "SECOND.json"
        with self.assertRaisesRegex(ValueError, "append-only"):
            case.publish()
        self.assertEqual((case.viewer / "results.json").read_bytes(), before)
        self.assertFalse(case.receipt.exists())

    def test_bad_root_scope_and_condition_binding_are_refused(self):
        for defect in (
            "root",
            "missing_mode",
            "split",
            "updates",
            "seed",
            "condition",
            "restored",
            "immutability",
            "visual",
            "analysis_accepted",
            "campaign",
            "analyzer",
        ):
            with self.subTest(defect=defect):
                case = self.case(defect)
                accepted, analysis = case.roots["full"], case.analyses["full"]
                if defect == "root":
                    accepted["artifact_accepted"] = False
                if defect == "missing_mode":
                    case.comparison["conditions"].pop("base")
                if defect == "split":
                    case.comparison["split"] = "DEV"
                if defect == "updates":
                    accepted["initialization"]["online_updates"] = 1
                if defect == "seed":
                    accepted["initialization"]["seed"] = 43
                if defect == "condition":
                    accepted["condition"] = "base"
                if defect == "restored":
                    accepted["initialization"]["full_agent_restored"] = True
                if defect == "immutability":
                    accepted["immutability"]["rtc2000_unchanged"] = False
                if defect == "visual":
                    accepted["visual_review"]["seeds"] = [20999]
                if defect == "analysis_accepted":
                    analysis["artifact_accepted"] = True
                if defect == "campaign":
                    analysis["configuration_manifest"] = case.operator
                if defect == "analyzer":
                    analysis["operator_manifest"] = case.campaign
                case.flush()
                if defect == "missing_mode":
                    case.comparison["conditions"].pop("base")
                    case.write(case.acceptance, case.comparison)
                self.assert_refused(case)

    def test_outcome_media_and_decode_refusals(self):
        for defect in (
            "order",
            "partial",
            "startup",
            "score",
            "author",
            "terminal",
            "missing_video",
            "decode",
            "frames",
            "fps",
            "resolution",
            "media_pin",
        ):
            with self.subTest(defect=defect):
                case = self.case(defect)
                analysis, accepted = case.analyses["full"], case.roots["full"]
                if defect == "order":
                    analysis["scored"][0]["seed"] = 21001
                if defect == "partial":
                    analysis["scored"].pop()
                if defect == "startup":
                    analysis["startup"]["native_controls"] = 1
                if defect == "score":
                    analysis["successes"] = 1
                if defect == "author":
                    analysis["scored"][0]["author"]["len"] = 900
                if defect == "terminal":
                    analysis["scored"][0]["truncated"] = False
                if defect == "missing_video":
                    accepted["media"].pop()
                if defect == "decode":
                    accepted["media"][1]["full_decode_exit_code"] = 1
                if defect == "frames":
                    accepted["media"][1]["probe"]["nb_read_frames"] = "900"
                if defect == "fps":
                    accepted["media"][1]["probe"]["avg_frame_rate"] = "30/1"
                if defect == "resolution":
                    accepted["media"][1]["probe"]["width"] = 320
                if defect == "media_pin":
                    accepted["media"][1]["video"] = copy.deepcopy(
                        accepted["media"][2]["video"]
                    )
                case.flush()
                self.assert_refused(case)

    def test_execution_provenance_and_changed_input_are_refused(self):
        for defect in ("failed", "worker_hash", "unbound", "changed"):
            with self.subTest(defect=defect):
                case = self.case(defect)
                execution = case.roots["full"]["execution"]
                path = Path(execution["canonical"]["path"])
                value = json.loads(path.read_text())
                if defect == "failed":
                    value["status"] = "failed"
                if defect == "worker_hash":
                    value["worker_receipt_sha256"] = "0" * 64
                item = case.write(path, value)
                execution["canonical"] = item
                if defect != "unbound":
                    case.analyses["full"]["input_pins"][0] = item
                case.flush()
                if defect == "changed":
                    path.write_text("Changed after acceptance")
                if defect == "unbound":
                    execution["canonical"] = case.write(
                        case.root / "unbound.json", value
                    )
                    case.flush()
                self.assert_refused(case)

    def test_lock_contention_does_not_copy_or_publish(self):
        case = self.case()
        before = (case.viewer / "results.json").read_bytes()
        with (case.viewer / ".original-methods-snapshot.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                case.publish()
        self.assertEqual((case.viewer / "results.json").read_bytes(), before)
        self.assertEqual(len(list(case.viewer.glob("*.mp4"))), 100)

    def test_content_collision_is_refused_and_old_result_stays(self):
        case = self.case()
        video = case.roots["full"]["media"][1]["video"]
        (case.viewer / (video["sha256"][:16] + ".mp4")).write_bytes(
            b"Explicit collision"
        )
        before = (case.viewer / "results.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "collision"):
            case.publish()
        self.assertEqual((case.viewer / "results.json").read_bytes(), before)

    def assert_refused(self, case):
        before = (case.viewer / "results.json").read_bytes()
        with self.assertRaises((ValueError, KeyError, TypeError)):
            case.publish()
        self.assertEqual((case.viewer / "results.json").read_bytes(), before)
        self.assertEqual(len(list(case.viewer.glob("*.mp4"))), 100)
        self.assertFalse(case.receipt.exists())


if __name__ == "__main__":
    unittest.main()
