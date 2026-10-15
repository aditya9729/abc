"""Installed CPU publication checks using real accepted R1 artifacts.

Run after root accepts the complete initialized cohort. No test fabricates a
successful simulator run, invokes ffmpeg, imports a model, or changes source
artifacts. Negative cases alter only copied JSON in a temporary directory.
"""

import copy
import fcntl
import json
import os
import tempfile
import unittest
from pathlib import Path

from abc_bench.report_r1lite_expo import pin, publish

C = Path("/home/user/aditya/RL/builds/r1lite-realtime-expo-20261014")


class InitializedPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.acceptance = Path(
            os.environ.get(
                "R1LITE_INITIALIZED_ACCEPTANCE_FIXTURE",
                str(
                    C / "online-launch-20261015/INITIALIZED_DEV20_ROOT_ACCEPTANCE.json"
                ),
            )
        )
        self.original = json.loads(self.acceptance.read_text())
        self.evidence = json.loads(
            Path(self.original["read_only_acceptance"]["path"]).read_text()
        )
        self.viewer = self.root / "viewer"
        self.viewer.mkdir()
        source = Path(
            os.environ.get("R1LITE_VIEWER_FIXTURE", str(C / "viewer-release-01/stage"))
        )
        self.before = json.loads((source / "results.json").read_text())
        self.assertEqual(self.before["r1lite"]["successes"], 8)
        self.before.pop("r1lite_initialized_full_expo", None)
        (self.viewer / "results.json").write_text(json.dumps(self.before))
        (self.viewer / "index.html").write_bytes((source / "index.html").read_bytes())
        # An existing artifact detects unrelated file writes without duplicating historical media.
        (self.viewer / "historical-artifact.bin").write_bytes(
            b"preserve existing artifact"
        )
        self.receipt = self.root / "publication.json"

    def copied_acceptance(self, change_root=None, change_evidence=None):
        accepted, evidence = copy.deepcopy(self.original), copy.deepcopy(self.evidence)
        if change_evidence:
            change_evidence(evidence)
        evidence_path = self.root / "copied-evidence.json"
        evidence_path.write_text(json.dumps(evidence))
        accepted["read_only_acceptance"] = pin(evidence_path)
        if change_root:
            change_root(accepted)
        root_path = self.root / "copied-acceptance.json"
        root_path.write_text(json.dumps(accepted))
        return root_path

    def assert_rejected_without_publication(self, acceptance, pattern):
        old_results = (self.viewer / "results.json").read_bytes()
        old_index = (self.viewer / "index.html").read_bytes()
        with self.assertRaisesRegex(ValueError, pattern):
            publish(acceptance, self.viewer, self.receipt, "initialized_full_expo")
        self.assertEqual((self.viewer / "results.json").read_bytes(), old_results)
        self.assertEqual((self.viewer / "index.html").read_bytes(), old_index)
        self.assertFalse(self.receipt.exists())
        self.assertFalse(list(self.viewer.glob("*.mp4")))

    def test_real_initialized_cohort_preserves_rtc_and_historical_fields(self):
        result = publish(
            self.acceptance, self.viewer, self.receipt, "initialized_full_expo"
        )
        after = json.loads((self.viewer / "results.json").read_text())
        for name, value in self.before.items():
            if name != "packaged_viewer_release":
                self.assertEqual(after[name], value, name)
        self.assertEqual(
            (self.viewer / "historical-artifact.bin").read_bytes(),
            b"preserve existing artifact",
        )
        cohort = after["r1lite_initialized_full_expo"]
        self.assertEqual(
            cohort["successes"],
            self.evidence["accepted_scalar_observations"]["successes"],
        )
        self.assertEqual(cohort["learning_updates"], 0)
        self.assertEqual(cohort["evaluation_checkpoint_step"], 0)
        self.assertEqual(len(result["videos"]), 20)
        for row in result["videos"]:
            self.assertEqual(pin(Path(row["path"])), row)
        self.assertEqual(
            after["packaged_viewer_release"]["template"],
            pin(self.viewer / "index.html"),
        )

    def test_root_observation_mismatch_is_rejected(self):
        def change(accepted):
            accepted["observations"] = copy.deepcopy(
                self.evidence["accepted_scalar_observations"]
            )
            accepted["observations"]["successes"] += 1

        self.assert_rejected_without_publication(
            self.copied_acceptance(change_root=change), "disagree"
        )

    def test_failed_canonical_receipt_cannot_be_recovered_by_publisher(self):
        def change(evidence):
            path = self.root / "failed-canonical.json"
            canonical = json.loads(
                Path(evidence["execution"]["canonical"]["path"]).read_text()
            )
            canonical["status"] = "failed"
            path.write_text(json.dumps(canonical))
            evidence["execution"]["canonical"] = pin(path)

        self.assert_rejected_without_publication(
            self.copied_acceptance(change_evidence=change), "clean completed"
        )

    def test_last_scored_video_pin_is_checked_before_publication(self):
        def change(evidence):
            evidence["media"][-1]["video"]["sha256"] = "0" * 64

        self.assert_rejected_without_publication(
            self.copied_acceptance(change_evidence=change), "artifact changed"
        )

    def test_native_length_mismatch_is_rejected(self):
        def change(evidence):
            evidence["native"]["episodes"][-1]["native_controls"] -= 1

        self.assert_rejected_without_publication(
            self.copied_acceptance(change_evidence=change), "native/media evidence"
        )

    def test_changed_read_only_artifact_is_rejected(self):
        def change(evidence):
            original = evidence["artifact_pins"][0]
            path = self.root / "changed-author.log"
            path.write_bytes(
                Path(original["path"]).read_bytes()
                + b"\nexplicit CPU corruption fixture\n"
            )
            evidence["artifact_pins"][0] = {**original, "path": str(path)}

        self.assert_rejected_without_publication(
            self.copied_acceptance(change_evidence=change), "artifact changed"
        )

    def test_existing_publication_lock_still_prevents_writes(self):
        before = (self.viewer / "results.json").read_bytes()
        with (self.viewer / ".original-methods-snapshot.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                publish(
                    self.acceptance, self.viewer, self.receipt, "initialized_full_expo"
                )
        self.assertEqual((self.viewer / "results.json").read_bytes(), before)
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()
