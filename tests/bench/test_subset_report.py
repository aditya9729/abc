"""Pure stdlib reporting fixtures; byte blobs are not simulator videos or scores."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from abc_bench.subset_report import SEEDS, read_condition


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def pin(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class ReportFixture:
    def __init__(self, root: Path):
        self.run, self.output = root / "fixture-run", root / "published"
        self.run.mkdir()
        self.output.mkdir()
        self.worker = {
            "status": "completed",
            "evaluation_events": [],
            "rollouts": [],
            "subset_video_recordings": [],
        }

    def cohort(
        self, event_index: int, *, success: tuple[bool, ...], actual_delta: int = 0
    ):
        stage = f"evaluate-learned-{event_index}"
        videos = {}
        for index, (seed, succeeded) in enumerate(zip(SEEDS, success, strict=True)):
            actual = seed + actual_delta
            video = self.run / f"fixture-bytes-event-{event_index}-seed-{seed}.mp4"
            video.write_bytes(
                f"explicit fixture bytes: event {event_index}; seed {seed}".encode()
            )
            video_ref = pin(video)
            videos[seed] = video_ref
            self.worker["subset_video_recordings"].append(
                {
                    "requested_seed": seed,
                    "actual_seed": actual,
                    "label": f"{stage}-{index}",
                    "domain": "sim",
                    "finalized": True,
                    "video": video_ref,
                }
            )
            rollout = self.run / f"fixture-rollout-{event_index}-{index}.json"
            write_json(
                rollout,
                {
                    "stage": stage,
                    "domain": "sim",
                    "rollout": {
                        "worlds": [
                            {
                                "requested_seed": seed,
                                "actual_seed": actual,
                                "steps": 1000,
                                "completed": True,
                                "paper_success": succeeded,
                                "native_success": False,
                            }
                        ]
                    },
                },
            )
            self.worker["rollouts"].append(pin(rollout))
        return videos

    def publish_receipt(self) -> None:
        write_json(self.run / "receipt.json", self.worker)


class SubsetReportTests(unittest.TestCase):
    def test_intermediate_learned_event_never_scores_final(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReportFixture(Path(directory))
            fixture.worker["evaluation_events"] = [
                {"head": "learned", "reason": "periodic"}
            ]
            fixture.cohort(0, success=(True, True, True))
            fixture.publish_receipt()
            result = read_condition(fixture.run, fixture.output, qf3=True)
            self.assertIsNone(result["score"])
            self.assertEqual(result["episodes"], [])
            self.assertNotEqual(result["status"], "Complete")
            self.assertEqual(list(fixture.output.iterdir()), [])

    def test_exact_final_event_selects_three_completed_layouts_and_their_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReportFixture(Path(directory))
            fixture.worker["evaluation_events"] = [
                {"head": "learned", "reason": "periodic"},
                {"head": "learned", "reason": "final"},
                {"head": "learned", "reason": "periodic"},
            ]
            fixture.cohort(0, success=(True, True, True))
            expected = fixture.cohort(1, success=(True, False, True))
            fixture.cohort(2, success=(False, False, False))
            # A partial world in the selected event must not enter the cohort.
            partial = fixture.run / "fixture-partial-world.json"
            write_json(
                partial,
                {
                    "stage": "evaluate-learned-1",
                    "domain": "sim",
                    "rollout": {
                        "worlds": [{"requested_seed": 9011, "completed": False}]
                    },
                },
            )
            fixture.worker["rollouts"].append(pin(partial))
            fixture.publish_receipt()
            result = read_condition(fixture.run, fixture.output, qf3=True)
            self.assertEqual(result["score"], "2/3")
            self.assertEqual(result["status"], "Complete")
            self.assertEqual(tuple(row["seed"] for row in result["episodes"]), SEEDS)
            self.assertEqual(len(list(fixture.output.iterdir())), 3)
            for row in result["episodes"]:
                ref = expected[row["seed"]]
                self.assertEqual(row["actual_seed"], row["seed"])
                self.assertEqual(row["video"], ref["sha256"][:16] + ".mp4")
                self.assertEqual(
                    (fixture.output / row["video"]).read_bytes(),
                    Path(ref["path"]).read_bytes(),
                )

    def test_changed_actual_seed_refuses_both_native_host_and_qf3(self):
        for qf3 in (False, True):
            with self.subTest(qf3=qf3), tempfile.TemporaryDirectory() as directory:
                fixture = ReportFixture(Path(directory))
                if qf3:
                    fixture.worker["evaluation_events"] = [
                        {"head": "learned", "reason": "final"}
                    ]
                    fixture.cohort(0, success=(True, True, True), actual_delta=100_000)
                    fixture.publish_receipt()
                else:
                    write_json(
                        fixture.run / "episode-0001.json",
                        {
                            "requested_seed": 9011,
                            "seed": 109011,
                            "completed": True,
                        },
                    )
                with self.assertRaisesRegex(ValueError, "actual layout differs"):
                    read_condition(fixture.run, fixture.output, qf3=qf3)
                self.assertEqual(list(fixture.output.iterdir()), [])

    def test_tampered_video_hash_refuses_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReportFixture(Path(directory))
            fixture.worker["evaluation_events"] = [
                {"head": "learned", "reason": "final"}
            ]
            videos = fixture.cohort(0, success=(True, True, True))
            fixture.publish_receipt()
            Path(videos[9011]["path"]).write_bytes(b"tampered explicit fixture bytes")
            with self.assertRaisesRegex(
                ValueError, "Video differs from its recorded identity"
            ):
                read_condition(fixture.run, fixture.output, qf3=True)
            self.assertEqual(list(fixture.output.iterdir()), [])
