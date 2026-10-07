"""HTTP contract checks use explicit synthetic receipts, never benchmark evidence."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from abc_bench.dashboard import load_results, make_server


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_results_do_not_fabricate_runs(self):
        result, artifacts = load_results(self.root)
        self.assertEqual(result["runs"], [])
        self.assertEqual(artifacts, {})
        self.assertEqual(
            [e["id"] for e in result["embodiments"]], ["native_yam", "r1lite"]
        )
        self.assertTrue(
            all(e["status"] == "unavailable" for e in result["embodiments"])
        )

    def test_missing_metrics_remain_missing_and_artifacts_are_allowlisted(self):
        (self.root / "video.mp4").write_bytes(b"synthetic video")
        outside = self.root.parent / (self.root.name + "_secret")
        outside.write_text("not exposed")
        try:
            (self.root / "escape.mp4").symlink_to(outside)
            run = {
                "run_id": "synthetic",
                "metrics": {"successes": None},
                "artifacts": [
                    {"path": "video.mp4", "label": "Video"},
                    {"path": str(outside), "url": "https://evil.invalid"},
                    {"path": "escape.mp4"},
                ],
            }
            (self.root / "results.json").write_text(json.dumps({"runs": [run]}))
            result, artifacts = load_results(self.root)
            self.assertIsNone(result["runs"][0]["metrics"]["successes"])
            self.assertNotIn("episodes", result["runs"][0]["metrics"])
            self.assertEqual(list(artifacts.values()), [self.root / "video.mp4"])
            self.assertNotIn("url", result["runs"][0]["artifacts"][1])
            self.assertNotIn("url", result["runs"][0]["artifacts"][2])
        finally:
            outside.unlink()

    def test_invalid_receipt_reports_error(self):
        (self.root / "results.json").write_text('{"runs": "invalid"}')
        result, _ = load_results(self.root)
        self.assertEqual(result["runs"], [])
        self.assertIn("runs must be", result["errors"][0])

    def test_nonfinite_metrics_report_error(self):
        (self.root / "results.json").write_text(
            '{"runs": [{"metrics": {"reward": NaN}}]}'
        )
        result, _ = load_results(self.root)
        self.assertEqual(result["runs"], [])
        self.assertIn("Non-finite", result["errors"][0])

    def test_http_read_only_routes(self):
        (self.root / "clip.mp4").write_bytes(b"video evidence")
        (self.root / "private.txt").write_text("never exposed")
        (self.root / "results.json").write_text(
            json.dumps(
                {"runs": [{"run_id": "synthetic", "artifacts": [{"path": "clip.mp4"}]}]}
            )
        )
        server = make_server(self.root, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/results", timeout=3) as response:
                self.assertEqual(json.load(response)["runs"][0]["run_id"], "synthetic")
            _, artifact_paths = load_results(self.root)
            artifact_url = "/artifacts/" + next(iter(artifact_paths))
            with urlopen(base + artifact_url, timeout=3) as response:
                self.assertEqual(response.read(), b"video evidence")
            for path in (
                "/private.txt",
                "/artifacts/../private.txt",
                "/artifacts/%2e%2e%2fprivate.txt",
                "/artifacts/999",
            ):
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + path, timeout=3)
                self.assertEqual(error.exception.code, 404)
            with urlopen(base, timeout=3) as response:
                html = response.read().decode()
                self.assertIn("Unavailable values are not measured", html)
                self.assertIn("textContent", html)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
