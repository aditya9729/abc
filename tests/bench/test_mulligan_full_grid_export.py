"""Candidate HTML resource fixture; not an installed candidate-wheel proof."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from abc_bench.export_original_methods import export_viewer


class FullGridExportTests(unittest.TestCase):
    def test_candidate_template_exports_without_changing_existing_artifacts(self):
        resource_fixture = Path(__file__).resolve().parents[2] / "abc_bench"
        expected = (
            resource_fixture / "static/original-methods/index.html"
        ).read_bytes()
        self.assertIn(b'id="mulligan-local-divl-full-grid" hidden', expected)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            originals = {
                "results.json": b'{"explicit_synthetic_fixture":true}',
                "previous.mp4": b"explicit synthetic media fixture",
            }
            for name, data in originals.items():
                (output / name).write_bytes(data)
            with patch(
                "abc_bench.export_original_methods.files", return_value=resource_fixture
            ):
                destination = export_viewer(output)
            self.assertEqual(destination.read_bytes(), expected)
            self.assertEqual(
                {name: (output / name).read_bytes() for name in originals}, originals
            )
            self.assertEqual(
                {path.name for path in output.iterdir()}, {"index.html", *originals}
            )


if __name__ == "__main__":
    unittest.main()
