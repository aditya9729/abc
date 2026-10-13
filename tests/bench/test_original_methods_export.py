"""Installed CPU-only checks for static export and existing artifact preservation."""

import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from abc_bench.export_original_methods import export_viewer


class ExportViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_exact_template_and_existing_result_artifacts_are_preserved(self):
        output = self.root / "viewer"
        output.mkdir()
        (output / "index.html").write_bytes(b"old template")
        originals = {
            "results.json": b'{"explicit_fixture": true}',
            "video.mp4": b"explicit video fixture",
        }
        for name, data in originals.items():
            (output / name).write_bytes(data)
        destination = export_viewer(output)
        expected = (
            files("abc_bench")
            .joinpath("static/original-methods/index.html")
            .read_bytes()
        )
        self.assertEqual(destination, output / "index.html")
        self.assertEqual(destination.read_bytes(), expected)
        self.assertEqual(
            {name: (output / name).read_bytes() for name in originals}, originals
        )
        self.assertEqual(
            {path.name for path in output.iterdir()}, {"index.html", *originals}
        )

    def test_new_output_directory_is_created(self):
        self.assertTrue(export_viewer(self.root / "new" / "viewer").is_file())

    def test_file_output_path_is_rejected_without_modification(self):
        output = self.root / "file"
        output.write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "must be a directory"):
            export_viewer(output)
        self.assertEqual(output.read_bytes(), b"preserve")

    def test_relative_or_symlink_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute ordinary directory"):
            export_viewer(Path("relative-explicit-CPU-fixture"))
        linked = self.root / "linked"
        linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "absolute ordinary directory"):
            export_viewer(linked)

    def test_symlink_or_directory_index_is_rejected(self):
        index = self.root / "index.html"
        index.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "ordinary file"):
            export_viewer(self.root)
        index.unlink()
        index.mkdir()
        with self.assertRaisesRegex(ValueError, "ordinary file"):
            export_viewer(self.root)

    def test_failed_atomic_replace_preserves_old_index_and_cleans_temporary(self):
        index = self.root / "index.html"
        index.write_bytes(b"old template")
        with (
            patch(
                "abc_bench.export_original_methods.os.replace",
                side_effect=OSError("explicit CPU write failure"),
            ),
            self.assertRaisesRegex(OSError, "explicit CPU write failure"),
        ):
            export_viewer(self.root)
        self.assertEqual(index.read_bytes(), b"old template")
        self.assertEqual([path.name for path in self.root.iterdir()], ["index.html"])


if __name__ == "__main__":
    unittest.main()
