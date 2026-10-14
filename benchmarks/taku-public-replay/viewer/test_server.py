"""Bounded actual HTTP checks on an ephemeral loopback port and synthetic files."""

import hashlib
import http.client
import json
import runpy
import tempfile
import threading
import unittest
from pathlib import Path

PUBLIC = runpy.run_path(str(Path(__file__).with_name("server.py")))


class PlaybackServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.video = bytes(range(256)) * 4
        self.names = []
        manifest = {"schema": "taku.local-scene-viewer/v1"}
        for key, data in zip(
            PUBLIC["MEDIA_KEYS"], (self.video, self.video + b"second"), strict=True
        ):
            digest = hashlib.sha256(data).hexdigest()
            name = digest[:16] + ".mp4"
            self.names.append(name)
            (self.directory / name).write_bytes(data)
            manifest[key] = {"status": "accepted_complete", "video": name, "sha256": digest}
        (self.directory / "media.json").write_text(json.dumps(manifest))
        (self.directory / "index.html").write_bytes(b"explicit synthetic HTML")
        (self.directory / "source-reference.jpg").write_bytes(b"explicit synthetic poster")
        (self.directory / "private.txt").write_bytes(b"must not be served")
        self.server = PUBLIC["make_server"](self.directory, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def request(self, route, method="GET", headers=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=2
        )
        try:
            connection.request(method, route, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_exact_five_files_full_get_and_head(self):
        names = ("index.html", "media.json", "source-reference.jpg", *self.names)
        for name in names:
            with self.subTest(name=name):
                data = (self.directory / name).read_bytes()
                status, headers, body = self.request("/" + name)
                self.assertEqual((status, body), (200, data))
                self.assertEqual(int(headers["Content-Length"]), len(data))
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                status, head, body = self.request("/" + name, method="HEAD")
                self.assertEqual((status, body), (200, b""))
                self.assertEqual(head["Content-Length"], headers["Content-Length"])
        self.assertEqual(self.request("/")[2], (self.directory / "index.html").read_bytes())

    def test_video_single_ranges_closed_open_suffix_and_clamped_end(self):
        cases = (
            ("bytes=0-31", 0, 31),
            ("bytes=700-", 700, 1023),
            ("bytes=-17", 1007, 1023),
            ("bytes=1000-99999", 1000, 1023),
            ("bytes=-99999", 0, 1023),
            ("bytes=4-4", 4, 4),
        )
        for value, start, end in cases:
            with self.subTest(value=value):
                status, headers, body = self.request("/" + self.names[0], headers={"Range": value})
                self.assertEqual((status, body), (206, self.video[start : end + 1]))
                self.assertEqual(headers["Content-Range"], f"bytes {start}-{end}/1024")
                self.assertEqual(headers["Content-Length"], str(end - start + 1))
                self.assertEqual(headers["Accept-Ranges"], "bytes")
        status, headers, body = self.request(
            "/" + self.names[0], method="HEAD", headers={"Range": "bytes=4-9"}
        )
        self.assertEqual((status, body), (206, b""))
        self.assertEqual(headers["Content-Length"], "6")
        self.assertEqual(headers["Content-Range"], "bytes 4-9/1024")

    def test_invalid_or_unsatisfiable_ranges_return_empty_416(self):
        values = (
            "bytes=",
            "bytes=-0",
            "bytes=1024-",
            "bytes=9-4",
            "items=1-2",
            "bytes=0-1,3-4",
            "bytes=x-4",
            "bytes=" + "9" * 130 + "-",
        )
        for value in values:
            with self.subTest(value=value):
                status, headers, body = self.request("/" + self.names[0], headers={"Range": value})
                self.assertEqual((status, body), (416, b""))
                self.assertEqual(headers["Content-Range"], "bytes */1024")
                self.assertEqual(headers["Content-Length"], "0")
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=2
        )
        try:
            connection.putrequest("GET", "/" + self.names[0])
            connection.putheader("Range", "bytes=0-1")
            connection.putheader("Range", "bytes=2-3")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual((response.status, response.read()), (416, b""))
        finally:
            connection.close()

    def test_unknown_traversal_and_symlink_paths_do_not_serve(self):
        for route in (
            "/private.txt",
            "/missing.mp4",
            "/../private.txt",
            "/%2e%2e/private.txt",
            "/subdir/",
        ):
            self.assertEqual(self.request(route)[0], 404)
        victim = self.directory / self.names[0]
        victim.unlink()
        victim.symlink_to(self.directory / "private.txt")
        self.assertEqual(self.request("/" + self.names[0])[0], 404)

    def test_post_is_rejected_and_files_remain_unchanged(self):
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        self.assertEqual(self.request("/media.json", method="POST")[0], 501)
        self.assertEqual({p.name: p.read_bytes() for p in self.directory.iterdir()}, before)

    def test_pending_or_unsafe_manifests_cannot_start_a_server(self):
        path = self.directory / "media.json"
        manifest = json.loads(path.read_text())
        for change in (
            lambda m: m["robot_log_replay"].update(status="pending"),
            lambda m: m["robot_log_replay"].update(video="../private.txt"),
            lambda m: m.update(reconstructed_laundry_scene=None),
        ):
            bad = json.loads(json.dumps(manifest))
            change(bad)
            path.write_text(json.dumps(bad))
            with self.assertRaises((ValueError, TypeError)):
                PUBLIC["make_server"](self.directory, 0)


if __name__ == "__main__":
    unittest.main()
