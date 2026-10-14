"""Local read-only Taku playback server with single-byte-range video responses.

Range selection follows the established serve_viewer_with_collection.py helper.
Only the five published files are exposed; no listings, proxies or writes.
"""

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

MEDIA_KEYS = ("robot_log_replay", "reconstructed_laundry_scene")


def file_path(directory: Path, name: str) -> Path:
    path = directory / name
    if path.resolve() != path or not path.is_file():
        raise ValueError("Published file must be ordinary: " + name)
    return path


def allowed_files(directory: Path) -> dict[str, str]:
    if not directory.is_absolute() or directory.resolve() != directory or not directory.is_dir():
        raise ValueError("--directory must be an absolute ordinary directory")
    manifest = json.loads(file_path(directory, "media.json").read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != "taku.local-scene-viewer/v1":
        raise ValueError("Wrong Taku media schema")
    files = {
        "index.html": "text/html; charset=utf-8",
        "media.json": "application/json",
        "source-reference.jpg": "image/jpeg",
    }
    for key in MEDIA_KEYS:
        entry = manifest.get(key)
        if not isinstance(entry, dict):
            raise TypeError("Missing published media entry: " + key)
        digest = entry.get("sha256")
        if (
            entry.get("status") != "accepted_complete"
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or entry.get("video") != digest[:16] + ".mp4"
        ):
            raise ValueError("Both published videos require accepted local hash names")
        files[entry["video"]] = "video/mp4"
    if len(files) != 5:
        raise ValueError("Two distinct published videos are required")
    for name in files:
        file_path(directory, name)
    return files


def byte_range(value: str, size: int) -> tuple[int, int]:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value) if len(value) < 128 else None
    if match is None or not any(match.groups()):
        raise ValueError("Invalid byte range")
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    else:
        suffix = int(last)
        if suffix < 1:
            raise ValueError("Empty suffix range")
        start, end = max(0, size - suffix), size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable range")
    return start, end


def make_server(directory: Path, port: int) -> ThreadingHTTPServer:
    files = allowed_files(directory)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.serve_file(body=True)

        def do_HEAD(self) -> None:
            self.serve_file(body=False)

        def serve_file(self, body: bool) -> None:
            route = urlsplit(self.path).path
            name = "index.html" if route == "/" else route.removeprefix("/")
            if name not in files:
                self.send_error(404)
                return
            try:
                path = file_path(directory, name)
                stream = path.open("rb")
            except (OSError, ValueError):
                self.send_error(404)
                return
            with stream:
                size = os.fstat(stream.fileno()).st_size
                start, end, code = 0, size - 1, 200
                if files[name] == "video/mp4" and "Range" in self.headers:
                    try:
                        if len(self.headers.get_all("Range")) != 1:
                            raise ValueError("Multiple byte ranges")
                        start, end = byte_range(self.headers["Range"], size)
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    code = 206
                length = end - start + 1
                self.send_response(code)
                self.send_header("Content-Type", files[name])
                self.send_header("Content-Length", str(length))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if files[name] == "video/mp4":
                    self.send_header("Accept-Ranges", "bytes")
                if code == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if not body:
                    return
                stream.seek(start)
                try:
                    remaining = length
                    while remaining:
                        data = stream.read(min(65536, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *_args: object) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        server = make_server(args.directory, args.port)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": "listening",
                "host": "127.0.0.1",
                "port": server.server_address[1],
                "directory": str(args.directory),
                "files": sorted(allowed_files(args.directory)),
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
