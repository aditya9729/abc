# Local scene viewer

This plain HTML page displays two complete recordings and the author's reference image.
It has no build step. The checked-in media manifest keeps both recordings pending.

Generate the recordings with the parent project's commands. Verify their complete frame counts and clean process exits.
Create a separate public directory outside Git. Copy `index.html` into that directory.
Copy the fetched `workflow-video-poster.jpg` as `source-reference.jpg`.
Copy each verified MP4 with its first 16 SHA-256 characters as the filename.

Use this manifest format for each recording:

```json
{
  "status": "accepted_complete",
  "video": "<first-16-sha256-characters>.mp4",
  "sha256": "<full-sha256>",
  "frames": 869,
  "fps": 12,
  "width": 960,
  "height": 640,
  "receipt": {"path": "<acceptance-file>", "bytes": 1, "sha256": "<receipt-sha256>"}
}
```

Replace every placeholder and the example byte count with actual evidence.
The robot-only recording has dimensions 640 by 480. The room recording has dimensions 960 by 640.
Keep the top-level schema and the two entry names from `media.json`.
The page validates display fields. It does not authenticate the receipt or accept a simulation run.

Serve only the public directory on localhost:

```sh
CUDA_VISIBLE_DEVICES='' python3 -I -B viewer/server.py \
  --port 8771 --directory /absolute/path/to/public
```

Run this command from the package root. The server binds only localhost and serves the five declared files.
It supports GET, HEAD and single byte ranges. Invalid video ranges return 416.
It has no directory listing or write route. Keep the public files unchanged while serving.
Use this server for seeking; Python's basic `http.server` did not support seeking in the browser check.

The native players support seeking and fullscreen. Refresh preserves unchanged players and their positions.
The scene uses recorded motion, estimated props, static towels and a scripted dryer door.

`test_viewer.cjs` contains six small synthetic DOM checks. Run it with `node --test viewer/test_viewer.cjs`.
Six HTTP tests use synthetic files and ephemeral localhost ports. Run them with:

```sh
CUDA_VISIBLE_DEVICES='' python3 -I -B -m unittest discover -s viewer -p test_server.py -v
```

These checks cover full responses, byte ranges, path restrictions and rejected writes.
`check_browser.py` checks actual video loading, seeking, playback, refresh and desktop/mobile layouts.
It uses the existing workspace Playwright runtime and a pinned Chromium executable path.
Its command-line help describes the required localhost URL, manifest and fresh output directory.
Hide CUDA when running it. Use a finite outer timeout.

See the main experiment report in `docs/bench/TAKU_LAUNDRY_RECONSTRUCTION_20261014.md` at the repository root.
The [viewer diagram](ARCHITECTURE.mmd) shows publication ownership.
