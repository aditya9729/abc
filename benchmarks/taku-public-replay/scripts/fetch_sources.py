"""Fetch only the pinned public assets needed by this experiment."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / "source_manifest.json"
    sources = json.loads(manifest.read_text())["sources"]
    args.output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        destination = args.output / source["name"]
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"Existing source differs from the pinned copy: {destination}")
            continue
        request = urllib.request.Request(
            source["url"], headers={"User-Agent": "TakuPublicReplay/0.2.1"}
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read(source["bytes"] + 1)
        if len(data) != source["bytes"] or hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError(
                f"Public source changed; review before updating the pin: {source['url']}"
            )
        with destination.open("xb") as output:
            output.write(data)
        print(destination)


if __name__ == "__main__":
    main()
