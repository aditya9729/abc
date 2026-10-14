"""Thin command interface; native rendering is an explicit subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--research", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    laundry = commands.add_parser("prepare-laundry")
    laundry.add_argument("--original", type=Path, required=True)
    laundry.add_argument("--research", type=Path, required=True)
    laundry.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render")
    render.add_argument("--prepared", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--wall-seconds", type=float, default=540)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        from .prepare import prepare

        report = prepare(arguments.research, arguments.output)
        print(json.dumps({key: report[key] for key in ["claim", "frames", "fps", "links"]}))
    elif arguments.command == "prepare-laundry":
        from .laundry import prepare_laundry

        print(json.dumps(prepare_laundry(arguments.original, arguments.research, arguments.output)))
    else:
        from .render import render_recording

        print(
            json.dumps(
                render_recording(arguments.prepared, arguments.output, arguments.wall_seconds)
            )
        )


if __name__ == "__main__":
    main()
