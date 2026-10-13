"""Export the packaged original-methods viewer without changing result data."""

from __future__ import annotations

import argparse
import os
import tempfile
from importlib.resources import files
from pathlib import Path


def export_viewer(output: Path) -> Path:
    """Write only index.html to an explicit absolute directory, atomically."""
    if not output.is_absolute() or output.is_symlink():
        raise ValueError("--output-dir must be an absolute ordinary directory")
    if output.exists() and not output.is_dir():
        raise ValueError("--output-dir must be a directory")
    index = output / "index.html"
    if index.is_symlink() or (index.exists() and not index.is_file()):
        raise ValueError("Existing index.html must be an ordinary file")
    template = (
        files("abc_bench").joinpath("static/original-methods/index.html").read_bytes()
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".index-", suffix=".tmp", dir=output, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(template)
        os.replace(temporary, index)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        destination = export_viewer(arguments.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(destination)


if __name__ == "__main__":
    main()
