from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "presets"
DEFAULT_TARGET = Path.home() / ".cache" / "lm-studio-tts" / "config-presets"


def copy_presets(source: Path, target: Path, overwrite: bool) -> None:
    if not source.exists():
        print(f"Source directory not found: {source}", file=sys.stderr)
        sys.exit(1)

    target.mkdir(parents=True, exist_ok=True)

    copied = 0
    for entry in source.iterdir():
        if not entry.is_file():
            continue
        destination = target / entry.name
        if destination.exists() and not overwrite:
            print(f"Skipping existing preset: {destination}")
            continue
        shutil.copy2(entry, destination)
        print(f"Copied {entry.name} -> {destination}")
        copied += 1

    if copied == 0:
        print("No presets copied.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy bundled presets into the LM Studio preset cache.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Folder containing preset json files (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Destination directory (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing presets in the target directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_presets(args.source, args.target, args.overwrite)


if __name__ == "__main__":
    main()
