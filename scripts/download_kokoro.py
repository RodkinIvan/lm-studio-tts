from __future__ import annotations

import argparse
from pathlib import Path

from tts_chat.kokoro_local import DEFAULT_KOKORO_REPO, ensure_local_kokoro_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kokoro TTS weights into this repository.")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_KOKORO_REPO,
        help="Hugging Face repo id for Kokoro (default: hexgrad/Kokoro-82M).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Directory to store the model. Defaults to tts_chat/models/kokoro-82m.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require files to already exist; fail instead of hitting the network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = ensure_local_kokoro_repo(
        repo_id=args.repo_id,
        target_dir=args.target,
        allow_download=not args.offline,
    )
    print(f"Kokoro model is ready at {model_dir}")


if __name__ == "__main__":
    main()
