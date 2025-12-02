from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from kokoro import KModel, KPipeline

DEFAULT_KOKORO_REPO = "hexgrad/Kokoro-82M"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "kokoro-82m"
ENV_MODEL_DIR = "KOKORO_MODEL_DIR"
ENV_ALLOW_DOWNLOAD = "KOKORO_ALLOW_DOWNLOAD"

REQUIRED_FILES = ("config.json", "kokoro-v1_0.pth")


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False", "no", "off"}


def _has_required_files(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    required = all((model_dir / fname).exists() for fname in REQUIRED_FILES)
    voices_dir = model_dir / "voices"
    return required and voices_dir.is_dir() and any(voices_dir.glob("*.pt"))


def ensure_local_kokoro_repo(
    *,
    repo_id: str = DEFAULT_KOKORO_REPO,
    target_dir: Optional[Path | str] = None,
    allow_download: Optional[bool] = None,
) -> Path:
    """
    Make sure the Kokoro model lives on disk and return its directory.

    By default this stores files under `tts_chat/models/kokoro-82m`.
    You can override the target via KOKORO_MODEL_DIR or the `target_dir` argument.
    Set KOKORO_ALLOW_DOWNLOAD=0 to forbid network calls and require a pre-populated folder.
    """
    resolved_dir = Path(target_dir or os.getenv(ENV_MODEL_DIR, DEFAULT_MODEL_DIR)).expanduser().resolve()
    download_ok = _truthy_env(ENV_ALLOW_DOWNLOAD, True) if allow_download is None else allow_download

    if _has_required_files(resolved_dir):
        return resolved_dir

    if not download_ok:
        raise RuntimeError(
            f"Kokoro model files not found at {resolved_dir}. "
            f"Provide the files manually or allow downloads via {ENV_ALLOW_DOWNLOAD}=1."
        )

    resolved_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=resolved_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Kokoro weights are not available locally and downloads are disabled or offline. "
            "Connect once to the internet or copy the Hugging Face repo into the model directory."
        ) from exc

    if not _has_required_files(resolved_dir):
        raise RuntimeError(
            f"Downloaded Kokoro repo to {resolved_dir}, but required files are missing. "
            "Ensure config.json, kokoro-v1_0.pth, and voices/*.pt are present."
        )

    return resolved_dir


def kokoro_paths(model_dir: Path) -> Tuple[Path, Path]:
    """Return (config_path, weights_path) for a Kokoro repo."""
    model_dir = model_dir.expanduser()
    return model_dir / "config.json", model_dir / "kokoro-v1_0.pth"


class LocalKPipeline(KPipeline):
    """Pipeline that only reads Kokoro assets from a local snapshot."""

    def __init__(
        self,
        *,
        lang_code: str,
        model_dir: Path,
        allow_remote_voices: bool = False,
        trf: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.repo_id = DEFAULT_KOKORO_REPO
        self.allow_remote_voices = allow_remote_voices

        config_path, weights_path = kokoro_paths(self.model_dir)
        model = KModel(config=str(config_path), model=str(weights_path))
        super().__init__(lang_code=lang_code, model=model, trf=trf, device=device)

    def _voice_path(self, voice: str) -> Path:
        filename = voice if voice.endswith(".pt") else f"{voice}.pt"
        return self.model_dir / "voices" / filename

    def _maybe_fetch_voice(self, voice: str) -> None:
        if not self.allow_remote_voices:
            return
        filename = voice if voice.endswith(".pt") else f"{voice}.pt"
        snapshot_download(
            repo_id=self.repo_id,
            local_dir=self.model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=[f"voices/{filename}"],
        )

    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]

        voice_path = self._voice_path(voice)
        if not voice_path.exists():
            self._maybe_fetch_voice(voice)

        if not voice_path.exists():
            raise FileNotFoundError(
                f"Voice '{voice}' not found at {voice_path}. "
                "Download the Kokoro voices (see README) or set allow_remote_voices=True."
            )

        pack = torch.load(voice_path, weights_only=True)
        self.voices[voice] = pack
        return pack
