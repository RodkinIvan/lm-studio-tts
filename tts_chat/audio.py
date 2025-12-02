from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None

try:
    import simpleaudio as sa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sa = None

from .constants import SAMPLE_RATE
from .kokoro_local import LocalKPipeline, ensure_local_kokoro_repo


class AudioPlayer:
    """Handle text-to-speech playback using Kokoro and multiple backends."""

    def __init__(self, lang_code: str = "a") -> None:
        model_dir = ensure_local_kokoro_repo()
        self._pipeline = LocalKPipeline(lang_code=lang_code, model_dir=model_dir)

    @staticmethod
    def _play_via_tempfile(audio: np.ndarray) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            file_path = Path(tmp_file.name)
            sf.write(file_path, audio, SAMPLE_RATE)

        try:
            if sys.platform == "darwin":
                cmd = ["afplay", str(file_path)]
            elif sys.platform.startswith("linux"):
                cmd = ["aplay", "-q", str(file_path)]
            elif sys.platform.startswith("win"):
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();",
                ]
            else:
                raise RuntimeError("Unsupported platform for audio playback fallback.")

            subprocess.run(cmd, check=True)
        finally:
            try:
                os.remove(file_path)
            except OSError:
                pass

    @staticmethod
    def _play_buffer(audio: np.ndarray) -> None:
        if sd is not None:
            sd.play(audio, SAMPLE_RATE)
            sd.wait()
            return

        if sa is not None:
            audio_clipped = np.clip(audio, -1.0, 1.0)
            channels = 1 if audio_clipped.ndim == 1 else audio_clipped.shape[1]
            int_data = np.int16(audio_clipped * 32767)
            play_obj = sa.play_buffer(int_data, channels, 2, SAMPLE_RATE)
            play_obj.wait_done()
            return

        AudioPlayer._play_via_tempfile(audio)

    def speak_text(self, text: str, voice: str, speed: float) -> None:
        if not text.strip():
            return

        generator = self._pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        for _, _, audio in generator:
            audio_array = np.asarray(audio, dtype=np.float32)
            self._play_buffer(audio_array)

    @staticmethod
    def stop() -> None:
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
