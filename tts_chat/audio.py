from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import queue
from pathlib import Path
from typing import ClassVar, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None

try:
    import simpleaudio as sa  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sa = None

from .constants import SAMPLE_RATE
from .kokoro_local import DEFAULT_KOKORO_REPO, LocalKPipeline, ensure_local_kokoro_repo


class AudioPlayer:
    """Handle text-to-speech playback using Kokoro and multiple backends."""

    _last_instance: ClassVar[Optional["AudioPlayer"]] = None
    _threads_configured: ClassVar[bool] = False

    def __init__(self, lang_code: str = "a") -> None:
        self._configure_threads()

        model_dir = ensure_local_kokoro_repo(repo_id=DEFAULT_KOKORO_REPO)
        self._pipeline = LocalKPipeline(lang_code=lang_code, model_dir=model_dir, repo_id=DEFAULT_KOKORO_REPO)

        self._text_queue: "queue.Queue[Tuple[str, str, float]]" = queue.Queue()
        self._play_queue: "queue.Queue[object]" = queue.Queue(maxsize=4)
        self._stop_flag = threading.Event()
        self._playback_sentinel = object()
        self._first_error: Optional[Exception] = None
        self._error_lock = threading.Lock()

        self._generator_thread = threading.Thread(target=self._generator_loop, daemon=True)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._generator_thread.start()
        self._playback_thread.start()
        AudioPlayer._last_instance = self

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
        self._raise_if_error()
        if not text.strip():
            return

        try:
            self._text_queue.put_nowait((text, voice, speed))
        except queue.Full:
            raise RuntimeError("Audio queue is full; cannot enqueue speech right now.")

    def _generator_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                item = self._text_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text, voice, speed = item
            try:
                generator = self._pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
                buffers = []
                for _, _, audio in generator:
                    buffers.append(np.asarray(audio, dtype=np.float32))
                if buffers:
                    merged = np.concatenate(buffers, axis=0)
                    self._put_play_item(merged)
            except Exception as exc:
                self._record_error(exc)

    def _put_play_item(self, audio_array: np.ndarray) -> None:
        while not self._stop_flag.is_set():
            try:
                self._play_queue.put(audio_array, timeout=0.1)
                return
            except queue.Full:
                continue

    def _playback_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                item = self._play_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is self._playback_sentinel:
                break

            try:
                self._play_buffer(item)  # type: ignore[arg-type]
            except Exception as exc:
                self._record_error(exc)

    def _record_error(self, exc: Exception) -> None:
        with self._error_lock:
            if self._first_error is None:
                self._first_error = exc

    def _raise_if_error(self) -> None:
        with self._error_lock:
            if self._first_error is not None:
                raise self._first_error

    def _flush_queues(self) -> None:
        while True:
            try:
                self._text_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._play_queue.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def stop() -> None:
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        inst = AudioPlayer._last_instance
        if inst is not None:
            inst._flush_queues()
            inst._stop_flag.set()
            try:
                inst._play_queue.put_nowait(inst._playback_sentinel)
            except Exception:
                pass

    @classmethod
    def _configure_threads(cls) -> None:
        if cls._threads_configured:
            return
        threads_env = os.getenv("KOKORO_NUM_THREADS")
        threads: Optional[int] = None
        if threads_env:
            try:
                parsed = int(threads_env)
                if parsed > 0:
                    threads = parsed
            except ValueError:
                pass
        if threads is None:
            cpu = os.cpu_count() or 2
            threads = max(1, min(cpu - 1 if cpu > 1 else 1, 4))
        try:
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(max(1, min(threads, 2)))
        except Exception:
            pass
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(threads)
        cls._threads_configured = True
