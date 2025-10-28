# LM Studio Voice Chat

Interactive chat UI that connects to an LM Studio-served model and plays responses through Kokoro text-to-speech. The refactored client lives entirely inside this directory.

## Prerequisites
- [LM Studio](https://lmstudio.ai/) installed with a model downloaded and ready to serve.
- Python 3.10+ (Tkinter is required; it ships with the standard CPython installer on macOS/Windows. On many Linux distros you may need to install `python3-tk` separately.)

## Installation
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Optional audio backends (`sounddevice`, `simpleaudio`) are included; the app falls back to an OS player if neither is available.

## Running the app
1. Start LM Studio and host your model through “Local Server”. Note the base URL (defaults to `http://127.0.0.1:1234`) and the model identifier exposed by the server (often `lmstudio`).
2. Launch the chat UI from this directory:
   ```bash
   python main.py --base-url http://127.0.0.1:1234
   ```
   Command-line options:
   - `--system`, `--user-role`, `--assistant-role` for prompt customization.
   - `--voice` and `--speed` to adjust Kokoro playback.
   - `--temperature` (default `0.0`) and `--seed` (default `42`) for deterministic sampling.
   - `--text-only` to disable audio output while keeping the chat UI.

The window provides:
- Streaming assistant replies with in-flight TTS playback.
- Right-click context menu to edit, delete, or continue messages.
- Preset panel to manage LM Studio prompt templates. Presets are saved under `~/.cache/lm-studio-tts/config-presets/`.

## Troubleshooting
- **No audio**: ensure the optional backends are installed or let the app fall back to system playback. You can also launch with `--text-only`.
- **Tkinter missing**: install OS packages (`sudo apt install python3-tk` on Debian/Ubuntu).
- **Requests failing**: confirm the LM Studio server is running and the base URL/model name match the ones shown in LM Studio.

