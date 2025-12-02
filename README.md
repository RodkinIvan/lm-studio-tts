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
1. Start LM Studio and expose your chosen model via the **Local Server** tab. Leave it listening on the default `http://127.0.0.1:1234` unless you have a reason to change it.
2. From this folder, launch the chat UI:
   ```bash
   python main.py
   ```
   The defaults assume LM Studio serves the model under the name `lmstudio` at `http://127.0.0.1:1234`. Adjust via CLI flags if needed:
   - `--model lmstudio` and/or `--base-url http://127.0.0.1:1234`
   - `--system`, `--user-role`, `--assistant-role` to tweak the prompt
   - `--voice` / `--speed` for Kokoro playback
   - `--temperature` (default `0.0`) and `--seed` (default `42`) for deterministic sampling
   - `--text-only` to disable audio while keeping the chat window

Once running you get:
- Streaming assistant responses, with real-time TTS if audio is enabled
- Right-click actions to edit, delete, or continue any message
- A preset panel that loads/saves templates under `~/.cache/lm-studio-tts/config-presets/`

### Download Kokoro locally (run once with internet)
The Kokoro TTS weights now live in this repo so the app can run offline.
Pull them down once with:
```bash
python scripts/download_kokoro.py
```
Files are stored under `tts_chat/models/kokoro-82m` by default. Override with `KOKORO_MODEL_DIR=/path/to/kokoro`. To force offline mode (fail instead of attempting a download), set `KOKORO_ALLOW_DOWNLOAD=0`.

### Optional: copy bundled presets
The repo ships with sample presets (`presets/llama-3-psychologist.preset.json`, etc.). Copy them into your preset cache with:
```bash
python scripts/copy_presets.py            # copies to ~/.cache/lm-studio-tts/config-presets
# or customise:
python scripts/copy_presets.py --target /custom/path --overwrite
```
This step is optional; the chat UI works fine without importing these presets.

## Troubleshooting
- **No audio**: ensure the optional backends are installed or let the app fall back to system playback. You can also launch with `--text-only`.
- **Tkinter missing**: install OS packages (`sudo apt install python3-tk` on Debian/Ubuntu).
- **Requests failing**: confirm the LM Studio server is running and the base URL/model name match the ones shown in LM Studio.
