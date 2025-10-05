from __future__ import annotations

import os

APP_NAME = "LM Studio Voice Chat"
DEFAULT_WINDOW_GEOMETRY = "1200x720"
MIN_WINDOW_WIDTH = 900
MIN_WINDOW_HEIGHT = 600

PRESET_ROOT = os.path.expanduser("~/.cache/lm-studio-tts/config-presets")
DEFAULT_PRESET_PATH = os.path.join(PRESET_ROOT, "llama-3-psychologist.json")
PRESET_EXTENSIONS = (".preset.json", ".json")

SAMPLE_RATE = 24_000

THEME = {
    "bg": "#0e1621",
    "user_bg": "#1f5b9c",
    "user_fg": "#ffffff",
    "assistant_bg": "#1f2c3b",
    "assistant_fg": "#e4ecfa",
    "system_fg": "#7f9ab5",
    "status_fg": "#d1d8e0",
    "input_bg": "#17212b",
    "input_fg": "#e4ecfa",
    "button_bg": "#f1f5fa",
    "button_fg": "#11161f",
    "button_active_bg": "#d8e4f2",
    "button_active_fg": "#11161f",
    "panel_bg": "#151f2a",
    "panel_heading_fg": "#e4ecfa",
    "panel_label_fg": "#b3c1d1",
    "entry_bg": "#0f1a24",
    "entry_fg": "#e4ecfa",
}

DEFAULT_TEMPLATE = (
    "{{- bos_token if bos_token is defined else '' -}}\n"
    "{%- for message in messages %}\n"
    "  {{ '<|start_header_id|>' + (message.name or message.role) + '<|end_header_id|>\\n\\n' }}\n"
    "  {{ message.content | trim + '<|eot_id|>' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "  {{ '<|start_header_id|>' + assistant_api_name + '<|end_header_id|>\\n\\n' }}\n"
    "{%- endif %}"
)
