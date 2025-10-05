from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from .constants import DEFAULT_PRESET_PATH, DEFAULT_TEMPLATE, PRESET_EXTENSIONS, PRESET_ROOT


@dataclass
class Preset:
    path: str
    name: str
    user_role: str
    assistant_role: str
    system_prompt: str
    jinja_template: str
    stop_sequences: List[str]
    bos_token: str


def _decode(value: str | None) -> str:
    if not value:
        return ""
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:
        return value


def ensure_preset_dir() -> None:
    os.makedirs(PRESET_ROOT, exist_ok=True)


def list_presets() -> List[str]:
    ensure_preset_dir()
    entries = [
        entry for entry in os.listdir(PRESET_ROOT)
        if any(entry.endswith(ext) for ext in PRESET_EXTENSIONS)
    ]
    entries.sort(key=str.lower)
    return entries


def load_preset(path: str | None = None) -> Preset:
    ensure_preset_dir()
    target = path or DEFAULT_PRESET_PATH
    try:
        with open(target, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return Preset(
            path=target,
            name=os.path.basename(target),
            user_role="user",
            assistant_role="assistant",
            system_prompt="",
            jinja_template=DEFAULT_TEMPLATE,
            stop_sequences=[],
            bos_token="",
        )

    template = _decode(data.get("jinja_template", DEFAULT_TEMPLATE)) or DEFAULT_TEMPLATE
    stop_sequences = [
        _decode(item)
        for item in data.get("stop_sequences")
        or data.get("antiprompt", [])
        or []
    ]

    return Preset(
        path=target,
        name=data.get("name") or os.path.basename(target),
        user_role=_decode(data.get("user_role", "user")) or "user",
        assistant_role=_decode(data.get("assistant_role", "assistant")) or "assistant",
        system_prompt=_decode(data.get("system_prompt", "")),
        jinja_template=template,
        stop_sequences=[item for item in stop_sequences if item],
        bos_token=_decode(data.get("bos_token", "")),
    )


def serialize_preset(preset: Preset) -> Dict[str, Any]:
    return {
        "name": preset.name,
        "user_role": preset.user_role,
        "assistant_role": preset.assistant_role,
        "system_prompt": preset.system_prompt,
        "jinja_template": preset.jinja_template,
        "bos_token": preset.bos_token,
        "stop_sequences": preset.stop_sequences,
    }


def save_preset(preset: Preset, path: str | None = None) -> str:
    ensure_preset_dir()
    target = path or preset.path or DEFAULT_PRESET_PATH
    data = serialize_preset(preset)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return target
