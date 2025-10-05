from __future__ import annotations

from typing import Any, Dict, List

from jinja2 import Environment

from .constants import DEFAULT_TEMPLATE
from .preset import Preset


def build_jinja_environment() -> Environment:
    env = Environment(autoescape=False, extensions=['jinja2.ext.loopcontrols'])
    return env


def render_prompt(
    env: Environment,
    preset: Preset,
    messages: List[Dict[str, Any]],
    *,
    user_api_name: str,
    assistant_api_name: str,
    add_generation_prompt: bool = True,
) -> str:
    template_source = preset.jinja_template or DEFAULT_TEMPLATE
    template = env.from_string(template_source)

    render_messages = [
        {
            'role': entry.get('role', ''),
            'name': entry.get('name', ''),
            'content': entry.get('content', ''),
        }
        for entry in messages
    ]

    return template.render(
        messages=render_messages,
        user_role=preset.user_role,
        assistant_role=preset.assistant_role,
        user_api_name=user_api_name,
        assistant_api_name=assistant_api_name,
        system_prompt=preset.system_prompt,
        bos_token=preset.bos_token,
        add_generation_prompt=add_generation_prompt,
    )
