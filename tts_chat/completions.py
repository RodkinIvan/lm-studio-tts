from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import requests

from .exceptions import ChatClientError


def stream_completions(
    prompt: str,
    *,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    stop_sequences: Optional[List[str]] = None,
    stop_event: Optional["threading.Event"] = None,
) -> Iterator[str]:
    from threading import Event  # local import to avoid circular typing

    if stop_event is not None and not isinstance(stop_event, Event):
        raise TypeError("stop_event must be a threading.Event or None")

    url = base_url.rstrip('/') + '/v1/completions'
    payload: Dict[str, Any] = {
        'model': model,
        'prompt': prompt,
        'temperature': temperature,
        'stream': True,
    }
    if max_tokens > 0:
        payload['max_tokens'] = max_tokens
    if stop_sequences:
        payload['stop'] = stop_sequences

    response = requests.post(url, json=payload, timeout=timeout, stream=True)
    response.raise_for_status()
    response.encoding = 'utf-8'

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if stop_event and stop_event.is_set():
                break
            if not raw_line or not raw_line.startswith('data: '):
                continue

            payload_line = raw_line[len('data: '):].strip()
            if payload_line == '[DONE]':
                break

            try:
                chunk = json.loads(payload_line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
                raise ChatClientError(f'Failed to parse stream chunk: {payload_line!r}') from exc

            choice = chunk.get('choices', [{}])[0]
            text = choice.get('text')
            if text:
                yield text
                continue

            delta = choice.get('delta', {})
            text = delta.get('text') or delta.get('content')
            if text:
                yield text
    finally:
        response.close()
