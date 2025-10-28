from __future__ import annotations

from typing import List, Tuple


def extract_speakable_segments(text: str, start_index: int) -> List[Tuple[str, int]]:
    segments: List[Tuple[str, int]] = []
    idx = start_index
    length = len(text)

    while idx < length:
        char = text[idx]
        if char in '.!?':
            end = idx + 1
            while end < length and (text[end].isspace() or text[end] in '.:;!?)*"'):
                end += 1
            segment = text[start_index:end].strip()
            if segment:
                segments.append((segment, end))
            start_index = end
            idx = end
            continue

        if char == '\n':
            segment = text[start_index:idx].strip()
            if segment:
                segments.append((segment, idx + 1))
            start_index = idx + 1
            idx = start_index
            continue

        idx += 1

    return segments


def leading_overlap(existing: str, new_text: str) -> int:
    max_overlap = min(len(existing), len(new_text))
    for size in range(max_overlap, 0, -1):
        if existing[-size:] == new_text[:size]:
            return size
    return 0
