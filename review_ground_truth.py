"""Pure helpers for manual ground-truth review."""
from __future__ import annotations

import re


def parse_human_bibs(value: str | None) -> list[str]:
    """Split human-entered bibs while preserving each value as text.

    Supported separators are English/Chinese commas, Chinese enumeration
    commas, English/Chinese semicolons, and newlines. Leading zeros are never
    converted to integers.
    """
    if not value:
        return []
    return [part.strip() for part in re.split(r"[,，、;；\r\n]+", value) if part.strip()]
