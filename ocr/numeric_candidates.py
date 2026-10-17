"""Numeric identifiers extracted without inventing new digit sequences."""
from __future__ import annotations

import re


def numeric_identifiers(text: str | None) -> list[str]:
    """Return contiguous digit runs in one OCR line, preserving leading zeros.

    The function never joins separated runs, so ``"10 258"`` yields two
    evidence values rather than inventing ``10258``. The existing full-mode
    candidate parser remains responsible for its explicitly supported joined
    digit-group rule.
    """
    out = []
    for value in re.findall(r"\d+", text or ""):
        if value not in out:
            out.append(value)
    return out


def numeric_identifier(text: str | None) -> str | None:
    values = numeric_identifiers(text)
    return values[0] if len(values) == 1 else None
