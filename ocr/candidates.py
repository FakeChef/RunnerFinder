"""
Turn one OCR text line into bib candidates, keeping every rejected piece
and the reason, so that all raw evidence stays available for review.

Examples:
  "NO. 10258"   -> "NO" (no_digit), "10258" (accepted)
  "UT-001"      -> "UT001" (accepted)
  "10 258"      -> "10258" (accepted, digit groups joined), "10" and "258"
                   kept as evidence with reason "part_of_joined"
  "XTERRA 2026" -> "XTERRA" (no_digit), "2026" (format-valid: only the
                   registered-bib list or a human can tell it is a year)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from config import BibFormatConfig
from ocr.bib_format import bib_rejection_reason, fix_digit_confusions, normalize_bib

PART_OF_JOINED = "part_of_joined"
NOT_REGISTERED = "not_registered"


@dataclass(frozen=True)
class Candidate:
    token: str               # piece of the raw line that was considered
    normalized: str          # normalize_bib(token)
    bib: str | None          # value after optional confusion fix (None if empty)
    corrected: bool          # True if fix_digit_confusions changed it
    format_reason: str | None   # None if the format is valid


def _make(token: str, cfg: BibFormatConfig, fix: bool) -> Candidate:
    normalized = normalize_bib(token)
    value = fix_digit_confusions(normalized, cfg.max_letters) if fix else normalized
    reason = bib_rejection_reason(value, cfg)
    if reason is not None and value != normalized:
        # The fix did not produce a valid bib: report the original reading.
        value, reason = normalized, bib_rejection_reason(normalized, cfg)
    return Candidate(token, normalized, value or None, value != normalized, reason)


def candidates_from_text(text: str, cfg: BibFormatConfig, fix_confusions: bool = True,
                         registered: set[str] | None = None) -> list[Candidate]:
    tokens = [t for t in (text or "").split() if t.strip()]
    if not tokens:
        return []
    out = [_make(t, cfg, fix_confusions) for t in tokens]

    # Digit groups printed or read with spaces ("10 258"): use the joined
    # value; the pieces stay as evidence but are not candidates themselves.
    if len(tokens) > 1 and all(normalize_bib(t).isdigit() for t in tokens):
        joined = _make("".join(tokens), cfg, fix_confusions)
        if joined.format_reason is None:
            out = [replace(c, format_reason=PART_OF_JOINED) for c in out] + [joined]

    if registered is not None:
        out = [replace(c, format_reason=NOT_REGISTERED)
               if c.format_reason is None and c.bib not in registered else c
               for c in out]
    return out


def load_registered_bibs(path, cfg: BibFormatConfig) -> set[str]:
    """One bib per line (or the first CSV column). Normalised, leading zeros kept."""
    bibs = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            value = normalize_bib(line.split(",")[0])
            if value:
                bibs.add(value)
    return bibs
