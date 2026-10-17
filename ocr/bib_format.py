"""
Bib number normalisation and validation.

Used for OCR output, user search input and manual corrections, so that a
search for "ut-001" finds a photo where OCR read "UT001". Bibs are always
TEXT: leading zeros are kept everywhere.
"""
from __future__ import annotations

import re

from config import BibFormatConfig

# Separators that appear around/inside bib numbers but are not part of them.
_SEPARATORS = re.compile(r"[\s\-_.#·:/]+")

# Rejection reason codes (stable: they appear in CSV reports).
EMPTY = "empty"
TOO_SHORT = "too_short"
TOO_LONG = "too_long"
INVALID_CHARACTERS = "invalid_characters"
NO_DIGIT = "no_digit"
TOO_MANY_LETTERS = "too_many_letters"
PATTERN_MISMATCH = "pattern_mismatch"


def normalize_bib(text: str | None) -> str:
    """Upper-case, separators removed. Leading zeros are kept."""
    if not text:
        return ""
    return _SEPARATORS.sub("", text.strip().upper())


def bib_rejection_reason(bib: str, cfg: BibFormatConfig) -> str | None:
    """Why a normalised string cannot be a bib, or None if it can."""
    if not bib:
        return EMPTY
    if any(ch not in cfg.allowed_chars for ch in bib):
        return INVALID_CHARACTERS
    if len(bib) < cfg.min_length:
        return TOO_SHORT
    if len(bib) > cfg.max_length:
        return TOO_LONG
    if cfg.require_digit and not any(ch.isdigit() for ch in bib):
        return NO_DIGIT
    if sum(ch.isalpha() for ch in bib) > cfg.max_letters:
        return TOO_MANY_LETTERS
    if cfg.patterns and not any(re.fullmatch(p, bib) for p in cfg.patterns):
        return PATTERN_MISMATCH
    return None


def is_valid_bib(bib: str, cfg: BibFormatConfig) -> bool:
    return bib_rejection_reason(bib, cfg) is None


def clean_bib(text: str | None, cfg: BibFormatConfig) -> str | None:
    """Normalise + validate in one step. Returns None if not a valid bib."""
    bib = normalize_bib(text)
    return bib if is_valid_bib(bib, cfg) else None


# OCR often confuses these letters with digits.
_LETTER_AS_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                    "Z": "2", "S": "5", "B": "8", "G": "6"}


def fix_digit_confusions(bib: str, max_prefix_letters: int) -> str:
    """
    Conservative fix for letter/digit confusion in the NUMERIC part.

    A bib is read as an optional letter prefix followed by the number. Only
    characters after the first digit are changed, and only if the result is
    all digits: "1O258" -> "10258", "UT0O1" -> "UT001". Letter prefixes are
    never changed ("T1234" stays; "TI234" stays, because it might be a real
    prefix). Returns the input unchanged when no safe fix applies.
    """
    m = re.match(r"^([A-Z]*)(\d.*)$", bib)
    if not m:
        return bib
    prefix, rest = m.groups()
    if len(prefix) > max_prefix_letters:
        return bib
    fixed = "".join(_LETTER_AS_DIGIT.get(ch, ch) for ch in rest)
    return prefix + fixed if fixed != rest and fixed.isdigit() else bib
