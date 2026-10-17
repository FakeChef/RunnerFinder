import pytest

from config import BibFormatConfig
from ocr.bib_format import clean_bib, is_valid_bib, normalize_bib

CFG = BibFormatConfig()  # defaults: 3-8 chars, at least one digit, <=3 letters


@pytest.mark.parametrize("raw, expected", [
    ("10258", "10258"), ("t1234", "T1234"), ("UT-001", "UT001"),
    (" ut 001 ", "UT001"), ("#10258", "10258"), ("", ""), (None, ""),
])
def test_normalize(raw, expected):
    assert normalize_bib(raw) == expected


@pytest.mark.parametrize("bib", ["001", "00123", "UT001", "T0001"])
def test_leading_zeros_preserved(bib):
    assert clean_bib(bib, CFG) == bib


@pytest.mark.parametrize("bib", ["10258", "T1234", "UT001", "123", "AB123456"])
def test_valid_bibs(bib):
    assert is_valid_bib(bib, CFG)


@pytest.mark.parametrize("bib", ["12", "123456789", "NIKE", "RUNNING1", "T12$4"])
def test_invalid_bibs(bib):
    assert not is_valid_bib(bib, CFG)


def test_length_is_configurable():
    cfg = BibFormatConfig(min_length=5, max_length=5)
    assert is_valid_bib("10258", cfg) and not is_valid_bib("1025", cfg)


def test_race_specific_patterns():
    cfg = BibFormatConfig(patterns=[r"^\d{5}$", r"^UT\d{3}$"])
    assert clean_bib("ut-001", cfg) == "UT001"
    assert clean_bib("T1234", cfg) is None
