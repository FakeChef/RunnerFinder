import pytest

from config import BibFormatConfig, OCRConfig
from detection.geometry import Box
from ocr.bib_format import bib_rejection_reason, fix_digit_confusions
from ocr.bib_reader import parse_paddle_results, paddleocr_init_kwargs, paddleocr_predict_kwargs
from ocr.candidates import candidates_from_text

CFG = BibFormatConfig()


# ---------------------------------------------------------------- format reasons

@pytest.mark.parametrize("bib, reason", [
    ("", "empty"), ("12", "too_short"), ("123456789", "too_long"), ("NIKE", "no_digit"),
    ("RUNNING1", "too_many_letters"), ("T12$4", "invalid_characters"), ("10258", None),
    ("00123", None), ("UT001", None)])
def test_rejection_reasons(bib, reason):
    assert bib_rejection_reason(bib, CFG) == reason


def test_pattern_mismatch_reason():
    assert bib_rejection_reason("T1234", BibFormatConfig(patterns=[r"^\d{5}$"])) == "pattern_mismatch"


# ---------------------------------------------------------------- confusion fixes

@pytest.mark.parametrize("raw, fixed", [
    ("1O258", "10258"), ("UT0O1", "UT001"), ("0O123", "00123"), ("1025B", "10258"),
    ("T1234", "T1234"),            # prefix untouched
    ("TI234", "TI234"),            # could be a real 2-letter prefix: unchanged
    ("O1234", "O1234"),            # first character is a letter: not touched
    ("10258A", "10258A"),          # A is not a known confusion: unchanged
    ("NIKE", "NIKE"), ("", ""),
])
def test_fix_digit_confusions(raw, fixed):
    assert fix_digit_confusions(raw, 3) == fixed


# ---------------------------------------------------------------- candidates

def accepted(cands):
    return [c.bib for c in cands if c.format_reason is None]


def test_label_and_number():
    cands = candidates_from_text("No. 10258", CFG)
    assert accepted(cands) == ["10258"]
    assert {c.normalized: c.format_reason for c in cands}["NO"] == "too_short"
    assert {c.normalized: c.format_reason for c in candidates_from_text("FINISHER 10258", CFG)}[
        "FINISHER"] == "no_digit"


def test_alphanumeric_and_separators():
    assert accepted(candidates_from_text("UT-001", CFG)) == ["UT001"]
    # Known limitation: a letter prefix separated by a space is NOT joined
    # (joining would turn "No. 10258" into "NO10258"). "T 1234" gives 1234.
    assert accepted(candidates_from_text("t 1234", CFG)) == ["1234"]
    assert accepted(candidates_from_text("T1234", CFG)) == ["T1234"]


def test_split_digit_groups_are_joined():
    cands = candidates_from_text("10 258", CFG)
    assert accepted(cands) == ["10258"]
    assert sorted(c.format_reason for c in cands if c.format_reason) == ["part_of_joined"] * 2


def test_confusion_fix_is_flagged():
    (c,) = candidates_from_text("1O258", CFG)
    assert (c.bib, c.corrected, c.normalized, c.format_reason) == ("10258", True, "1O258", None)
    (c,) = candidates_from_text("1O258", CFG, fix_confusions=False)
    assert (c.bib, c.corrected, c.format_reason) == ("1O258", False, None)   # 1 letter: format-valid


def test_year_is_format_valid_but_registered_list_rejects_it():
    assert accepted(candidates_from_text("XTERRA 2026", CFG)) == ["2026"]
    cands = candidates_from_text("XTERRA 2026", CFG, registered={"10258"})
    assert accepted(cands) == []
    assert "not_registered" in [c.format_reason for c in cands]


def test_empty_text():
    assert candidates_from_text("", CFG) == [] and candidates_from_text("   ", CFG) == []


# ---------------------------------------------------------------- PaddleOCR adapter

def test_paddleocr_settings_default_to_verified_windows_cpu_setup():
    kw = paddleocr_init_kwargs(OCRConfig())
    assert kw["enable_mkldnn"] is False and kw["device"] == "cpu"
    assert kw["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert kw["text_recognition_model_name"] == "en_PP-OCRv5_mobile_rec"
    assert not kw["use_doc_orientation_classify"] and not kw["use_doc_unwarping"]
    assert "cpu_threads" not in kw
    assert paddleocr_init_kwargs(OCRConfig(enable_mkldnn=True, cpu_threads=4))["cpu_threads"] == 4
    assert paddleocr_predict_kwargs(OCRConfig()) == {"text_rec_score_thresh": 0.0}


def test_parse_paddle_results():
    res = [{"rec_texts": ["10258", "NIKE"], "rec_scores": [0.97, 0.5],
            "rec_boxes": [[10, 20, 110, 60], [5, 5, 50, 15]],
            "rec_polys": [[[10, 20], [110, 20], [110, 60], [10, 60]], [[5, 5], [50, 5], [50, 15], [5, 15]]]}]
    lines = parse_paddle_results(res)
    assert [(l.text, l.confidence) for l in lines] == [("10258", 0.97), ("NIKE", 0.5)]
    assert lines[0].box == Box(10, 20, 110, 60)


def test_parse_paddle_results_uses_polygons_when_boxes_missing():
    res = [{"rec_texts": ["UT001"], "rec_scores": [0.9], "rec_boxes": [],
            "rec_polys": [[[12, 30], [80, 28], [82, 50], [11, 52]]]}]
    assert parse_paddle_results(res)[0].box == Box(11, 28, 82, 52)
    assert parse_paddle_results([]) == [] and parse_paddle_results(None) == []
