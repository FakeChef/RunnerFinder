"""Bib candidate filter (V0.3.1, rule set 2 in V1.3.1): scoring rules, statuses and YAML configuration."""
from pathlib import Path

import pytest

from ocr.bib_filter import DEFAULT_RULES, BibCandidateFilter

RULES_FILE = Path(__file__).resolve().parents[2] / "config" / "bib_rules.yaml"


@pytest.fixture
def f():
    return BibCandidateFilter.load(RULES_FILE)


@pytest.mark.parametrize("bib, numeric, conf, raw", [
    ("60914", "60914", 0.95, "60914"),
    ("B60914", "60914", 0.70, "B 60914"),
    ("68564", "68564", 0.88, "68564"),
    ("A12466", "12466", 0.60, "A12466"),
    ("12026", "12026", 0.80, "12026"),      # a real bib that contains "2026" is still accepted
    ("A1076", "1076", 0.97, "A1076"),       # V1.3.1: 4 digits with a prefix (Wujiang)
    ("0175", "0175", 0.97, "0175"),         # 4 digits, leading zero kept
    ("UT1234", "1234", 0.95, "UT1234"),     # two-letter prefix
    ("00128", "00128", 0.95, "00128"),
    ("A2026", "2026", 0.95, "A2026"),       # a year WITH a letter prefix is a bib form
])
def test_real_bibs_are_accepted(f, bib, numeric, conf, raw):
    r = f.evaluate(bib, numeric, conf, [raw], "person_body")
    assert r.accepted, r


@pytest.mark.parametrize("bib, numeric, conf, raw, reason", [
    ("10KM", "10", 0.95, "10KM", "excluded_text"),
    ("1365", "1365", 0.97, "13-65", "likely_storage_number"),
    ("13655", "13655", 0.97, "13-655", "likely_storage_number"),       # 5 digits, still hidden
    ("71185178", "71185178", 0.82, "7-1185178", "likely_storage_number"),
    ("2026", "2026", 0.99, "2026 PARENT-CHILD", "year"),       # V1.3.1: a year inside other text
    ("12", "12", 0.99, "12", "too_few_digits"),
    ("210", "210", 0.95, "210K", "distance"),
])
def test_noise_is_rejected_with_a_reason(f, bib, numeric, conf, raw, reason):
    r = f.evaluate(bib, numeric, conf, [raw], "person_body")
    assert not r.accepted and reason in r.reasons, r


def test_storage_number_is_a_score_not_a_permanent_reject():
    lenient = BibCandidateFilter({**DEFAULT_RULES, "accept_threshold": -100})
    r = lenient.evaluate("13655", "13655", 0.97, ["13-655"], "person_body")
    assert r.accepted and "likely_storage_number" in r.reasons and r.score == 5 + 1 + 2 - 8


@pytest.mark.parametrize("bib, numeric, conf, raw, reason", [
    ("2026", "2026", 1.0, "2026", "possible_year"),     # V1.3.1: a lone year may be a real bib
    ("123", "123", 0.99, "123", "short_number"),        # 3 digits: searchable, never confirmed alone
    ("128", "128", 0.99, "128", "short_number"),
    ("7912", "7912", 0.95, "quohua 7912", "text_context"),
    ("4219605", "4219605", 0.9, "4219605", "long_number"),
])
def test_uncertain_candidates_stay_searchable(f, bib, numeric, conf, raw, reason):
    r = f.evaluate(bib, numeric, conf, [raw], "person_body")
    assert not r.accepted and r.status == "uncertain" and reason in r.reasons, r


def test_four_digits_are_not_rejected_and_five_are_not_preferred(f):
    four = f.evaluate("A1076", "1076", 0.97, ["A1076"], "person_body")
    five = f.evaluate("A10760", "10760", 0.97, ["A10760"], "person_body")
    assert four.accepted and five.accepted and four.score == five.score


def test_status_is_stored_and_old_results_are_read_compatibly():
    from ocr.bib_filter import status_of
    assert BibCandidateFilter().evaluate("A1076", "1076", 0.97, ["A1076"]).as_dict()["status"] == "accepted"
    assert status_of(None) == "accepted"                                   # V0.2 rows
    assert status_of({"accepted": False, "reasons": ["low_score"]}) == "uncertain"      # rule set 1
    assert status_of({"accepted": False, "reasons": ["year"]}) == "noise"
    assert status_of({"accepted": True, "reasons": []}) == "accepted"


def test_repeated_number_outside_person_crops(f):
    args = ("60914", "60914", 0.95, ["60914"])
    assert f.evaluate(*args, "full_image", non_person_repeats=2).score == 8
    r = f.evaluate(*args, "full_image", non_person_repeats=3)
    assert r.score == 5 and "repeated_outside_person" in r.reasons
    assert f.evaluate(*args, "person_body", non_person_repeats=9).score == 8


def test_rules_come_from_yaml_and_threshold_is_configurable(tmp_path):
    y = tmp_path / "rules.yaml"
    y.write_text("short_length_score: 5\naccept_threshold: 2\n", encoding="utf-8")
    custom = BibCandidateFilter.load(y)
    assert custom.evaluate("920", "920", 1.0, ["920"], "person_body").accepted
    assert custom.rules_hash != BibCandidateFilter().rules_hash
    y.write_text("acept_threshold: 2\n", encoding="utf-8")              # typo is not ignored
    with pytest.raises(ValueError, match="unknown keys"):
        BibCandidateFilter.load(y)


def test_shipped_rules_file_equals_built_in_defaults(f):
    assert f.rules_hash == BibCandidateFilter().rules_hash
    assert BibCandidateFilter.load(Path("does-not-exist.yaml")).rules_hash == f.rules_hash


def test_config_folder_does_not_shadow_config_module():
    import config
    assert hasattr(config, "get_config") and config.get_config().paths.bib_rules_file == RULES_FILE


def test_small_secondary_text_in_the_same_person_crop_is_uncertain():
    """V1.3.1: the bib number is the largest text on a bib (bag-deposit zone, sponsor and title lines are not)."""
    from ocr.bib_filter import PhotoCandidate, evaluate_photo
    f = BibCandidateFilter()
    big = PhotoCandidate("A0175", "0175", 0.97, ("A0175",), "person_body", 120.0, ("C1",))
    small = PhotoCandidate("299066", "299066", 0.95, ("299066",), "person_body", 25.0, ("C1",))
    other_runner = PhotoCandidate("A0222", "0222", 0.95, ("A0222",), "person_body", 25.0, ("C2",))
    big_noise = PhotoCandidate("10KM", "10", 0.99, ("10KM",), "person_body", 300.0, ("C2",))
    r = evaluate_photo(f, [big, small, other_runner, big_noise])
    assert r[0].status == "accepted" and "small_text" not in r[0].reasons
    assert r[1].status == "uncertain" and "small_text" in r[1].reasons          # searchable, not hidden
    assert r[2].status == "accepted"            # another person crop: no comparison; big noise is no reference
    assert r[3].status == "noise"


def test_a_year_read_upside_down_is_treated_like_the_year(f):
    """0927: the wristband's printed "2026" is read upside down as "9202" (27 photos)."""
    r = f.evaluate("9202", "9202", 0.97, ["9202"], "person_body")
    assert r.status == "uncertain" and "possible_year" in r.reasons
    assert f.evaluate("A9202", "9202", 0.97, ["A9202"], "person_body").accepted      # with a prefix: a bib
    assert f.evaluate("9203", "9203", 0.97, ["9203"], "person_body").accepted
