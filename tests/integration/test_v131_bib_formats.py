"""V1.3.1 universal bib formats: filter statuses, search tiers (exact / numeric / normalized / alias),
the mandatory A0175 cases, noise, event isolation, review decisions, re-scoring and restore, videos."""
import json
import sqlite3
from pathlib import Path

import pytest

from config import BibFormatConfig
from database import database as db
from indexing.filter_bibs import fingerprints, rescore_event, restore_event_filter
from ocr.bib_filter import BibCandidateFilter, filtered_evidence_json
from ocr.bib_format import clean_bib
from search.db_search import parse_query, search
from search.video_search import search_videos

# (file, bib, numeric, confidence, raw OCR text) - stored like a V1.3.1 analysis (rule set 2)
RACE = [
    ("a0175.jpg", "A0175", "0175", 0.97, "A0175"),
    ("b0175.jpg", "B0175", "0175", 0.96, "B0175"),
    ("a_space_0175.jpg", "0175", "0175", 0.95, "A 0175"),     # OCR split "A 0175": prefix not in the bib
    ("bare175.jpg", "175", "175", 0.99, "175"),               # 3 digits: uncertain
    ("n1750.jpg", "1750", "1750", 0.97, "1750"),              # longer numbers that contain 175
    ("n11750.jpg", "11750", "11750", 0.97, "11750"),
    ("n00128.jpg", "00128", "00128", 0.95, "00128"),
    ("n128.jpg", "128", "128", 0.99, "128"),
    ("ut1234.jpg", "UT1234", "1234", 0.95, "UT1234"),
    ("n1234.jpg", "1234", "1234", 0.95, "1234"),
    ("n12345.jpg", "12345", "12345", 0.95, "12345"),
    ("n91234.jpg", "91234", "91234", 0.95, "91234"),
    ("b60914.jpg", "B60914", "60914", 0.70, "B 60914"),
    ("a1076.jpg", "A1076", "1076", 0.97, "A1076"),
    ("year_text.jpg", "2026", "2026", 0.99, "2026 PARENT-CHILD"),   # printed year: noise
    ("year_alone.jpg", "2026", "2026", 1.0, "2026"),                # lone year: uncertain
    ("dist.jpg", "42KM", "42", 0.95, "42KM"),                       # distance: noise
    ("dist_k.jpg", "210", "210", 0.95, "210K"),
    ("bag.jpg", "1365", "1365", 0.97, "13-65"),                     # bag-check: noise
    ("bag5.jpg", "72995", "72995", 0.90, "6-72995"),
    ("wave.jpg", "A20", "20", 0.92, "A20"),                         # group / wave label: noise
]
OTHER = [("other_a0175.jpg", "A0175", "0175", 0.97, "A0175")]       # same bib, another event


def _store(conn, event, rows, bib_filter=None):
    ev = db.get_or_create_event(conn, event)
    run = db.start_run(conn, ev, source="ocr", source_folder=event)
    ids = {}
    for name, bib, num, conf, raw in rows:
        path = f"{event}/{name}"
        pid = db.insert_photo(conn, ev, {"file_path": path, "path_key": path, "file_name": name})
        evidence = json.dumps({"readings": [{"raw_text": raw}]})
        if bib_filter is not None:
            evidence, _ = filtered_evidence_json(bib_filter, evidence, bib, num, conf, raw, "person_body")
        db.save_photo_result(conn, pid, run, [{
            "bib_number": bib, "numeric_bib": num, "confidence": conf, "ocr_text": raw,
            "source": "person_body", "evidence_json": evidence}])
        ids[name] = pid
    conn.commit()
    return ev, ids


@pytest.fixture
def index(tmp_path):
    path = tmp_path / "data" / "idx.db"
    conn = db.init_db(path)
    f = BibCandidateFilter()
    race, ids = _store(conn, "Race", RACE, f)
    other, other_ids = _store(conn, "Other", OTHER, f)
    yield conn, path, race, other, {**ids, **other_ids}
    conn.close()


def results(conn, q, event_id=None, **kw):
    """[(file, tier, match, status, bibs)] in result order."""
    groups, _ = search(conn, q, event_id, **kw)
    return [(Path(g.photo_identity).name, g.tier, g.candidates[0].match, g.status,
             tuple(c.bib for c in g.candidates)) for g in groups]


# ------------------------------------------------------------------ mandatory A0175 validation
def test_search_A0175_exact_first_and_never_B0175(index):
    conn, _, race, *_ = index
    r = results(conn, "A0175", race)
    assert r[0] == ("a0175.jpg", 1, "exact", "accepted", ("A0175",))
    assert ("a_space_0175.jpg", 3, "normalized", "accepted", ("0175",)) in r     # OCR line "A 0175"
    assert all("B0175" not in x[4] for x in r)                                 # A0175 is not B0175
    assert [x[1] for x in r] == sorted(x[1] for x in r)                        # tiers in order
    assert {x[0] for x in r if x[1] < 4} == {"a0175.jpg", "a_space_0175.jpg"}


def test_search_0175_returns_prefixed_bibs_labelled(index):
    conn, _, race, *_ = index
    r = results(conn, "0175", race)
    assert r[0] == ("a_space_0175.jpg", 1, "exact", "accepted", ("0175",))
    numeric = {x[0]: x[4] for x in r if x[2] == "numeric"}
    assert numeric == {"a0175.jpg": ("A0175",), "b0175.jpg": ("B0175",)}       # both, with their prefix
    assert [x for x in r if x[2] == "alias"] == [("bare175.jpg", 4, "alias", "uncertain", ("175",))]


def test_search_175_no_substring_alias_only_below(index):
    conn, _, race, *_ = index
    r = results(conn, "175", race)
    assert r[0] == ("bare175.jpg", 1, "exact", "uncertain", ("175",))         # the 3-digit read itself
    assert {x[0] for x in r[1:]} == {"a0175.jpg", "b0175.jpg", "a_space_0175.jpg"}
    assert all(x[1] == 4 and x[2] == "alias" for x in r[1:])                   # aliases rank last
    assert not {"n1750.jpg", "n11750.jpg"} & {x[0] for x in r}                 # never substring


def test_A0175_and_B0175_stay_distinct(index):
    conn, _, race, *_ = index
    assert [x[0] for x in results(conn, "B0175", race) if x[1] < 4] == ["b0175.jpg"]
    assert "b0175.jpg" not in [x[0] for x in results(conn, "A0175", race)]


def test_no_exact_matches_lost_for_every_stored_bib(index):
    conn, _, race, *_ = index
    for name, bib, num, conf, raw in RACE:
        groups, filtered = search(conn, bib, race, include_noise=True)
        assert any(Path(g.photo_identity).name == name and g.tier == 1 for g in groups), bib


# ------------------------------------------------------------------ synthetic formats
@pytest.mark.parametrize("query, full, prefix, core", [
    ("128", "128", "", "128"), ("00128", "00128", "", "00128"), ("A1076", "A1076", "A", "1076"),
    ("B60914", "B60914", "B", "60914"), ("UT1234", "UT1234", "UT", "1234"), ("F1", "F1", "F", "1"),
    ("A-0175", "A0175", "A", "0175"), ("A 0175", "A0175", "A", "0175"), ("a0175", "A0175", "A", "0175"),
])
def test_query_forms(query, full, prefix, core):
    q = parse_query(query)
    assert (q.full, q.prefix, q.core) == (full, prefix, core)


def test_separator_queries_equal_the_plain_full_bib(index):
    conn, _, race, *_ = index
    assert results(conn, "A-0175", race) == results(conn, "A 0175", race) == results(conn, "A0175", race)


def test_leading_zero_alias_128_00128(index):
    conn, _, race, *_ = index
    assert results(conn, "00128", race) == [("n00128.jpg", 1, "exact", "accepted", ("00128",)),
                                            ("n128.jpg", 4, "alias", "uncertain", ("128",))]
    assert results(conn, "128", race) == [("n128.jpg", 1, "exact", "uncertain", ("128",)),
                                          ("n00128.jpg", 4, "alias", "accepted", ("00128",))]


def test_prefixed_formats_and_numeric_core(index):
    conn, _, race, *_ = index
    assert results(conn, "UT1234", race)[0] == ("ut1234.jpg", 1, "exact", "accepted", ("UT1234",))
    r = results(conn, "1234", race)
    assert {x[0] for x in r} == {"n1234.jpg", "ut1234.jpg"}                    # never 12345 / 91234
    assert results(conn, "60914", race) == [("b60914.jpg", 2, "numeric", "accepted", ("B60914",))]
    assert results(conn, "A1076", race) == [("a1076.jpg", 1, "exact", "accepted", ("A1076",))]


def test_short_bib_F1_is_not_eligible_without_race_data(index):
    conn, _, race, *_ = index
    assert clean_bib("F1", BibFormatConfig()) is None          # format stage: never stored (< 3 chars)
    assert results(conn, "F1", race) == []


# ------------------------------------------------------------------ noise and ambiguity
def test_noise_is_hidden_with_reasons_and_uncertain_is_shown(index):
    conn, _, race, *_ = index
    groups, filtered = search(conn, "2026", race)
    assert [(Path(g.photo_identity).name, g.status) for g in groups] == [("year_alone.jpg", "uncertain")]
    assert [(c.bib, c.filter_reasons) for c in filtered] == [("2026", "year; text_context")]
    for q, reason in (("42", "too_few_digits"), ("210", "distance"), ("1365", "likely_storage_number"),
                      ("72995", "likely_storage_number"), ("20", "too_few_digits")):
        groups, filtered = search(conn, q, race)
        assert groups == [] and any(reason in c.filter_reasons for c in filtered), q
        assert search(conn, q, race, include_noise=True)[0], q                   # reviewable


def test_events_stay_isolated(index):
    conn, _, race, other, _ = index
    assert [x[0] for x in results(conn, "A0175", other)] == ["other_a0175.jpg"]
    assert "other_a0175.jpg" not in [x[0] for x in results(conn, "A0175", race)]
    assert {x[0] for x in results(conn, "A0175") if x[1] == 1} == {"a0175.jpg", "other_a0175.jpg"}


def test_reject_and_restore_still_work(index):
    conn, _, race, _, ids = index
    db.reject_bib(conn, ids["a0175.jpg"], "A0175")
    assert "a0175.jpg" not in [x[0] for x in results(conn, "A0175", race)]
    assert "a0175.jpg" in [x[0] for x in results(conn, "A0175", race, include_rejected=True)]
    db.restore_bib(conn, ids["a0175.jpg"], "A0175")
    assert results(conn, "A0175", race)[0][0] == "a0175.jpg"
    db.add_feedback(conn, ids["bare175.jpg"], "correct", original_bib="175")      # verdict beats filter
    conn.commit()
    assert results(conn, "175", race)[0] == ("bare175.jpg", 1, "exact", "accepted", ("175",))


# ------------------------------------------------------------------ existing data: re-scoring
OLD_RULES_ROWS = [("old_a1076.jpg", "A1076", "1076", 0.97, "A1076"), ("old_year.jpg", "2026", "2026", 1.0, "2026"),
                  ("old_b60914.jpg", "B60914", "60914", 0.9, "B60914")]


@pytest.fixture
def old_index(tmp_path):
    """An event analysed by V1.3.0 (rule set 1): A1076 rejected as low_score, 2026 as year."""
    path = tmp_path / "data" / "old.db"
    conn = db.init_db(path)
    ev, ids = _store(conn, "Old", OLD_RULES_ROWS)
    old = {"A1076": {"accepted": False, "score": 2, "reasons": ["low_score"], "rules": "v1"},
           "2026": {"accepted": False, "score": 0, "reasons": ["year"], "rules": "v1"},
           "B60914": {"accepted": True, "score": 10, "reasons": [], "rules": "v1"}}
    for bib, flt in old.items():
        row_id, ej = conn.execute("SELECT id, evidence_json FROM bib_detection WHERE bib_number = ?", (bib,)).fetchone()
        e = json.loads(ej)
        e["filter"] = flt
        conn.execute("UPDATE bib_detection SET evidence_json = ? WHERE id = ?", (json.dumps(e), row_id))
    other, _ = _store(conn, "Untouched", [("u.jpg", "A2222", "2222", 0.9, "A2222")])
    conn.commit()
    yield conn, path, ev, other, ids
    conn.close()


def test_old_results_are_searchable_as_low_confidence_without_rescoring(old_index):
    conn, _, ev, *_ = old_index
    assert results(conn, "A1076", ev) == [("old_a1076.jpg", 1, "exact", "uncertain", ("A1076",))]
    assert results(conn, "2026", ev) == []                     # rule-set-1 "year" stays hidden


def test_rescore_dry_run_apply_restore(old_index):
    conn, path, ev, other, ids = old_index
    db.reject_bib(conn, ids["old_b60914.jpg"], "B60914")       # a review decision that must survive
    before = fingerprints(conn, ev)
    dry = rescore_event(conn, path, "Old", BibCandidateFilter(), check_bibs=("A1076", "1076", "2026"))
    assert not dry.applied and fingerprints(conn, ev) == before and not (path.parent / "backups").exists()
    assert dry.checks["A1076"]["before"]["low_confidence"] == 1 and dry.checks["A1076"]["after"]["confirmed"] == 1
    assert dry.checks["2026"]["before"]["photos"] == 0 and dry.checks["2026"]["after"]["low_confidence"] == 1
    assert (dry.photos_confirmed_before, dry.photos_confirmed_after) == (1, 2)

    s = rescore_event(conn, path, "Old", BibCandidateFilter(), apply=True)
    assert s.applied and s.backup_validated and all(s.integrity.values()), s.integrity
    assert results(conn, "A1076", ev)[0][3] == "accepted"
    assert fingerprints(conn, other)["raw"] == fingerprints(conn, other)["raw"]
    after = fingerprints(conn, ev)
    assert {k: before[k] == after[k] for k in ("raw", "evidence_without_filter", "feedback", "photos")} == \
        {"raw": True, "evidence_without_filter": True, "feedback": True, "photos": True}
    assert after["evidence"] != before["evidence"]

    # restore from the validated backup: back to the rule-set-1 results, review decision kept
    r = restore_event_filter(conn, path, "Old", s.backup, apply=True)
    assert r.applied and all(r.integrity.values())
    assert fingerprints(conn, ev)["evidence"] == before["evidence"]
    assert results(conn, "A1076", ev)[0][3] == "uncertain"
    assert [tuple(x) for x in conn.execute("SELECT verdict FROM detection_feedback")] == [("incorrect",)]


def test_restore_refuses_a_backup_of_another_index(old_index, tmp_path):
    conn, path, *_ = old_index
    foreign = tmp_path / "foreign.db"
    fc = db.init_db(foreign)
    _store(fc, "Old", [("x.jpg", "A9999", "9999", 0.9, "A9999")])
    fc.close()
    with pytest.raises(ValueError, match="does not match"):
        restore_event_filter(conn, path, "Old", foreign, apply=True)


def test_rescore_needs_an_existing_event(old_index):
    conn, path, *_ = old_index
    with pytest.raises(ValueError, match="no event named"):
        rescore_event(conn, path, "Nope", BibCandidateFilter())


# ------------------------------------------------------------------ videos
def test_video_search_uses_the_same_query_rules(index):
    conn, _, race, *_ = index
    cur = conn.execute("INSERT INTO videos (event_id, file_path, path_key, file_name, analysis_status, "
                       "recognition_status) VALUES (?, 'v.mp4', 'v.mp4', 'v.mp4', 'done', 'VERIFIED')", (race,))
    conn.execute("INSERT INTO video_candidates (video_id, bib_number, numeric_bib, role, votes, frames) "
                 "VALUES (?, 'A0175', '0175', 'accepted', 4, 5)", (cur.lastrowid,))
    conn.commit()
    assert [m.match for m in search_videos(conn, "A0175", race)] == ["exact"]
    assert [m.match for m in search_videos(conn, "0175", race)] == ["numeric"]
    assert [m.match for m in search_videos(conn, "175", race)] == ["alias"]
    assert search_videos(conn, "B0175", race) == [] and search_videos(conn, "1750", race) == []
