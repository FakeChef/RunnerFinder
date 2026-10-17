"""SQLite search: the CSV search rules (tests/unit/test_photo_search.py) on the index."""
import os
from pathlib import Path

import pytest

from database import database as db
from database.csv_import import import_batch
from search.db_search import search
from search.photo_search import group_records, load_records, search_records


def write_batch(folder: Path, photos: Path, rows, image_factory, missing=()):
    """A diagnostic batch folder: photo_bibs.csv + photos.csv for (file name, bib) rows."""
    folder.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(f for f, _ in rows))
    for n in names:
        if n not in missing:
            image_factory(photos / n)
    lines = ["photo_no,file,bib,best_confidence,readings,region_ids"]
    lines += [f"{names.index(f) + 1},{photos / f},{b},0.9,1,C1" for f, b in rows]
    (folder / "photo_bibs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plines = ["photo_no,file,status"] + [f"{i + 1},{photos / n},bib_read" for i, n in enumerate(names)]
    (folder / "photos.csv").write_text("\n".join(plines) + "\n", encoding="utf-8")
    return folder


@pytest.fixture
def conn(tmp_path):
    c = db.init_db(tmp_path / "idx.db")
    yield c
    c.close()


@pytest.fixture
def indexed(tmp_path, conn, image_factory):
    batch = write_batch(tmp_path / "batch", tmp_path / "photos",
                        [("one.jpg", "A18866"), ("one.jpg", "B18866"), ("one.jpg", "2026"),
                         ("two.jpg", "00123"), ("three.jpg", "18866"), ("four.jpg", "10")],
                        image_factory)
    import_batch(conn, batch, "Race")
    return batch


def names(groups):
    return sorted((Path(g.photo_identity).name, g.numeric_identifier,
                   tuple(sorted(c.bib for c in g.candidates))) for g in groups)


def test_exact_numeric_matching_keeps_leading_zeros(conn, indexed):
    groups, _ = search(conn, "18866")
    assert names(groups) == [("one.jpg", "18866", ("A18866", "B18866")),
                             ("three.jpg", "18866", ("18866",))]
    # V1.3.1: 123 never matches 00123 as an exact or numeric match, only as a leading-zero alias (tier 4)
    assert [(g.tier, c.match) for g in search(conn, "123")[0] for c in g.candidates] == [(4, "alias")]
    assert names(search(conn, "00123")[0]) == [("two.jpg", "00123", ("00123",))]
    # V1.3.1: "18866x" is a full-bib query: it never returns other letters (A/B18866), only the
    # number read without letters; non-bib queries return nothing
    assert names(search(conn, "18866x")[0]) == [("three.jpg", "18866", ("18866",))]
    assert search(conn, "")[0] == [] and search(conn, "x-y")[0] == [] and search(conn, "ABCD12")[0] == []


def test_one_card_per_photo_and_numeric_bib(conn, indexed):
    assert len(search(conn, "18866")[0]) == 2                   # A/B18866 share one card
    # V0.3.1: "2026" is hidden by the noise filter; grouping is tested with it shown
    assert names(search(conn, "2026", include_noise=True)[0]) == [("one.jpg", "2026", ("2026",))]


def test_filtering_is_optional_and_non_destructive(conn, indexed):
    # V0.3.1: the length/value filters are tested with the noise filter's candidates shown
    kept, filtered = search(conn, "2026", min_length=4, max_length=5, excluded_numeric={"2026"},
                            include_noise=True)
    assert kept == [] and [c.bib for c in filtered] == ["2026"]
    assert len(search(conn, "10", min_length=4, include_noise=True)[1]) == 1
    assert names(search(conn, "2026", include_noise=True)[0]) == [("one.jpg", "2026", ("2026",))]


def test_reject_and_restore_through_sqlite_feedback(conn, indexed):
    group = next(g for g in search(conn, "18866")[0] if Path(g.photo_identity).name == "one.jpg")
    detections = conn.execute("SELECT COUNT(*) FROM bib_detection").fetchone()[0]

    db.reject_bib(conn, group.photo_id, "A18866")
    g = next(g for g in search(conn, "18866")[0] if g.photo_id == group.photo_id)
    assert [c.bib for c in g.visible] == ["B18866"] and [c.bib for c in g.rejected] == ["A18866"]

    db.reject_bib(conn, group.photo_id, "B18866")             # all rejected: card hidden
    assert group.photo_id not in [g.photo_id for g in search(conn, "18866")[0]]
    shown = search(conn, "18866", include_rejected=True)[0]
    assert {c.bib for g in shown if g.photo_id == group.photo_id for c in g.rejected} == {"A18866", "B18866"}

    db.restore_bib(conn, group.photo_id, "A18866")
    g = next(g for g in search(conn, "18866")[0] if g.photo_id == group.photo_id)
    assert [c.bib for c in g.visible] == ["A18866"]
    fb = conn.execute("SELECT verdict, original_bib, numeric_bib FROM detection_feedback").fetchall()
    assert [tuple(r) for r in fb] == [("incorrect", "B18866", "18866")]
    assert conn.execute("SELECT COUNT(*) FROM bib_detection").fetchone()[0] == detections  # AI rows kept


def test_reject_twice_and_restore_without_verdict_are_harmless(conn, indexed):
    pid = search(conn, "00123")[0][0].photo_id
    db.reject_bib(conn, pid, "00123")
    db.reject_bib(conn, pid, "00123")
    assert conn.execute("SELECT COUNT(*) FROM detection_feedback").fetchone()[0] == 1
    db.restore_bib(conn, pid, "00123")
    db.restore_bib(conn, pid, "00123")
    assert names(search(conn, "00123")[0]) == [("two.jpg", "00123", ("00123",))]


def test_event_filter(tmp_path, conn, indexed, image_factory):
    other = write_batch(tmp_path / "batch2", tmp_path / "photos2", [("x.jpg", "18866")], image_factory)
    import_batch(conn, other, "Other race")
    race = conn.execute("SELECT id FROM events WHERE name = 'Race'").fetchone()[0]
    assert len(search(conn, "18866")[0]) == 3                              # all events
    assert {g.event_name for g in search(conn, "18866", event_id=race)[0]} == {"Race"}


def test_missing_original_is_reported_not_guessed(tmp_path, conn, image_factory):
    batch = write_batch(tmp_path / "b", tmp_path / "p", [("gone.jpg", "31087")], image_factory,
                        missing=("gone.jpg",))
    import_batch(conn, batch, "Race")
    g = search(conn, "31087")[0][0]
    assert g.resolved.status == "missing" and g.resolved.path is None


def test_manual_corrections_are_searchable(conn, indexed):
    pid = search(conn, "00123")[0][0].photo_id
    conn.execute("INSERT INTO detection_feedback (photo_id, verdict, original_bib, corrected_bib, numeric_bib) "
                 "VALUES (?, 'corrected', '00123', '00128', '00128')", (pid,))
    db.add_feedback(conn, pid, "added", corrected_bib="T555")
    conn.commit()
    assert search(conn, "00123")[0] == []                                   # replaced
    assert names(search(conn, "00128")[0]) == [("two.jpg", "00128", ("00128",))]
    assert search(conn, "555")[0][0].candidates[0].verification == "added"


def test_same_results_as_csv_search_for_every_number(conn, indexed):
    csv_path = indexed / "photo_bibs.csv"
    records = load_records(csv_path)
    for n in sorted({r.numeric_identifier for r in records if r.numeric_identifier}):
        csv_side = {(os.path.normcase(g.photo_identity), tuple(sorted(r.bib for r in g.records)))
                    for g in group_records(search_records(records, n), csv_path)}
        db_side = {(os.path.normcase(g.photo_identity), tuple(sorted(c.bib for c in g.candidates)))
                   for g in search(conn, n, include_noise=True)[0]}   # CSV search has no noise filter
        assert db_side == csv_side, n
