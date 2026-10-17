import csv
from pathlib import Path

from search.photo_search import (load_records, load_overrides, numeric_part,
                                 filter_records, group_records, resolve_photo_path,
                                 save_override, search_records)


def test_numeric_exact_and_prefixes():
    assert numeric_part("A18866") == "18866"
    assert numeric_part("B18866") == "18866"
    assert numeric_part("00123") == "00123"
    assert search_records(load_records(_csv("A18866\nB18866\n00123\n")), "18866")[0].bib == "A18866"
    assert len(search_records(load_records(_csv("A18866\nB18866\n00123\n")), "123")) == 0


def test_old_csv_derives_numeric_identifier(tmp_path):
    path = tmp_path / "photo_bibs.csv"
    path.write_text("photo_no,file,bib\n1,x.jpg,UT00123\n", encoding="utf-8")
    assert load_records(path)[0].numeric_identifier == "00123"


def test_duplicate_relative_filename_is_not_guessed(tmp_path):
    csv_path = tmp_path / "batch" / "photo_bibs.csv"
    csv_path.parent.mkdir()
    (csv_path.parent / "a").mkdir(); (csv_path.parent / "b").mkdir()
    (csv_path.parent / "a" / "same.jpg").write_bytes(b"x")
    (csv_path.parent / "b" / "same.jpg").write_bytes(b"x")
    assert resolve_photo_path("same.jpg", csv_path).status == "ambiguous"


def test_optional_photo_root_resolves_relative_path_without_drive_scan(tmp_path):
    csv_path = tmp_path / "batch" / "photo_bibs.csv"
    csv_path.parent.mkdir()
    root = tmp_path / "photos"
    root.mkdir()
    (root / "IMG_1.jpg").write_bytes(b"x")
    result = resolve_photo_path("IMG_1.jpg", csv_path, root)
    assert result.status == "unique"
    assert result.path == (root / "IMG_1.jpg").resolve()


def test_grouping_deduplicates_photo_numeric_and_keeps_full_bibs(tmp_path):
    root = tmp_path / "photos"
    root.mkdir(); (root / "one.jpg").write_bytes(b"x")
    csv_path = tmp_path / "batch" / "photo_bibs.csv"
    csv_path.parent.mkdir()
    csv_path.write_text(
        "photo_no,file,bib,numeric_identifier\n"
        "1,one.jpg,A18866,18866\n"
        "1,one.jpg,B18866,18866\n"
        "1,one.jpg,2026,2026\n", encoding="utf-8")
    groups = group_records(load_records(csv_path), csv_path, root)
    bib_group = next(g for g in groups if g.numeric_identifier == "18866")
    assert len(groups) == 2
    assert {r.bib for r in bib_group.records} == {"A18866", "B18866"}


def test_filtering_is_optional_and_non_destructive(tmp_path):
    path = _csv("2026\n10\n18866\n")
    records = load_records(path)
    kept, filtered = filter_records(records, min_length=4, max_length=5, excluded_numeric={"2026"})
    assert [r.bib for r in kept] == ["18866"]
    assert {r.bib for r in filtered} == {"2026", "10"}
    assert len(load_records(path)) == 3


def test_override_can_reject_and_restore(tmp_path):
    csv_path = tmp_path / "photo_bibs.csv"
    csv_path.write_text("photo_no,file,bib\n1,x.jpg,18866\n", encoding="utf-8")
    save_override(csv_path, r"C:\photo.jpg", "18866", "reject")
    assert load_overrides(csv_path)[(str(csv_path.resolve()), r"C:\photo.jpg", "18866")] == "reject"
    save_override(csv_path, r"C:\photo.jpg", "18866", "restore")
    assert load_overrides(csv_path)[(str(csv_path.resolve()), r"C:\photo.jpg", "18866")] == "restore"


def _csv(bibs: str) -> Path:
    import tempfile
    path = Path(tempfile.mkdtemp()) / "photo_bibs.csv"
    path.write_text("photo_no,file,bib\n" + "\n".join(f"1,x.jpg,{b}" for b in bibs.splitlines()) + "\n", encoding="utf-8")
    return path
