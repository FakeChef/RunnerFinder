"""
Re-score the stored bib candidates of ONE event with the bib filter (V0.3.1, rule set 2 in V1.3.1).
No OCR is run: the filter works on what is already in the index.

    index_cli.py filter-bibs --event "2026 Race"            # dry run (default): report only
    index_cli.py filter-bibs --event "2026 Race" --apply    # validated backup, then store the results
    index_cli.py filter-bibs --event "2026 Race" --clear --apply   # remove all filter results
    index_cli.py filter-bibs --event "2026 Race" --check-bibs A0175,0175,175     # before/after searches
    index_cli.py filter-bibs --event "2026 Race" --restore-from <backup.db> [--apply]
                                                            # put back the event's filter results

Only evidence_json["filter"] of the selected event's rows changes. No row is ever deleted; raw OCR
texts, stored bibs, other evidence, review decisions (detection_feedback) and other events are
verified unchanged after every apply. Historical events are never re-scored automatically.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ocr.bib_filter import (NON_PERSON_SOURCES, BibCandidateFilter, PhotoCandidate, evaluate_photo,
                            evidence_texts)
from search.filter_status import ACCEPTED, NOISE, UNCERTAIN, status_of


@dataclass
class FilterSummary:
    event_name: str
    rules: str = ""
    applied: bool = False
    cleared: bool = False
    restored_from: str = ""
    backup: str = ""
    backup_validated: bool = False
    rows: int = 0
    visible_before: int = 0          # shown by default (accepted + uncertain)
    visible_after: int = 0
    newly_hidden: int = 0
    newly_visible: int = 0
    numbers_before: int = 0          # distinct numbers with an accepted candidate
    numbers_after: int = 0
    status_before: Counter = field(default_factory=Counter)
    status_after: Counter = field(default_factory=Counter)
    photos: int = 0
    photos_confirmed_before: int = 0     # photos with an accepted candidate
    photos_confirmed_after: int = 0
    photos_findable_before: int = 0      # photos with an accepted or uncertain candidate
    photos_findable_after: int = 0
    reasons: Counter = field(default_factory=Counter)
    examples: dict = field(default_factory=dict)     # reason -> a few "bib <- raw text"
    checks: dict = field(default_factory=dict)       # query -> {"before": ..., "after": ...}
    integrity: dict = field(default_factory=dict)    # after apply: raw/feedback/other events unchanged


def _filter(evidence_json: str | None) -> dict | None:
    if not evidence_json:
        return None
    return json.loads(evidence_json).get("filter")


def _event_id(conn: sqlite3.Connection, event_name: str) -> int:
    ev = conn.execute("SELECT id FROM events WHERE name = ?", (event_name,)).fetchone()
    if ev is None:
        raise ValueError(f"no event named '{event_name}'")
    return ev[0]


def _rows(conn: sqlite3.Connection, event_id: int) -> list[tuple]:
    return conn.execute(
        "SELECT d.id, d.photo_id, d.bib_number, d.numeric_bib, d.confidence, d.ocr_text, d.source, "
        "d.evidence_json, d.bbox_height, COALESCE(d.region_ids, '') FROM bib_detection d "
        "JOIN photos p ON p.id = d.photo_id "
        "WHERE p.event_id = ? AND d.bib_number IS NOT NULL ORDER BY d.id", (event_id,)).fetchall()


def _hash(values) -> str:
    return hashlib.sha256(repr(list(values)).encode("utf-8")).hexdigest()


def fingerprints(conn: sqlite3.Connection, event_id: int) -> dict:
    """Hashes that must not change when only the filter results of one event change."""
    raw, without_filter, full = [], [], []
    cur = conn.execute("SELECT d.* FROM bib_detection d JOIN photos p ON p.id = d.photo_id "
                       "WHERE p.event_id = ? ORDER BY d.id", (event_id,))
    cols = [c[0] for c in cur.description]
    for r in cur:
        row = dict(zip(cols, tuple(r)))
        ev = json.loads(row["evidence_json"]) if row.get("evidence_json") else {}
        ev.pop("filter", None)
        raw.append(tuple(v for k, v in sorted(row.items()) if k != "evidence_json"))
        without_filter.append(json.dumps(ev, sort_keys=True, ensure_ascii=False))
        full.append(row.get("evidence_json"))
    other = conn.execute("SELECT d.* FROM bib_detection d JOIN photos p ON p.id = d.photo_id "
                         "WHERE p.event_id <> ? ORDER BY d.id", (event_id,)).fetchall()
    return {"rows": len(raw), "raw": _hash(raw), "evidence_without_filter": _hash(without_filter),
            "evidence": _hash(full),
            "feedback": _hash(tuple(r) for r in conn.execute("SELECT * FROM detection_feedback ORDER BY id")),
            "other_events": _hash(tuple(r) for r in other),
            "photos": _hash(tuple(r) for r in conn.execute("SELECT * FROM photos ORDER BY id"))}


def validate_backup(backup_path: Path | str, event_id: int, expected: dict) -> dict:
    """Open a backup read-only: integrity ok and the event's rows identical to `expected` (fingerprints)."""
    b = sqlite3.connect(f"file:{Path(backup_path).as_posix()}?mode=ro", uri=True)
    try:
        integrity = b.execute("PRAGMA integrity_check").fetchone()[0]
        fp = fingerprints(b, event_id)
    finally:
        b.close()
    diff = [k for k in expected if fp.get(k) != expected[k]]
    return {"integrity": integrity, "differences": diff, "ok": integrity == "ok" and not diff}


def _search_counts(conn: sqlite3.Connection, query: str, event_id: int) -> dict:
    from search.db_search import search
    groups, filtered = search(conn, query, event_id)
    return {"photos": len(groups),
            "confirmed": sum(g.status == ACCEPTED for g in groups),
            "low_confidence": sum(g.status == UNCERTAIN for g in groups),
            "hidden": sum(1 for c in filtered if c.noise_filtered),
            "tiers": dict(Counter({1: "exact", 2: "numeric", 3: "normalized", 4: "alias"}[g.tier] for g in groups)),
            "bibs": sorted({c.bib for g in groups for c in g.visible})[:8]}


def _stats(rows_status: list[tuple[int, str | None, str]]) -> tuple[Counter, int, int, int]:
    status, confirmed, findable, numbers = Counter(), set(), set(), set()
    for photo_id, numeric, st in rows_status:
        status[st] += 1
        if st == ACCEPTED:
            confirmed.add(photo_id)
            if numeric:
                numbers.add(numeric)
        if st in (ACCEPTED, UNCERTAIN):
            findable.add(photo_id)
    return status, len(confirmed), len(findable), len(numbers)


def _new_results(conn, rows, bib_filter, clear, s) -> tuple[list[tuple[str, int]], list[str]]:
    outside = defaultdict(set)          # event context: photos where a number was read outside a person crop
    for r in rows:
        if r[6] in NON_PERSON_SOURCES and r[3]:
            outside[r[3]].add(r[1])
    results = {}
    if not clear:                        # V1.3.1: candidates of one photo are scored together (text size)
        by_photo = defaultdict(list)
        for r in rows:
            by_photo[r[1]].append(r)
        for photo_rows in by_photo.values():
            cands = [PhotoCandidate(r[2], r[3], r[4],
                                    tuple(evidence_texts(json.loads(r[7]) if r[7] else {}, r[5])), r[6],
                                    r[8], tuple(x for x in r[9].split(";") if x), len(outside.get(r[3], ())))
                     for r in photo_rows]
            for r, res in zip(photo_rows, evaluate_photo(bib_filter, cands)):
                results[r[0]] = res
    updates, statuses = [], []
    for row_id, photo_id, bib, numeric, conf, ocr_text, source, evidence_json, *_ in rows:
        evidence = json.loads(evidence_json) if evidence_json else {}
        if clear:
            evidence.pop("filter", None)
            new_json, st = json.dumps(evidence, ensure_ascii=False), ACCEPTED
        else:
            res = results[row_id]
            evidence["filter"] = res.as_dict()
            new_json, st = json.dumps(evidence, ensure_ascii=False), res.status
            if st != ACCEPTED:
                for reason in res.reasons:
                    s.reasons[reason] += 1
                    ex = s.examples.setdefault(reason, [])
                    if len(ex) < 4:
                        texts = [r.get("raw_text") for r in
                                 (json.loads(new_json).get("readings") or [])][:1] or [ocr_text]
                        ex.append(f"{bib} <- '{texts[0]}' (score {res.score})")
        statuses.append(st)
        if new_json != (evidence_json or "{}"):
            updates.append((new_json, row_id))
    return updates, statuses


def _write(conn: sqlite3.Connection, db_path, updates, event_id, s, tag) -> None:
    """Validated backup, one transaction, then verify that only the filter results changed."""
    from database.migrations import backup_database
    conn.commit()
    before = fingerprints(conn, event_id)
    s.backup = str(backup_database(conn, db_path, tag))
    check = validate_backup(s.backup, event_id, before)
    if not check["ok"]:
        raise RuntimeError(f"backup {s.backup} failed validation: {check}; nothing changed")
    s.backup_validated = True
    with conn:                                    # one transaction: all rows or none
        conn.executemany("UPDATE bib_detection SET evidence_json = ? WHERE id = ?", updates)
    after = fingerprints(conn, event_id)
    s.integrity = {k: before[k] == after[k] for k in
                   ("rows", "raw", "evidence_without_filter", "feedback", "other_events", "photos")}
    s.applied = True
    if not all(s.integrity.values()):
        raise RuntimeError(f"unexpected change outside the filter results: {s.integrity}; "
                           f"restore with --restore-from {s.backup}")


def _checks_after_dry_run(conn, db_path, updates, event_id, queries) -> dict:
    """Search results after the change, computed on a temporary copy (the index is not touched)."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "dry_run.db"
        dest = sqlite3.connect(copy)
        try:
            conn.backup(dest)
            dest.executemany("UPDATE bib_detection SET evidence_json = ? WHERE id = ?", updates)
            dest.commit()
            return {q: _search_counts(dest, q, event_id) for q in queries}
        finally:
            dest.close()


def rescore_event(conn: sqlite3.Connection, db_path: Path | str, event_name: str,
                  bib_filter: BibCandidateFilter, apply: bool = False,
                  clear: bool = False, check_bibs: tuple[str, ...] = ()) -> FilterSummary:
    event_id = _event_id(conn, event_name)
    rows = _rows(conn, event_id)
    s = FilterSummary(event_name, rules="(clear)" if clear else bib_filter.rules_hash,
                      cleared=clear, rows=len(rows))
    s.photos = conn.execute("SELECT COUNT(*) FROM photos WHERE event_id = ?", (event_id,)).fetchone()[0]
    before = [status_of(_filter(r[7])) for r in rows]
    updates, after = _new_results(conn, rows, bib_filter, clear, s)
    _summarize(s, rows, before, after)
    s.checks = {q: {"before": _search_counts(conn, q, event_id)} for q in check_bibs}
    if apply:
        _write(conn, db_path, updates, event_id, s, "pre-filter")
        for q in check_bibs:
            s.checks[q]["after"] = _search_counts(conn, q, event_id)
    elif check_bibs:
        for q, res in _checks_after_dry_run(conn, db_path, updates, event_id, check_bibs).items():
            s.checks[q]["after"] = res
    return s


def restore_event_filter(conn: sqlite3.Connection, db_path: Path | str, event_name: str,
                         backup_path: Path | str, apply: bool = False,
                         check_bibs: tuple[str, ...] = ()) -> FilterSummary:
    """Put back the filter results of ONE event from a backup made by filter-bibs (or any backup of the
    same index). The backup is validated first: integrity ok and the event's raw rows, other evidence,
    review decisions and photos identical to the current index (only filter results may differ)."""
    event_id = _event_id(conn, event_name)
    cur = fingerprints(conn, event_id)
    check = validate_backup(backup_path, event_id, {k: cur[k] for k in
                                                    ("rows", "raw", "evidence_without_filter", "feedback",
                                                     "photos")})
    if not check["ok"]:
        raise ValueError(f"backup does not match this index for event '{event_name}': {check}")
    b = sqlite3.connect(f"file:{Path(backup_path).as_posix()}?mode=ro", uri=True)
    try:
        old = {r[0]: r[7] for r in _rows(b, event_id)}
    finally:
        b.close()
    rows = _rows(conn, event_id)
    s = FilterSummary(event_name, rules="(restore)", restored_from=str(backup_path), rows=len(rows))
    s.photos = conn.execute("SELECT COUNT(*) FROM photos WHERE event_id = ?", (event_id,)).fetchone()[0]
    updates = [(old[r[0]], r[0]) for r in rows if old.get(r[0]) != r[7]]
    before = [status_of(_filter(r[7])) for r in rows]
    after = [status_of(_filter(old.get(r[0]))) for r in rows]
    _summarize(s, rows, before, after)
    s.checks = {q: {"before": _search_counts(conn, q, event_id)} for q in check_bibs}
    if apply:
        _write(conn, db_path, updates, event_id, s, "pre-restore")
        for q in check_bibs:
            s.checks[q]["after"] = _search_counts(conn, q, event_id)
    elif check_bibs:
        for q, res in _checks_after_dry_run(conn, db_path, updates, event_id, check_bibs).items():
            s.checks[q]["after"] = res
    return s


def _summarize(s: FilterSummary, rows, before: list[str], after: list[str]) -> None:
    s.status_before, s.photos_confirmed_before, s.photos_findable_before, s.numbers_before = _stats(
        [(r[1], r[3], st) for r, st in zip(rows, before)])
    s.status_after, s.photos_confirmed_after, s.photos_findable_after, s.numbers_after = _stats(
        [(r[1], r[3], st) for r, st in zip(rows, after)])
    for b, a in zip(before, after):
        vb, va = b != NOISE, a != NOISE
        s.visible_before += vb
        s.visible_after += va
        s.newly_hidden += vb and not va
        s.newly_visible += va and not vb


def summary_json(s: FilterSummary) -> str:
    d = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in s.__dict__.items()}
    return json.dumps(d, indent=1, ensure_ascii=False, default=str)


def copy_index(src: Path | str, dst: Path | str) -> Path:
    """Consistent copy of an index (sqlite backup API, WAL included). Used for isolated validation."""
    s = sqlite3.connect(f"file:{Path(src).as_posix()}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    return Path(dst)

