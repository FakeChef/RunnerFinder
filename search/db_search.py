"""
Bib search on the SQLite index (V0.2). Same rules as the CSV search
in search/photo_search.py, which stays available as a fallback:

  * one result group per photo and numeric bib, with every full-bib
    candidate and its OCR evidence attached
  * rejected candidates (feedback 'incorrect') are hidden unless asked for;
    reject/restore only write detection_feedback, never the AI results
  * optional length / excluded-value filters return the filtered candidates
    separately and delete nothing
  * original paths are absolute in the database; a missing file is reported,
    never guessed
  * V0.3.1 / V1.3.1: the filter status (evidence_json["filter"], ocr/bib_filter.status_of) decides
    visibility: "noise" is hidden unless include_noise, "uncertain" is shown after the confirmed
    results (low confidence), "accepted" is normal; a manual 'correct' verdict always shows them
  * V1.3.1 queries: digits ("0175") or a full bib ("A0175", "a-0175", "A 0175"): upper-cased,
    separators removed, <=3 letters + one digit run + <=3 letters (parse_query)
  * match tiers, best first:
      1 exact       the stored bib is the query ("0175" -> 0175, "A0175" -> A0175)
      2 numeric     digits query: the stored bib's numeric part is the query (A0175 and B0175 for
                    "0175", each labelled with its full bib); full-bib query: the stored bib is the
                    number without letters (prefix not read)
      3 normalized  (V1.2) the stored bib, its OCR line or a raw reading is <=3 letters, separators,
                    ONE digit run, separators, <=3 letters, and equals the query ("60914B" that the OCR
                    digit fix stored as "609148"; "A 0175" for the query "A0175")
      4 alias       leading zeros only: "175" <-> "0175" / "00175" (always below the other tiers)
    A full-bib query never returns other letters: "A0175" does not find B0175.
    Never fuzzy and never partial: digits split by a separator ("6-7208S",
    "609 14") or longer/shorter runs ("609140", "6091") do not match.
    Search layer only: stored OCR values and the schema are unchanged.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from search.filter_status import ACCEPTED, NOISE, SOFT_REASONS, UNCERTAIN, status_of
from search.photo_search import ResolvedPhoto


@dataclass(frozen=True)
class BibCandidate:
    photo_id: int
    bib: str
    numeric_identifier: str
    confidence: float | None
    readings: int | None
    region_ids: str
    verification: str          # unverified | correct | corrected | added | incorrect
    rejected: bool             # verification == 'incorrect'
    event_id: int
    event_name: str
    file_path: str
    file_status: str
    noise_filtered: bool = False   # hidden by the filter (status "noise", ocr/bib_filter.py)
    filter_reasons: str = ""       # e.g. "likely_storage_number; distance" or "short_number"
    match: str = "numeric"         # exact | numeric | normalized | alias (see module docstring)
    matched_text: str = ""         # normalized matches: the OCR text that matched ("60914B")
    status: str = ACCEPTED         # V1.3.1: accepted | uncertain | noise

    @property
    def tier(self) -> int:
        return MATCH_TIERS.get(self.match, 2)

    @property
    def low_confidence(self) -> bool:
        return self.status == UNCERTAIN


MATCH_TIERS = {"exact": 1, "numeric": 2, "normalized": 3, "alias": 4}
STATUS_RANK = {ACCEPTED: 0, UNCERTAIN: 1, NOISE: 2}

# <=3 letters, separators, one digit run, separators, <=3 letters
_NORMALIZED_FORM = re.compile(r"([A-Z]{0,3})[ \-_./·]*(\d+)[ \-_./·]*([A-Z]{0,3})")
_QUERY_SEPARATORS = re.compile(r"[\s\-_.#·:/]+")
MAX_ALIAS_LENGTH = 8           # longest stored core an alias is generated for (bib format: <= 8 chars)


def normalized_number(text: str | None) -> str | None:
    """The digit run of a bib-like text ("B-60914" -> "60914", "60914B" -> "60914"), else None."""
    m = _NORMALIZED_FORM.fullmatch((text or "").strip().upper())
    return m.group(2) if m else None


def _normalized_parts(text: str | None) -> tuple[str, str, str] | None:
    m = _NORMALIZED_FORM.fullmatch((text or "").strip().upper())
    return m.groups() if m else None


@dataclass(frozen=True)
class Query:
    """A parsed search query: full = letters + core + letters, without separators."""
    full: str
    prefix: str
    core: str
    suffix: str

    @property
    def digits_only(self) -> bool:
        return not (self.prefix or self.suffix)

    def aliases(self) -> list[str]:
        """Other spellings of the core that differ only by leading zeros ("175" <-> "0175", "00175")."""
        stripped = self.core.lstrip("0")
        if not stripped:
            return []
        return [("0" * k) + stripped for k in range(MAX_ALIAS_LENGTH - len(stripped) + 1)
                if ("0" * k) + stripped != self.core]

    def letters_match(self, bib: str) -> bool:
        """Full-bib query: the stored bib has the same letters, or none at all (prefix not read)."""
        p = _normalized_parts(bib)
        return p is not None and (p[0], p[2]) in ((self.prefix, self.suffix), ("", ""))


def parse_query(query: str | None) -> Query | None:
    """'0175', 'A0175', 'a-0175', 'A 0175', 'UT1234' -> Query; None if it is not a bib-like query."""
    full = _QUERY_SEPARATORS.sub("", (query or "").strip().upper())
    m = re.fullmatch(r"([A-Z]{0,3})(\d+)([A-Z]{0,3})", full)
    return Query(full, *m.groups()) if m else None


@dataclass(frozen=True)
class DBSearchGroup:
    """One photo / numeric-bib result with all its full-bib candidates."""
    event_id: int
    event_name: str
    photo_id: int
    photo_identity: str        # absolute original path
    resolved: ResolvedPhoto
    numeric_identifier: str
    candidates: tuple[BibCandidate, ...]

    @property
    def tier(self) -> int:
        """Best match tier of the photo (1 exact, 2 numeric, 3 normalized, 4 alias)."""
        return min(c.tier for c in self.candidates)

    @property
    def status(self) -> str:
        """Best filter status of the shown candidates (accepted < uncertain < noise)."""
        cs = self.visible or self.candidates
        return min((c.status for c in cs), key=lambda s: STATUS_RANK[s])

    @property
    def visible(self) -> tuple[BibCandidate, ...]:
        return tuple(c for c in self.candidates if not c.rejected)

    @property
    def rejected(self) -> tuple[BibCandidate, ...]:
        return tuple(c for c in self.candidates if c.rejected)


# AI results (with any verdict on them), plus bibs the photographer corrected
# or added by hand. An AI bib replaced by a 'corrected' verdict is not shown;
# the corrected value is. :n is a JSON list of numeric parts (the core and its aliases).
_CANDIDATES_SQL = """
SELECT d.photo_id, d.bib_number AS bib, d.numeric_bib, d.confidence, d.readings,
       COALESCE(d.region_ids, '') AS region_ids,
       COALESCE(f.verdict, 'unverified') AS verification,
       e.id AS event_id, e.name AS event_name, p.file_path, p.file_status,
       json_extract(d.evidence_json, '$.filter') AS filter_json,
       d.ocr_text, json_extract(d.evidence_json, '$.readings') AS readings_json
FROM bib_detection d
JOIN photos p ON p.id = d.photo_id
JOIN events e ON e.id = p.event_id
LEFT JOIN detection_feedback f ON f.photo_id = d.photo_id AND f.original_bib = d.bib_number
WHERE d.numeric_bib IN (SELECT value FROM json_each(:n)) AND d.bib_number IS NOT NULL
  AND COALESCE(f.verdict, '') <> 'corrected'
  AND (:event_id IS NULL OR p.event_id = :event_id)
UNION ALL
SELECT f.photo_id, f.corrected_bib, f.numeric_bib, 1.0, NULL, '', f.verdict,
       e.id, e.name, p.file_path, p.file_status, NULL, NULL, NULL
FROM detection_feedback f
JOIN photos p ON p.id = f.photo_id
JOIN events e ON e.id = p.event_id
WHERE f.verdict IN ('corrected', 'added') AND f.numeric_bib IN (SELECT value FROM json_each(:n))
  AND (:event_id IS NULL OR p.event_id = :event_id)
"""


# V1.2 tier 3: rows whose numeric part differs but whose texts contain the query (ocr_text is
# the raw text of a reading; the evidence JSON is only searched for rows with several readings);
# the exact normalization rule is then checked in Python (normalized_number)
_NORMALIZED_SQL = """
SELECT d.photo_id, d.bib_number AS bib, d.numeric_bib, d.confidence, d.readings,
       COALESCE(d.region_ids, '') AS region_ids,
       COALESCE(f.verdict, 'unverified') AS verification,
       e.id AS event_id, e.name AS event_name, p.file_path, p.file_status,
       json_extract(d.evidence_json, '$.filter') AS filter_json,
       d.ocr_text, json_extract(d.evidence_json, '$.readings') AS readings_json
FROM bib_detection d
JOIN photos p ON p.id = d.photo_id
JOIN events e ON e.id = p.event_id
LEFT JOIN detection_feedback f ON f.photo_id = d.photo_id AND f.original_bib = d.bib_number
WHERE d.bib_number IS NOT NULL AND COALESCE(d.numeric_bib, '') <> :q
  AND (instr(d.bib_number, :q) > 0 OR instr(COALESCE(d.ocr_text, ''), :q) > 0
       OR (COALESCE(d.readings, 0) > 1 AND instr(COALESCE(d.evidence_json, ''), :q) > 0))
  AND COALESCE(f.verdict, '') <> 'corrected'
  AND (:event_id IS NULL OR p.event_id = :event_id)
"""


def _raw_texts(ocr_text: str | None, readings_json: str | None) -> list[str]:
    texts = []
    try:
        texts += [r.get("raw_text") for r in json.loads(readings_json or "[]") if isinstance(r, dict)]
    except (TypeError, ValueError):
        pass
    return [t for t in [*texts, ocr_text] if t]


def _normalized_text(q: str | Query, bib: str, ocr_text: str | None, readings_json: str | None) -> str | None:
    """The first of the row's texts (raw readings, OCR line, stored bib) that normalizes to the query:
    its digit run equals the core and, for a full-bib query, its letters equal the query's letters."""
    if isinstance(q, str):
        q = parse_query(q)
    for text in [*_raw_texts(ocr_text, readings_json), bib]:
        p = _normalized_parts(text)
        if p and p[1] == q.core and (q.digits_only or (p[0], p[2]) == (q.prefix, q.suffix)):
            return text
    return None


def _other_letters_seen(q: Query, ocr_text: str | None, readings_json: str | None) -> bool:
    """Full-bib query, stored bib without letters: a raw reading shows different letters
    ("A 0175" stored as "0175" is not B0175)."""
    for text in _raw_texts(ocr_text, readings_json):
        p = _normalized_parts(text)
        if p and (p[0] or p[2]) and (p[0], p[2]) != (q.prefix, q.suffix):
            return True
    return False


def _status(filter_json: str | None, verification: str) -> tuple[str, list[str]]:
    """(status, reasons) of a stored row; a photographer's verdict always wins over the filter."""
    if verification in ("correct", "corrected", "added"):
        return ACCEPTED, []
    try:
        f = json.loads(filter_json) if filter_json else None
    except ValueError:
        f = None
    return status_of(f), list((f or {}).get("reasons") or [])


def _candidate(r, q: Query, match: str, status: str, reasons: list[str], numeric: str | None = None,
               matched_text: str = "") -> BibCandidate:
    return BibCandidate(r[0], r[1], numeric or r[2], r[3], r[4], r[5], r[6], r[6] == "incorrect",
                        r[7], r[8], r[9], r[10], status == NOISE,
                        "; ".join(reasons) if status != ACCEPTED else "", match, matched_text, status)


def find_candidates(conn: sqlite3.Connection, query: str,
                    event_id: int | None = None) -> list[BibCandidate]:
    """Tiers 1-4 (see the module docstring); queries that are not bib-like return nothing."""
    q = parse_query(query)
    if q is None:
        return []
    out: dict[tuple[int, str], BibCandidate] = {}

    def keep(c: BibCandidate) -> None:
        key = (c.photo_id, c.bib)
        best = out.get(key)          # schema-2 rows may hold several readings per bib
        if best is None or (c.tier, -(c.confidence or 0)) < (best.tier, -(best.confidence or 0)):
            out[key] = c

    numerics = [q.core, *q.aliases()]
    for r in conn.execute(_CANDIDATES_SQL, {"n": json.dumps(numerics), "event_id": event_id}):
        bib, numeric = r[1], r[2]
        status, reasons = _status(r[11], r[6])
        if bib == q.full:
            match = "exact"
        elif q.digits_only:
            match = "numeric" if numeric == q.core else "alias"
        elif not q.letters_match(bib):
            continue                              # other letters: a different bib (A0175 is not B0175)
        elif _other_letters_seen(q, r[12], r[13]):
            continue                              # stored without letters, but the OCR line shows others
        elif numeric == q.core:
            # the stored bib has no letters: the OCR line may still show them ("A 0175")
            seen = _normalized_text(q, bib, r[12], r[13])
            match = "normalized" if seen and seen != bib else "numeric"
            if match == "normalized":
                keep(_candidate(r, q, match, status, reasons, matched_text=seen))
                continue
        else:
            match = "alias"
        keep(_candidate(r, q, match, status, reasons))
    for r in conn.execute(_NORMALIZED_SQL, {"q": q.core, "event_id": event_id}):
        text = _normalized_text(q, r[1], r[12], r[13])
        if text is None:
            continue
        status, reasons = _status(r[11], r[6])
        # reasons that only describe the stored form ("609148": 6 digits) do not hide a normalized
        # match; specific noise (bag check, distance, year in text, ...) still does
        if status == NOISE and set(reasons) <= SOFT_REASONS:
            status = UNCERTAIN
        keep(_candidate(r, q, "normalized", status, reasons, numeric=q.core, matched_text=text))
    return list(out.values())


def filter_candidates(candidates: list[BibCandidate], min_length: int | None = None,
                      max_length: int | None = None, excluded_numeric: set[str] | None = None
                      ) -> tuple[list[BibCandidate], list[BibCandidate]]:
    """Return (kept, filtered) without changing anything stored."""
    excluded_numeric = excluded_numeric or set()
    kept, filtered = [], []
    for c in candidates:
        n = c.numeric_identifier
        drop = ((min_length is not None and len(n) < min_length)
                or (max_length is not None and len(n) > max_length)
                or n in excluded_numeric)
        (filtered if drop else kept).append(c)
    return kept, filtered


def resolve_stored_path(file_path: str, file_status: str) -> ResolvedPhoto:
    if file_path and os.path.isfile(file_path):
        return ResolvedPhoto(Path(file_path), "unique", "original full path")
    detail = f"original path does not exist: {file_path}"
    if file_status == "missing":
        detail += " (marked missing at the last scan)"
    return ResolvedPhoto(None, "missing", detail)


def group_candidates(candidates: list[BibCandidate]) -> list[DBSearchGroup]:
    """One group per photo and numeric bib; full-bib candidates sorted by tier, then confidence."""
    groups: dict[tuple[int, str], list[BibCandidate]] = {}
    for c in candidates:
        groups.setdefault((c.photo_id, c.numeric_identifier), []).append(c)
    out = []
    for (photo_id, numeric), cs in groups.items():
        cs.sort(key=lambda c: (c.tier, STATUS_RANK[c.status], -(c.confidence or 0), c.bib))
        first = cs[0]
        out.append(DBSearchGroup(first.event_id, first.event_name, photo_id, first.file_path,
                                 resolve_stored_path(first.file_path, first.file_status),
                                 numeric, tuple(cs)))
    # best match tier first (exact, numeric, normalized, alias), confirmed before low confidence
    return sorted(out, key=lambda g: (g.tier, STATUS_RANK[g.status], g.event_name.lower(),
                                      g.photo_identity.lower(), g.numeric_identifier))


def search(conn: sqlite3.Connection, query: str, event_id: int | None = None,
           min_length: int | None = None, max_length: int | None = None,
           excluded_numeric: set[str] | None = None, include_rejected: bool = False,
           include_noise: bool = False) -> tuple[list[DBSearchGroup], list[BibCandidate]]:
    """
    (groups, filtered). Groups whose candidates are all rejected are left out
    unless include_rejected; such groups still list their rejected candidates
    so they can be restored.
    Candidates with filter status "noise" are left out of the groups and returned in `filtered`
    (noise_filtered=True, with their reasons) unless include_noise; "uncertain" candidates stay in
    the groups (low confidence, ranked after the confirmed ones). Nothing is deleted from the index.
    """
    kept, filtered = filter_candidates(find_candidates(conn, query, event_id),
                                       min_length, max_length, excluded_numeric)
    if not include_noise:
        filtered = filtered + [c for c in kept if c.noise_filtered]
        kept = [c for c in kept if not c.noise_filtered]
    groups = group_candidates(kept)
    if not include_rejected:
        groups = [g for g in groups if g.visible]
    return groups, filtered
