"""
V1.3 video bib search (standard library only: runs in the GUI process).

Same matching rules as the photo search (search/db_search.py), applied to video vote keys:
  * V1.3.1 queries as in the photo search (db_search.parse_query): digits ("15293") or a full bib
    ("A15293", "a-15293"); tier "exact" when the full bib equals the query, "numeric" when only its
    digit run does ("A15293" for 15293; for a full-bib query: a key read without letters), "alias" when
    only leading zeros differ ("175" <-> "0175"); a full-bib query never returns other letters
  * no substring, partial or fuzzy matching
  * only ACCEPTED bibs of VERIFIED videos are matches by default
  * AMBIGUOUS videos: their candidate keys with >= 2 votes are returned only with include_ambiguous,
    flagged unconfirmed (never a confirmed match)
  * rejected video bibs (video_feedback 'incorrect') are hidden unless include_rejected
Videos of an index without the V1.3 video tables simply give no results.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

AMBIGUOUS_MIN_VOTES = 2


@dataclass(frozen=True)
class VideoMatch:
    video_id: int
    event_id: int
    event_name: str
    file_path: str
    file_status: str
    missing: bool
    numeric_bib: str
    bib: str                     # best full form, e.g. "A15293"
    role: str                    # accepted | candidate
    votes: int
    frames: int
    recognition_status: str      # VERIFIED | AMBIGUOUS
    rejected: bool
    thumbnail_fingerprint: str | None
    duration_s: float | None
    match: str                   # exact | numeric | alias

    @property
    def confirmed(self) -> bool:
        return self.role == "accepted" and self.recognition_status == "VERIFIED"


def has_video_tables(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'video_candidates'"
                        ).fetchone() is not None


_SQL = """
SELECT v.id, e.id, e.name, v.file_path, v.file_status, c.numeric_bib, c.bib_number, c.role, c.votes, c.frames,
       v.recognition_status, (f.id IS NOT NULL) AS rejected, v.thumbnail_fingerprint, v.duration_s
FROM video_candidates c
JOIN videos v ON v.id = c.video_id
JOIN events e ON e.id = v.event_id
LEFT JOIN video_feedback f ON f.video_id = c.video_id AND f.numeric_bib = c.numeric_bib AND f.verdict = 'incorrect'
WHERE c.numeric_bib IN (SELECT value FROM json_each(:n))
  AND v.analysis_status = 'done'
  AND (:event_id IS NULL OR v.event_id = :event_id)
  AND ((c.role = 'accepted' AND v.recognition_status = 'VERIFIED')
       OR (:ambiguous AND v.recognition_status = 'AMBIGUOUS' AND c.votes >= :min_votes))
ORDER BY (c.role = 'accepted') DESC, e.name, v.file_path
"""


def search_videos(conn: sqlite3.Connection, query: str, event_id: int | None = None, *,
                  include_ambiguous: bool = False, include_rejected: bool = False) -> list[VideoMatch]:
    from search.db_search import parse_query
    q = parse_query(query)
    if q is None or not has_video_tables(conn):
        return []
    out = []
    for r in conn.execute(_SQL, {"n": json.dumps([q.core, *q.aliases()]), "event_id": event_id,
                                 "ambiguous": int(include_ambiguous), "min_votes": AMBIGUOUS_MIN_VOTES}):
        if r[11] and not include_rejected:
            continue
        bib, numeric = r[6] or r[5], r[5]
        if bib == q.full:
            match = "exact"
        elif not q.digits_only and not q.letters_match(bib):
            continue                              # other letters: a different bib
        else:
            match = "numeric" if numeric == q.core else "alias"
        missing = not (r[3] and os.path.isfile(r[3]))
        out.append(VideoMatch(r[0], r[1], r[2], r[3], r[4], missing, r[5], r[6], r[7], r[8], r[9], r[10],
                              bool(r[11]), r[12], r[13], match))
    tier = {"exact": 1, "numeric": 2, "alias": 3}
    return sorted(out, key=lambda m: (not m.confirmed, tier[m.match], m.event_name.lower(), m.file_path.lower()))


def video_counts(conn: sqlite3.Connection) -> dict[int, dict]:
    """Dashboard: per event {videos, analysed, verified, ambiguous, unindexed, failed, missing, pending}."""
    if not has_video_tables(conn):
        return {}
    out = {}
    for r in conn.execute("""
        SELECT event_id, COUNT(*),
               COALESCE(SUM(analysis_status = 'done'), 0),
               COALESCE(SUM(recognition_status = 'VERIFIED' AND analysis_status = 'done'), 0),
               COALESCE(SUM(recognition_status = 'AMBIGUOUS' AND analysis_status = 'done'), 0),
               COALESCE(SUM(recognition_status = 'UNINDEXED' AND analysis_status = 'done'), 0),
               COALESCE(SUM(analysis_status = 'error' OR file_status = 'unreadable'), 0),
               COALESCE(SUM(file_status = 'missing'), 0),
               COALESCE(SUM(analysis_status = 'pending' AND file_status = 'ok'), 0)
        FROM videos GROUP BY event_id"""):
        out[r[0]] = dict(zip(("videos", "analysed", "verified", "ambiguous", "unindexed", "failed", "missing",
                              "pending"), r[1:]))
    return out
