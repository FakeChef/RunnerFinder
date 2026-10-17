"""
Search: bib number -> photos. Implemented in Phase 6/7.

Searches the v_photo_bibs view (AI results with manual corrections
applied), and returns:
  - exact matches, ordered by capture time
  - "possible matches" (one character different, or partial reads such
    as "1025" for "10258") shown separately for visual checking
"""
from __future__ import annotations

import sqlite3


def search_bib(conn: sqlite3.Connection, event_id: int, query: str) -> dict:
    raise NotImplementedError("Phase 6")
