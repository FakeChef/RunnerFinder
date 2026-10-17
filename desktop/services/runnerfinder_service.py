"""
Phase 7.2: events, search, thumbnails and reject/restore from the real
RunnerFinder V0.3.1 index. GPU check and analysis are still the demo
implementation (Phase 7.3).

Engine functions used, all UNCHANGED:
    config.get_config()                 database path, thumbnail cache, thumbnail settings
    database.database.init_db()         open (or upgrade) the existing schema-3 index
    database.database.connect()         one connection per background task
    database.database.list_events()     event list
    search.db_search.search()           exact numeric search (noise filter, rejects, groups)
    utils.thumbnails.ensure_thumbnail() existing thumbnail, else made once (originals only read)
    database.database.reject_bib() / restore_bib()   the same review logic as the Search page

No OCR, detection or analysis module is imported. All database work runs on
one background thread (one task at a time), so the window never waits and
review writes never overlap with searches.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QObject

from desktop.services.background import BackgroundRunner
from desktop.services.demo_service import DemoEngineService
from desktop.services.models import EventSummary, SearchResult

# Dashboard: read-only counts on the existing schema (no change to the index)
_SUMMARY_SQL = """
SELECT e.id, e.name,
       COUNT(p.id) AS photos,
       COALESCE(SUM(p.analysis_status = 'done'), 0) AS analysed,
       COALESCE(SUM(p.analysis_status = 'error'), 0) AS failed,
       COALESCE(SUM(p.analysis_status = 'pending' AND p.file_status = 'ok'), 0) AS pending,
       COALESCE(SUM(p.file_status = 'missing'), 0) AS missing,
       (SELECT MAX(COALESCE(r.finished_time, r.started_time)) FROM analysis_runs r WHERE r.event_id = e.id) AS last
FROM events e LEFT JOIN photos p ON p.event_id = e.id
GROUP BY e.id ORDER BY last IS NULL, last DESC, e.name
"""
_NUMBERS_SQL = """
SELECT p.event_id, COUNT(DISTINCT d.numeric_bib) FROM bib_detection d JOIN photos p ON p.id = d.photo_id
WHERE d.numeric_bib IS NOT NULL AND COALESCE(json_extract(d.evidence_json, '$.filter.accepted'), 1) = 1
GROUP BY p.event_id
"""


class RunnerFinderSearchService(DemoEngineService):
    """Real search/review on the V0.3.1 index; GPU and analysis still simulated."""

    def __init__(self, parent: QObject | None = None, db_path: str | os.PathLike | None = None,
                 thumbnail_cache: str | os.PathLike | None = None):
        super().__init__(parent)
        import config                                   # first engine import (environment set-up)
        from database import database as db
        cfg = config.get_config()
        self.db_path = Path(db_path or cfg.paths.database_file)
        self.thumbnail_cache = Path(thumbnail_cache or cfg.paths.thumbnail_cache)
        self.thumbnail_config = cfg.thumbnails
        db.init_db(self.db_path, video_extension=False).close()   # existing schema 3: nothing changes (V1.3: the
                                                                  # worker adds the video tables when it analyses)
        self.mode_description = (f"Search: RunnerFinder index {self.db_path}  ·  "
                                 "Analysis: demo (simulated, engine not connected yet)")
        self._db = BackgroundRunner(self, threads=1)    # one database task at a time
        self._event_ids: dict[str, int] = {}

    @property
    def search_connected(self) -> bool:
        return True

    # ------------------------------------------------------------------ plumbing
    def _submit(self, fn, on_done) -> None:
        """fn(connection) on the database thread with its own connection; on_done(result) in the GUI."""
        db_path = self.db_path

        def task():
            from database import database as db
            conn = db.connect(db_path)
            try:
                return fn(conn)
            finally:
                conn.close()
        self._db.run(task, on_done, self.error.emit)

    def wait_idle(self, msecs: int = 30000) -> bool:
        """For tests and shutdown: wait until queued database tasks are finished."""
        return self._db.wait_idle(msecs)

    # ------------------------------------------------------------------ events
    def list_events(self) -> None:
        from database import database as db

        def work(conn):
            return [(r["id"], r["name"]) for r in db.list_events(conn)]

        def done(rows):
            self._event_ids = {name: eid for eid, name in rows}
            self.events_changed.emit([name for _, name in rows])
        self._submit(work, done)

    def list_event_summaries(self) -> None:
        def work(conn):
            numbers = dict(conn.execute(_NUMBERS_SQL).fetchall())
            folders = {}
            for eid, folder in conn.execute("SELECT DISTINCT event_id, source_folder FROM analysis_runs "
                                            "WHERE source_folder IS NOT NULL ORDER BY id"):
                folders.setdefault(eid, []).append(folder)
            from search.video_search import video_counts
            vids = video_counts(conn)                    # V1.3 (empty for an index without video tables)
            out = []
            for r in conn.execute(_SUMMARY_SQL):
                v = vids.get(r["id"], {})
                out.append(EventSummary(r["name"], r["photos"], r["analysed"], r["failed"], r["pending"], r["missing"],
                                        numbers.get(r["id"], 0), (r["last"] or "")[:16],
                                        tuple(dict.fromkeys(folders.get(r["id"], []))),
                                        v.get("videos", 0), v.get("verified", 0), v.get("ambiguous", 0),
                                        v.get("unindexed", 0), v.get("pending", 0), v.get("failed", 0)))
            return out
        self._submit(work, self.event_summaries_changed.emit)

    # ------------------------------------------------------------------ search
    def search(self, event: str, bib: str, include_rejected: bool = False,
               include_noise: bool = False, include_ambiguous: bool = False) -> None:
        from search.db_search import search as db_search
        from search.video_search import search_videos
        from utils.thumbnails import ensure_thumbnail, existing_thumbnail
        event_id = self._event_ids.get(event)
        cache, tcfg = self.thumbnail_cache, self.thumbnail_config

        def work(conn):
            groups, filtered = db_search(conn, bib, event_id, include_rejected=include_rejected,
                                         include_noise=include_noise)
            hidden = sum(1 for c in filtered if c.noise_filtered)
            results = []
            for g in groups:
                fp = conn.execute("SELECT fingerprint FROM photos WHERE id = ?", (g.photo_id,)).fetchone()
                thumb = None
                if g.resolved.path is not None:
                    t = ensure_thumbnail(g.resolved.path, cache, fp[0] if fp else None, tcfg)
                    thumb = str(t) if t else None
                shown = g.candidates if include_rejected else g.visible
                reasons = sorted({c.filter_reasons or "low_score" for c in shown if c.noise_filtered})
                # V1.3.1: uncertain candidates are shown (low confidence) unless a confirmed one is there
                low = ("" if any(c.status == "accepted" for c in shown if not c.noise_filtered) else
                       "; ".join(sorted({c.filter_reasons or "low_score" for c in shown if c.low_confidence})))
                results.append(SearchResult(
                    event=g.event_name, bib=g.numeric_identifier, file_path=g.photo_identity,
                    thumbnail_path=thumb, photo_id=g.photo_id,
                    bibs=tuple(dict.fromkeys(c.bib for c in shown)),
                    active_bibs=tuple(dict.fromkeys(
                        c.bib for c in g.candidates
                        if not c.rejected and c.verification not in ("corrected", "added"))),
                    rejected_bibs=tuple(dict.fromkeys(c.bib for c in g.candidates if c.rejected)),
                    filter_reason="; ".join(reasons),
                    missing=g.resolved.status == "missing",
                    match={1: "exact", 2: "numeric", 3: "normalized", 4: "alias"}[g.tier],
                    low_confidence=low,
                    matched_text=next((c.matched_text for c in g.candidates if c.matched_text), "")))
            # V1.3: videos of the same event, after the photos (photo search above is unchanged)
            for m in search_videos(conn, bib, event_id, include_ambiguous=include_ambiguous,
                                   include_rejected=include_rejected):
                thumb = existing_thumbnail(cache, m.thumbnail_fingerprint)   # made by the worker
                results.append(SearchResult(
                    event=m.event_name, bib=m.numeric_bib, file_path=m.file_path,
                    thumbnail_path=str(thumb) if thumb else None, photo_id=None, bibs=(m.bib,),
                    active_bibs=() if m.rejected or not m.confirmed else (m.numeric_bib,),
                    rejected_bibs=(m.numeric_bib,) if m.rejected else (),
                    missing=m.missing, match=m.match, media_type="VIDEO", video_id=m.video_id,
                    recognition_status=m.recognition_status, votes=m.votes, frames=m.frames,
                    confirmed=m.confirmed))
            return results, hidden

        def done(payload):
            results, hidden = payload
            self.search_hidden.emit(hidden)
            self.search_finished.emit(event, bib, results)
        self._submit(work, done)

    # ------------------------------------------------------------------ review
    def reject(self, result: SearchResult) -> None:
        from database import database as db
        if result.is_video:
            return self._review_video(result, reject=True)
        if result.photo_id is None or not result.active_bibs:
            self.error.emit("Nothing to reject on this photo.")
            return

        def work(conn):
            for bib in result.active_bibs:
                db.reject_bib(conn, result.photo_id, bib)
            return f"Rejected {', '.join(result.active_bibs)} on {result.file_name}"
        self._submit(work, self.review_finished.emit)

    def _review_video(self, result: SearchResult, reject: bool) -> None:
        """V1.3: reject / restore a video bib (video_feedback; the recognition result is not changed)."""
        from database import video_store as vs
        bibs = result.active_bibs if reject else result.rejected_bibs
        if result.video_id is None or not bibs:
            self.error.emit(f"Nothing to {'reject' if reject else 'restore'} on this video.")
            return

        def work(conn):
            for b in bibs:
                (vs.reject_video_bib if reject else vs.restore_video_bib)(conn, result.video_id, b)
            return f"{'Rejected' if reject else 'Restored'} {', '.join(bibs)} on video {result.file_name}"
        self._submit(work, self.review_finished.emit)

    def restore(self, result: SearchResult) -> None:
        from database import database as db
        if result.is_video:
            return self._review_video(result, reject=False)
        if result.photo_id is None or not result.rejected_bibs:
            self.error.emit("Nothing to restore on this photo.")
            return

        def work(conn):
            for bib in result.rejected_bibs:
                db.restore_bib(conn, result.photo_id, bib)
            return f"Restored {', '.join(result.rejected_bibs)} on {result.file_name}"
        self._submit(work, self.review_finished.emit)
