"""Phase 7.2 integration tests: real V0.3.1 index (a COPY of v02\\data\\runner_photo_finder.db).

The original index and the thumbnail cache of v02 are never written: the tests use a
copied database and a temporary thumbnail folder. No OCR runs.
Run from the repository folder:
    <python with PySide6 + Pillow> -m unittest desktop.tests.test_search_integration -v
"""
import gc
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop import os_actions  # noqa: E402
from desktop.app import build  # noqa: E402
from desktop.main_window import SEARCH  # noqa: E402
from desktop.services.runnerfinder_service import RunnerFinderSearchService  # noqa: E402
from desktop.tests.private_data import BLIND_LABELS, INDEX_DB, THUMBNAILS, available, needs  # noqa: E402

SOURCE_DB = INDEX_DB
EVENT = "0927 Race"
NO_OCR = {"paddle", "paddleocr", "paddlex", "ultralytics", "torch", "cv2"}
NO_ENGINE_ANALYSIS = {"ocr.photo_ocr", "ocr.bib_reader", "ocr.diagnostic", "detection", "indexing.analyze"}
APP = QApplication.instance() or QApplication(sys.argv)


def dispose(window):
    """Destroy a test window NOW, in the GUI thread. Left to Python's garbage collector, its Qt
    objects can be deleted from a background thread (crash seen in Phase 7.3A)."""
    window.close()
    window.deleteLater()
    APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    APP.processEvents()
    gc.collect()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Collector:
    """Collects service signals and waits for them."""

    def __init__(self, service):
        self.events, self.results, self.hidden, self.reviews, self.errors = [], [], [], [], []
        service.events_changed.connect(self.events.append)
        service.search_finished.connect(lambda e, b, r: self.results.append((e, b, r)))
        service.search_hidden.connect(self.hidden.append)
        service.review_finished.connect(self.reviews.append)
        service.error.connect(self.errors.append)

    def wait(self, attr, n=1, timeout=60):
        end = time.monotonic() + timeout
        while time.monotonic() < end and len(getattr(self, attr)) < n:
            APP.processEvents()
            time.sleep(0.005)
        if len(getattr(self, attr)) < n:
            raise AssertionError(f"timeout waiting for {attr}; errors: {self.errors}")
        return getattr(self, attr)[n - 1]


@needs(SOURCE_DB, "RUNNERFINDER_TEST_INDEX")
class SearchIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_hash = sha(SOURCE_DB)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "index.db"
        shutil.copy2(SOURCE_DB, cls.db)
        cls.thumbs = Path(cls.tmp.name) / "thumbnails"
        # start from a COPY of the existing V0.3.1 cache (making ~1000 thumbnails from originals is slow);
        # test_thumbnails_are_loaded_or_created deletes some copies to check that they are made again
        if THUMBNAILS.is_dir():
            shutil.copytree(THUMBNAILS, cls.thumbs)
        cls.service = RunnerFinderSearchService(db_path=cls.db, thumbnail_cache=cls.thumbs)
        cls.c = Collector(cls.service)
        cls.service.list_events()
        cls.c.wait("events")

    @classmethod
    def tearDownClass(cls):
        cls.service.wait_idle()
        cls.tmp.cleanup()

    def search(self, bib, **kw):
        n = len(self.c.results) + 1
        self.service.search(EVENT, bib, **kw)
        return self.c.wait("results", n)[2]

    def reference(self, bib, **kw):
        """The same search directly with the engine (search/db_search.py)."""
        from search.db_search import search as db_search
        conn = sqlite3.connect(f"file:{self.db.as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        ev = conn.execute("SELECT id FROM events WHERE name=?", (EVENT,)).fetchone()[0]
        groups, _ = db_search(conn, bib, ev, **kw)
        out = {(g.photo_id, tuple(sorted(dict.fromkeys(c.bib for c in (g.candidates if kw.get('include_rejected')
                                                                            else g.visible))))) for g in groups}
        conn.close()
        return out

    # ------------------------------------------------------------------ events
    def test_event_list_loads_from_sqlite(self):
        conn = sqlite3.connect(f"file:{self.db.as_posix()}?mode=ro", uri=True)
        names = {r[0] for r in conn.execute("SELECT name FROM events")}
        conn.close()
        self.assertEqual(set(self.c.events[0]), names)
        self.assertIn(EVENT, self.c.events[0])

    # ------------------------------------------------------------------ search parity
    def test_search_matches_db_search_for_many_bibs(self):
        import csv
        bibs = set()
        if available(BLIND_LABELS):                        # the 40 blind-label bibs of the 0927 validation
            with open(BLIND_LABELS, encoding="utf-8") as f:
                bibs = {r["bib"] for r in csv.DictReader(f)}
        bibs |= {"67657", "68905", "30022", "62558", "18866", "69932", "13655", "2026", "123"}
        for bib in sorted(bibs):
            for kw in ({}, {"include_noise": True}, {"include_rejected": True}):
                got = {(r.photo_id, tuple(sorted(r.bibs))) for r in self.search(bib, **kw)}
                self.assertEqual(got, self.reference(bib, **kw), (bib, kw))

    def test_exact_numeric_and_card_fields(self):
        res = self.search("67657")
        self.assertEqual(len(res), 43)
        r = res[0]
        self.assertEqual((r.event, r.bib), (EVENT, "67657"))
        self.assertTrue(r.file_name.upper().endswith(".JPG") and r.photo_id is not None)
        self.assertTrue(r.bibs and r.active_bibs and not r.rejected_bibs and not r.filtered)
        # no partial matching; V1.3.1: "123" finds "0123"/"00123" only as labelled leading-zero aliases
        self.assertTrue(all(x.match == "alias" for x in self.search("123")))

    def test_filtered_candidate_is_hidden_and_shown_with_reason(self):
        self.assertEqual(self.search("13655"), [])
        self.assertEqual(self.c.hidden[-1], 1)
        shown = self.search("13655", include_noise=True)
        self.assertEqual(len(shown), 1)
        self.assertTrue(shown[0].filtered and "likely_storage_number" in shown[0].filter_reason)

    # ------------------------------------------------------------------ thumbnails
    def test_thumbnails_are_loaded_or_created(self):
        first = self.search("60914")
        self.assertTrue(first)
        for r in first:                                   # loaded from the cache ...
            self.assertTrue(r.thumbnail_path and Path(r.thumbnail_path).is_file())
            Path(r.thumbnail_path).unlink()               # ... and made again from the original
        res = self.search("60914")
        self.assertEqual(len(res), len(first))
        for r in res:
            self.assertIsNotNone(r.thumbnail_path)
            p = Path(r.thumbnail_path)
            self.assertTrue(p.is_file() and p.resolve().is_relative_to(self.thumbs.resolve()))
            img = QImage(str(p))
            self.assertFalse(img.isNull())
            self.assertEqual(max(img.width(), img.height()), 512)

    # ------------------------------------------------------------------ review
    def test_reject_and_restore_through_sqlite_feedback(self):
        res = self.search("62558")
        target = next(r for r in res if r.active_bibs)
        before = len(res)
        self.service.reject(target)
        self.assertIn("Rejected", self.c.wait("reviews", len(self.c.reviews) + 1))
        after = self.search("62558")
        self.assertEqual(len(after), before - 1)
        self.assertNotIn(target.photo_id, [r.photo_id for r in after])
        shown = next(r for r in self.search("62558", include_rejected=True) if r.photo_id == target.photo_id)
        self.assertTrue(shown.rejected_bibs and not shown.active_bibs)
        self.service.restore(shown)
        self.assertIn("Restored", self.c.wait("reviews", len(self.c.reviews) + 1))
        self.assertEqual(len(self.search("62558")), before)
        conn = sqlite3.connect(f"file:{self.db.as_posix()}?mode=ro", uri=True)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM detection_feedback WHERE photo_id=?",
                                      (target.photo_id,)).fetchone()[0], 0)
        conn.close()

    # ------------------------------------------------------------------ GUI with the real service
    def test_gui_search_page_with_real_service(self):
        _, window, controller = build(APP, lambda parent: RunnerFinderSearchService(
            parent, db_path=self.db, thumbnail_cache=self.thumbs))
        page = window.search_page
        controller.start()
        end = time.monotonic() + 30
        while time.monotonic() < end and page.event_combo.findText(EVENT) < 0:
            APP.processEvents()
        page.select_event(EVENT)
        window.show_page(SEARCH)
        page.bib_edit.setText("69932")
        page.search_button.click()
        end = time.monotonic() + 30
        while time.monotonic() < end and not page.result_label.text().startswith(("Results", "No photos")):
            APP.processEvents()
        self.assertTrue(page.result_label.text().startswith("Results"), page.result_label.text())
        self.assertEqual(page.model.rowCount(), len(self.reference("69932")))
        page.results.setCurrentIndex(page.model.index(0, 0))
        self.assertTrue(page.reject_button.isEnabled() and not page.restore_button.isEnabled())
        opened = []
        orig = os_actions.open_photo
        os_actions.open_photo = lambda p: opened.append(p) or True
        try:
            page.open_button.click()
        finally:
            os_actions.open_photo = orig
        self.assertTrue(opened and os.path.isfile(opened[0]))
        dispose(window)

    # ------------------------------------------------------------------ safety
    def test_zz_no_ocr_and_source_index_unchanged(self):
        loaded = set(sys.modules)
        self.assertFalse({m.split(".")[0] for m in loaded} & NO_OCR)
        self.assertFalse({m for m in loaded if m in NO_ENGINE_ANALYSIS or m.startswith("detection")})
        self.service.wait_idle()
        self.assertEqual(sha(SOURCE_DB), self.source_hash)              # v02 index never written


if __name__ == "__main__":
    unittest.main()
