"""V1.3 GUI: unified photo + video search, PHOTO/VIDEO labels, ambiguous handling, video reject/restore, dashboard
video counts. Real search service on a temporary index (no OCR, no video decoding)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from database import database as db  # noqa: E402
from database import video_store as vs  # noqa: E402
from desktop.app import build  # noqa: E402
from desktop.main_window import SEARCH  # noqa: E402
from desktop.services.runnerfinder_service import RunnerFinderSearchService  # noqa: E402
from utils.thumbnails import save_from_image  # noqa: E402

APP = QApplication.instance() or QApplication(sys.argv)
FP_VIDEO = "c0ffee" * 10 + "abcd"


def pump_until(cond, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class UnifiedSearchGuiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.db, self.thumbs = t / "index.db", t / "thumbs"
        photo = t / "IMG_1.jpg"
        Image.new("RGB", (64, 48), (200, 50, 50)).save(photo)
        video = t / "CLIP_1.MP4"
        video.write_bytes(b"not decoded by the GUI")
        conn = db.init_db(self.db)                      # worker-style init: video tables present
        ev = db.get_or_create_event(conn, "race")
        pid = db.insert_photo(conn, ev, {"file_path": str(photo), "path_key": str(photo).lower(), "file_name": "IMG_1.jpg",
                                         "fingerprint": "ab" * 32})
        db.save_photo_result(conn, pid, None, [{"bib_number": "B60914", "numeric_bib": "60914", "confidence": 0.99,
                                                "source": "person_body"}])
        vid = vs.insert_video(conn, ev, {"file_path": str(video), "path_key": str(video).lower(),
                                         "file_name": "CLIP_1.MP4", "fingerprint": FP_VIDEO})
        vs.save_video_result(conn, vid, None, {"status": "VERIFIED", "strategy": "adaptive_s5_s7/1", "frames_used": 5,
                                               "escalated": False, "sample_times": [], "evidence": {},
                                               "thumbnail_fingerprint": FP_VIDEO,
                                               "candidates": [{"bib_number": "A60914", "numeric_bib": "60914",
                                                               "role": "accepted", "votes": 4, "frames": 5}]})
        (t / "CLIP_2.MP4").write_bytes(b"x")
        amb = vs.insert_video(conn, ev, {"file_path": str(t / "CLIP_2.MP4"), "path_key": str(t / "clip_2.mp4"),
                                         "file_name": "CLIP_2.MP4"})
        vs.save_video_result(conn, amb, None, {"status": "AMBIGUOUS", "strategy": "adaptive_s5_s7/1", "frames_used": 7,
                                               "escalated": True, "sample_times": [], "evidence": {},
                                               "candidates": [{"bib_number": "60914", "numeric_bib": "60914",
                                                               "role": "candidate", "votes": 3, "frames": 7},
                                                              {"bib_number": "15293", "numeric_bib": "15293",
                                                               "role": "candidate", "votes": 2, "frames": 7}]})
        conn.close()
        save_from_image(Image.new("RGB", (90, 160), (10, 120, 10)), self.thumbs, FP_VIDEO)
        self.app_parts = build(APP, lambda parent: RunnerFinderSearchService(parent, db_path=self.db,
                                                                             thumbnail_cache=self.thumbs))
        _, self.window, self.controller = self.app_parts
        self.service = self.controller.service
        self.page = self.window.search_page
        self.controller.start()
        pump_until(lambda: self.page.event_combo.count() > 0)

    def tearDown(self):
        self.service.wait_idle()
        self.window.close()
        self.window.deleteLater()
        APP.processEvents()
        self.tmp.cleanup()

    def search(self, bib):
        self.window.show_page(SEARCH)
        self.page.select_event("race")
        self.page.bib_edit.setText(bib)
        self.page.label_before = self.page.result_label.text()
        self.page._search()
        self.assertTrue(pump_until(lambda: self.page.result_label.text().startswith(("Results", "No photos"))))
        self.service.wait_idle()
        APP.processEvents()
        return [self.page.model.item(i).data(Qt_RESULT) for i in range(self.page.model.rowCount())]

    def test_photo_and_video_in_one_result_list(self):
        results = self.search("60914")
        self.assertEqual([(r.media_type, r.file_name) for r in results], [("PHOTO", "IMG_1.jpg"), ("VIDEO", "CLIP_1.MP4")])
        self.assertIn("1 photo, 1 video", self.page.result_label.text())
        texts = [self.page.model.item(i).text().splitlines() for i in range(self.page.model.rowCount())]
        self.assertEqual(texts[0][0], "IMG_1.jpg")
        self.assertTrue(texts[0][1].startswith("PHOTO · B60914"))
        self.assertTrue(texts[1][1].startswith("▶ VIDEO · A60914"))
        v = results[1]
        self.assertTrue(v.confirmed and v.thumbnail_path and Path(v.thumbnail_path).is_file())
        self.page.results.setCurrentIndex(self.page.model.index(1, 0))
        APP.processEvents()
        self.assertEqual(self.page.open_button.text().strip(), "Open Video")
        self.assertTrue(self.page.detail_type.text().startswith("VIDEO"))
        self.assertIn("4 of 5", self.page.detail_status.text())
        self.page.results.setCurrentIndex(self.page.model.index(0, 0))
        APP.processEvents()
        self.assertEqual(self.page.open_button.text().strip(), "Open Photo")
        self.assertEqual(self.page.detail_type.text(), "PHOTO")

    def test_ambiguous_videos_only_on_request_and_unconfirmed(self):
        self.assertEqual([r.file_name for r in self.search("15293")], [])
        self.page.show_ambiguous_check.setChecked(True)
        self.assertTrue(pump_until(lambda: self.page.model.rowCount() == 1))
        r = self.page.model.item(0).data(Qt_RESULT)
        self.assertEqual((r.file_name, r.confirmed, r.active_bibs), ("CLIP_2.MP4", False, ()))
        self.assertIn("ambiguous", self.page.model.item(0).text())
        self.page.results.setCurrentIndex(self.page.model.index(0, 0))
        APP.processEvents()
        self.assertFalse(self.page.reject_button.isEnabled())       # nothing confirmed to reject
        self.assertIn("AMBIGUOUS", self.page.detail_status.text())

    def test_reject_and_restore_a_video_bib(self):
        self.search("60914")
        self.page.results.setCurrentIndex(self.page.model.index(1, 0))
        APP.processEvents()
        self.assertTrue(self.page.reject_button.isEnabled())
        self.page.reject_button.click()
        self.assertTrue(pump_until(lambda: self.page.model.rowCount() == 1))
        self.assertEqual(self.page.model.item(0).data(Qt_RESULT).media_type, "PHOTO")
        self.page.show_rejected_check.setChecked(True)
        self.assertTrue(pump_until(lambda: self.page.model.rowCount() == 2))
        self.page.results.setCurrentIndex(self.page.model.index(1, 0))
        APP.processEvents()
        self.assertTrue(self.page.restore_button.isEnabled())
        self.page.restore_button.click()
        self.page.show_rejected_check.setChecked(False)
        self.assertTrue(pump_until(lambda: self.page.model.rowCount() == 2
                                   and self.page.model.item(1).data(Qt_RESULT).active_bibs == ("60914",)))

    def test_dashboard_counts_videos(self):
        dash = self.window.dashboard_page
        self.assertTrue(pump_until(lambda: dash._summaries and dash._summaries[0].videos == 2))
        s = dash._summaries[0]
        self.assertEqual((s.photos, s.videos, s.videos_verified, s.videos_ambiguous), (1, 2, 1, 1))

    def test_gui_does_not_migrate_an_index_without_video_tables(self):
        import hashlib
        import sqlite3
        p = Path(self.tmp.name) / "v12.db"
        c = sqlite3.connect(p)
        c.executescript(db.SCHEMA_FILE.read_text(encoding="utf-8"))       # the 1.2 schema: no video tables
        c.execute("PRAGMA user_version = 3")
        c.commit()
        c.execute("PRAGMA journal_mode = WAL")          # as every index opened by RunnerFinder (1.2 included)
        c.close()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        svc = RunnerFinderSearchService(db_path=p, thumbnail_cache=self.thumbs)
        svc.search("", "60914")
        svc.wait_idle()
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), h)
        self.assertFalse((Path(self.tmp.name) / "backups").exists())


from desktop.pages.search_page import RESULT_ROLE as Qt_RESULT  # noqa: E402

if __name__ == "__main__":
    unittest.main()
