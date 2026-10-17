"""V1.2 bib search normalization (search layer only) on a synthetic index built with the engine's own
database functions. Stored values are written the way the OCR stage stores them; nothing is changed.
Run from the repository folder:  <python with PySide6 + Pillow> -m unittest desktop.tests.test_v12_search -v
"""
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from database import database as db  # noqa: E402
from search.db_search import normalized_number, search  # noqa: E402
# results_store / ocr.numeric_candidates are imported inside build_index(): test discovery imports every
# test module first, and other suites check that the GUI process never loads the ocr package.

EVENT = "Search V1.2"
# (photo, OCR text, stored bib as the OCR stage stores it, digit fix, filter accepted, filter reasons)
ROWS = [
    ("exact.jpg",       "60914",    "60914",  0, True,  []),
    ("prefix.jpg",      "B60914",   "B60914", 0, True,  []),
    ("hyphen.jpg",      "B-60914",  "B60914", 0, True,  []),                      # separators removed
    ("space.jpg",       "B 60914",  "60914",  0, True,  []),                      # token "60914"
    ("suffix.jpg",      "60914B",   "609148", 1, False, ["low_score"]),           # B -> 8 digit fix
    ("lower.jpg",       "60914b",   "609148", 1, False, ["low_score"]),
    ("short.jpg",       "6091",     "6091",   0, False, ["low_score"]),
    ("long.jpg",        "609140",   "609140", 0, False, ["low_score"]),
    ("lead.jpg",        "160914",   "160914", 0, False, ["low_score"]),
    ("bagcheck.jpg",    "6-0914s",  "609145", 1, False, ["likely_storage_number"]),
    ("distance.jpg",    "60914KM",  "60914KM", 0, False, ["excluded_text"]),       # stored as-is (2 letters)
    ("split.jpg",       "609 14",   "60914",  0, True,  []),                      # joined by the OCR stage
]


def build_index(path: Path) -> None:
    from PIL import Image

    from database.results_store import get_or_create_photo
    from ocr.numeric_candidates import numeric_identifier   # how the OCR stage derives numeric_bib
    photos = path.parent / "photos"
    photos.mkdir()
    conn = db.init_db(path)
    ev = db.get_or_create_event(conn, EVENT)
    run = db.start_run(conn, ev, source="ocr", source_folder=str(photos), photos_total=len(ROWS))
    for name, text, bib, fix, accepted, reasons in ROWS:
        Image.new("RGB", (64, 48), (90, 120, 160)).save(photos / name, quality=80)   # small real photo
        pid = get_or_create_photo(conn, ev, str(photos / name))
        evidence = {"readings": [{"raw_text": text, "ocr_confidence": 0.98}],
                    "filter": {"accepted": accepted, "score": 7 if accepted else 2, "reasons": reasons}}
        db.save_photo_result(conn, pid, run, [{
            "bib_number": bib, "numeric_bib": numeric_identifier(bib), "confidence": 0.95, "ocr_text": text,
            "source": "person_body", "readings": 1, "region_ids": "C1", "letter_digit_fix": fix,
            "evidence_json": json.dumps(evidence)}])
    conn.close()


class NormalizationRuleTest(unittest.TestCase):
    def test_matches(self):
        for text in ("60914", "B60914", "B-60914", "B 60914", "60914B", "b60914", "AB-60914-C", "B.60914"):
            self.assertEqual(normalized_number(text), "60914", text)

    def test_never_fuzzy_or_partial(self):
        for text, expected in (("6091", "6091"), ("609140", "609140"), ("160914", "160914"),
                               ("6-0914S", None), ("609 14", None), ("60.914", None), ("ABCD60914", None),
                               ("B60914ZHANG", None), ("971A10KM", None), ("", None), (None, None)):
            self.assertEqual(normalized_number(text), expected, text)


class SearchTierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "index.db"
        build_index(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def conn(self):
        c = sqlite3.connect(str(self.db))
        c.row_factory = sqlite3.Row
        return c

    def names(self, groups):
        return [Path(g.photo_identity).name for g in groups]

    def test_60914_matches_all_ocr_variants_in_tier_order(self):
        c = self.conn()
        groups, filtered = search(c, "60914")
        c.close()
        names = self.names(groups)
        self.assertEqual(set(names), {"exact.jpg", "space.jpg", "split.jpg", "prefix.jpg", "hyphen.jpg",
                                      "suffix.jpg", "lower.jpg"})
        tiers = {Path(g.photo_identity).name: g.tier for g in groups}
        self.assertEqual(tiers["exact.jpg"], 1)
        self.assertEqual((tiers["prefix.jpg"], tiers["hyphen.jpg"]), (2, 2))
        self.assertEqual((tiers["suffix.jpg"], tiers["lower.jpg"]), (3, 3))
        self.assertEqual([g.tier for g in groups], sorted(g.tier for g in groups))   # priority order
        suffix = next(g for g in groups if g.photo_identity.endswith("suffix.jpg"))
        self.assertEqual((suffix.candidates[0].bib, suffix.candidates[0].matched_text), ("609148", "60914B"))
        self.assertEqual(suffix.numeric_identifier, "60914")
        # the distance text "60914KM" normalizes to 60914 but its row is noise: hidden with its reason
        self.assertEqual([(Path(f.file_path).name, f.filter_reasons) for f in filtered],
                         [("distance.jpg", "excluded_text")])

    def test_no_partial_or_longer_numbers(self):
        c = self.conn()
        for q in ("6091", "609140", "160914", "0914", "609145"):
            groups, _ = search(c, q, include_noise=True)
            names = self.names(groups)
            self.assertNotIn("suffix.jpg", names, q)
            self.assertNotIn("exact.jpg", names, q)
        self.assertEqual(self.names(search(c, "6091", include_noise=True)[0]), ["short.jpg"])
        self.assertEqual(self.names(search(c, "160914", include_noise=True)[0]), ["lead.jpg"])
        self.assertEqual(self.names(search(c, "609145", include_noise=True)[0]), ["bagcheck.jpg"])
        c.close()

    def test_bag_check_never_matches_by_normalization(self):
        c = self.conn()
        groups, filtered = search(c, "60914", include_noise=True)
        c.close()
        self.assertNotIn("bagcheck.jpg", self.names(groups))
        self.assertIn("distance.jpg", self.names(groups))                  # shown only with include_noise

    def test_reject_and_restore_a_normalized_match(self):
        c = self.conn()
        suffix = next(g for g in search(c, "60914")[0] if g.photo_identity.endswith("suffix.jpg"))
        db.reject_bib(c, suffix.photo_id, "609148")
        self.assertNotIn("suffix.jpg", self.names(search(c, "60914")[0]))
        again = next(g for g in search(c, "60914", include_rejected=True)[0] if g.photo_identity.endswith("suffix.jpg"))
        self.assertTrue(again.candidates[0].rejected)
        db.restore_bib(c, suffix.photo_id, "609148")
        self.assertIn("suffix.jpg", self.names(search(c, "60914")[0]))
        self.assertEqual(c.execute("SELECT COUNT(*) FROM detection_feedback").fetchone()[0], 0)
        c.close()

    def test_stored_values_unchanged(self):
        c = self.conn()
        before = c.execute("SELECT bib_number, numeric_bib, ocr_text, evidence_json FROM bib_detection ORDER BY id").fetchall()
        for q in ("60914", "6091", "609140"):
            search(c, q, include_noise=True, include_rejected=True)
        after = c.execute("SELECT bib_number, numeric_bib, ocr_text, evidence_json FROM bib_detection ORDER BY id").fetchall()
        self.assertEqual([tuple(r) for r in before], [tuple(r) for r in after])
        self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 3)
        c.close()

    def test_desktop_cards_show_the_normalized_reading(self):
        from PySide6.QtWidgets import QApplication
        from desktop.pages.search_page import RESULT_ROLE, SearchPage
        from desktop.services.runnerfinder_service import RunnerFinderSearchService
        app = QApplication.instance() or QApplication(sys.argv)
        service = RunnerFinderSearchService(db_path=self.db, thumbnail_cache=Path(self.tmp.name) / "thumbs")
        page = SearchPage()
        service.search_finished.connect(page.show_results)
        got = []
        service.search_finished.connect(lambda e, b, r: got.append(r))
        service.list_events()
        service.search(EVENT, "60914")
        end = time.monotonic() + 30
        while not got and time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)
        service.wait_idle()
        results = got[0]
        self.assertEqual([r.match for r in results][:1], ["exact"])
        suffix = next(r for r in results if r.file_name == "suffix.jpg")
        self.assertEqual((suffix.match, suffix.matched_text, suffix.active_bibs), ("normalized", "60914B", ("609148",)))
        row = next(i for i in range(page.model.rowCount()) if page.model.item(i).data(RESULT_ROLE).file_name == "suffix.jpg")
        self.assertIn("≈ OCR read 60914B", page.model.item(row).text())
        page.results.setCurrentIndex(page.model.index(row, 0))
        self.assertIn("Normalized match", page.detail_status.text())
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
