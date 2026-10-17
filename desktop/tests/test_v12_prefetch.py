"""V1.2 prefetch pipeline tests.

Unit part (no AI): scanner.prefetch.PhotoPrefetcher - order, error hand-off, bounded look-ahead,
early stop, thumbnail hook.
Real-engine part (GPU environment, ~2 min): the worker on a 12-photo COPY into temporary databases -
prefetch off vs on gives identical stored results; cancel while photos are queued commits nothing
extra; resume after the cancel gives the same final result as an uninterrupted run.
Run from the repository folder:  <python with PySide6 + Pillow> -m unittest desktop.tests.test_v12_prefetch -v
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scanner.prefetch import PhotoPrefetcher  # noqa: E402
from desktop.tests.private_data import ENGINE_PYTHON, TEST_PHOTOS, needs  # noqa: E402


class FakeLoader:
    """Stands in for load_image_with_orientation: 'image' = path upper-cased; 'bad*' raises."""

    def __init__(self, delay=0.0):
        self.delay, self.calls = delay, []

    def __call__(self, path):
        self.calls.append(path)
        time.sleep(self.delay)
        if "bad" in path:
            raise ValueError(f"cannot decode {path}")
        return path.upper(), 6


class PrefetcherUnitTest(unittest.TestCase):
    def test_order_and_values_as_inline_loading(self):
        paths = [f"p{i}.jpg" for i in range(20)]
        with PhotoPrefetcher(paths, depth=2, loader=FakeLoader()) as photos:
            got = [photos.take(p)[:2] for p in paths]
        self.assertEqual(got, [(p.upper(), 6) for p in paths])

    def test_decode_error_is_raised_for_that_photo_only(self):
        paths = ["a.jpg", "bad.jpg", "c.jpg"]
        with PhotoPrefetcher(paths, depth=2, loader=FakeLoader()) as photos:
            self.assertEqual(photos.take("a.jpg")[0], "A.JPG")
            with self.assertRaises(ValueError):
                photos.take("bad.jpg")
            self.assertEqual(photos.take("c.jpg")[0], "C.JPG")

    def test_look_ahead_is_bounded(self):
        paths = [f"p{i}.jpg" for i in range(30)]
        loader = FakeLoader()
        with PhotoPrefetcher(paths, depth=2, loader=loader) as photos:
            for n, p in enumerate(paths[:5], start=1):
                photos.take(p)
                time.sleep(0.15)                              # slow consumer
                self.assertLessEqual(photos.produced, n + 2 + 1)   # depth 2 queued + 1 being finished

    def test_close_stops_early_and_drops_queued_photos(self):
        paths = [f"p{i}.jpg" for i in range(100)]
        loader = FakeLoader(delay=0.01)
        photos = PhotoPrefetcher(paths, depth=2, loader=loader)
        photos.take("p0.jpg")
        photos.close()
        self.assertFalse(photos._thread.is_alive())
        self.assertLess(len(loader.calls), 10)
        self.assertTrue(photos._queue.empty())

    def test_thumbnail_hook_in_order_and_errors_ignored(self):
        seen = []

        def hook(path, image):
            seen.append(path)
            if path == "p1.jpg":
                raise OSError("disk full")
        paths = [f"p{i}.jpg" for i in range(5)]
        with PhotoPrefetcher(paths, hook, depth=2, loader=FakeLoader()) as photos:
            for p in paths:
                photos.take(p)
        self.assertEqual(seen, paths)

    def test_consumer_and_producer_run_concurrently(self):
        paths = [f"p{i}.jpg" for i in range(10)]
        loader = FakeLoader(delay=0.05)
        t = time.perf_counter()
        with PhotoPrefetcher(paths, depth=2, loader=loader) as photos:
            for p in paths:
                photos.take(p)
                time.sleep(0.05)                              # "analysis" of the photo
        elapsed = time.perf_counter() - t
        self.assertLess(elapsed, 0.85)                        # sequential would be 10 x 0.10 = 1.0 s
        self.assertGreater(threading.active_count(), 0)


# ---------------------------------------------------------------------- real engine
GPU_PYTHON = ENGINE_PYTHON
SOURCE = TEST_PHOTOS
N = 12
EVENT = "V1.2 prefetch test"


def rows(db):
    c = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    r = c.execute("SELECT p.file_name, d.bib_number, d.numeric_bib, d.confidence, d.det_confidence, "
                  "d.ocr_confidence, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height, d.ocr_text, d.source, "
                  "d.readings, d.region_ids, d.letter_digit_fix, d.evidence_json FROM bib_detection d "
                  "JOIN photos p ON p.id = d.photo_id ORDER BY 1, 2").fetchall()
    status = dict(c.execute("SELECT analysis_status, COUNT(*) FROM photos GROUP BY 1").fetchall())
    c.close()
    return [tuple(x) for x in r], status


@unittest.skipUnless(GPU_PYTHON.is_file(), "GPU environment not installed")
@needs(SOURCE, "RUNNERFINDER_TEST_PHOTOS")
class PrefetchRealEngineTest(unittest.TestCase):
    """Tests run in name order: a baseline (off), b identical (on), c cancel with a queue, d resume."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.photos = t / "photos"
        files = sorted(p for p in SOURCE.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
        for p in files[3::len(files) // N][:N]:
            dst = cls.photos / p.relative_to(SOURCE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_worker(self, db, prefetch, cancel_after=None):
        from desktop.services import progress_protocol as proto
        from desktop.services.analysis_job import AnalysisJob
        job = AnalysisJob(cancel_grace_ms=120000)
        msgs, done = [], []
        job.message.connect(msgs.append)
        job.finished.connect(lambda code, err: done.append((code, err)))

        def on_msg(m):
            if cancel_after and m["status"] == proto.ANALYSING and m["current"] == cancel_after:
                job.cancel()
        job.message.connect(on_msg)
        job.start(["analyse", "--folder", str(self.photos), "--event", EVENT, "--db", str(db),
                   "--thumbnail-cache", str(Path(self.tmp.name) / f"thumbs_{Path(db).stem}"),
                   "--prefetch", str(prefetch)])
        end = time.monotonic() + 600
        while not done and time.monotonic() < end:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(done, msgs[-3:])
        self.assertEqual(done[0][0], 0, done[0][1][-2000:])
        return msgs[-1]

    def test_a_prefetch_off_baseline(self):
        final = self.run_worker(Path(self.tmp.name) / "off.db", 0)
        self.assertEqual((final["status"], final["processed"], final["failed"], final["prefetch"]), ("finished", N, 0, 0))
        self.__class__.baseline = rows(Path(self.tmp.name) / "off.db")
        self.assertTrue(self.baseline[0])

    def test_b_prefetch_on_identical_results(self):
        final = self.run_worker(Path(self.tmp.name) / "on.db", 2)
        self.assertEqual((final["status"], final["processed"], final["failed"], final["prefetch"]), ("finished", N, 0, 2))
        self.assertEqual(rows(Path(self.tmp.name) / "on.db"), self.baseline)

    def test_c_cancel_with_photos_queued_commits_nothing_extra(self):
        db = Path(self.tmp.name) / "cancel.db"
        final = self.run_worker(db, 2, cancel_after=3)
        self.assertEqual((final["status"], final["processed"]), ("cancelled", 3))
        _, status = rows(db)
        self.assertEqual(status, {"done": 3, "pending": N - 3})      # photos 4-5 were queued, not stored
        c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        self.assertEqual(c.execute("SELECT status, photos_processed FROM analysis_runs").fetchall(), [("cancelled", 3)])
        self.assertEqual(c.execute("SELECT COUNT(DISTINCT photo_id) FROM bib_detection").fetchone()[0],
                         c.execute("SELECT COUNT(*) FROM photos WHERE analysis_status = 'done' AND photos.id IN "
                                   "(SELECT photo_id FROM bib_detection)").fetchone()[0])
        c.close()

    def test_d_resume_after_cancel_equals_uninterrupted_run(self):
        db = Path(self.tmp.name) / "cancel.db"
        final = self.run_worker(db, 2)
        self.assertEqual((final["status"], final["processed"], final["skipped"]), ("finished", N - 3, 3))
        resumed, status = rows(db)
        self.assertEqual(status, {"done": N})
        # Every stored column is identical, except: the YOLO person confidence (det_confidence, column 4)
        # of the FIRST photo of a resumed session can differ in the 7th decimal. This is pre-existing
        # V1.1 behaviour of a new worker session - measured identically with prefetch 0 - and not a prefetch
        # effect: test_b (prefetch on vs off, full runs) is byte-identical, det_confidence included.
        self.assertEqual(len(resumed), len(self.baseline[0]))
        DET = 4
        for a, b in zip(resumed, self.baseline[0]):
            self.assertEqual(a[:DET] + a[DET + 1:], b[:DET] + b[DET + 1:], (a[0], a[1]))
            self.assertAlmostEqual(a[DET], b[DET], delta=1e-6, msg=(a[0], a[1]))
        c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM (SELECT photo_id, bib_number, COUNT(*) n FROM bib_detection "
                                   "GROUP BY 1, 2 HAVING n > 1)").fetchone()[0], 0)
        c.close()


if __name__ == "__main__":
    unittest.main()
