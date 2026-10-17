"""V1.3 three-stage pipeline (S3) tests.

Unit part (no AI): scanner.pipeline3.DetectionPipeline - order, error hand-off (decode and detection),
bounded look-ahead, the per-detection OpenMP hook in the detection thread, stop and drain.
Real-engine part (GPU environment, ~3 min): the worker on a 12-photo COPY into temporary databases -
V1.2 pipeline vs S3 identical; cancel with photos queued in both stages commits nothing extra; resume;
an unreadable photo is stored exactly as by the V1.2 pipeline; a killed worker is recovered.
Run from the code folder:  <python with PySide6 + Pillow> -m unittest desktop.tests.test_v13_s3 -v
"""
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scanner.pipeline3 import DetectionPipeline  # noqa: E402
from desktop.tests.private_data import ENGINE_PYTHON, TEST_PHOTOS, needs  # noqa: E402


class FakeLoader:
    """'image' = path upper-cased; 'bad*' raises like an unreadable JPEG."""

    def __init__(self, delay=0.0):
        self.delay, self.calls = delay, []

    def __call__(self, path):
        self.calls.append(path)
        time.sleep(self.delay)
        if "bad" in path:
            raise ValueError(f"cannot decode {path}")
        return path.upper(), 6


class FakeDetect:
    """'detection' = ('DET', image); 'BOOM' images raise; records the calling thread."""

    def __init__(self, delay=0.0):
        self.delay, self.calls, self.threads = delay, [], set()

    def __call__(self, image):
        self.calls.append(image)
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        if "BOOM" in image:
            raise RuntimeError(f"detector failed on {image}")
        return ("DET", image)


class DetectionPipelineUnitTest(unittest.TestCase):
    def test_order_and_values_as_inline_processing(self):
        paths = [f"p{i}.jpg" for i in range(25)]
        det = FakeDetect()
        with DetectionPipeline(paths, det, depth=2, loader=FakeLoader()) as photos:
            got = [photos.take(p) for p in paths]
        self.assertEqual(got, [(p.upper(), 6, got[i][2], ("DET", p.upper())) for i, p in enumerate(paths)])
        self.assertEqual(det.calls, [p.upper() for p in paths])
        self.assertEqual(det.threads, {"photo-detect"})

    def test_decode_error_is_raised_for_that_photo_only_and_not_detected(self):
        paths = ["a.jpg", "bad.jpg", "c.jpg"]
        det = FakeDetect()
        with DetectionPipeline(paths, det, depth=2, loader=FakeLoader()) as photos:
            self.assertEqual(photos.take("a.jpg")[3], ("DET", "A.JPG"))
            with self.assertRaises(ValueError):
                photos.take("bad.jpg")
            self.assertEqual(photos.take("c.jpg")[3], ("DET", "C.JPG"))
        self.assertEqual(det.calls, ["A.JPG", "C.JPG"])

    def test_detection_error_is_raised_for_that_photo_only(self):
        paths = ["a.jpg", "boom.jpg", "c.jpg"]
        with DetectionPipeline(paths, FakeDetect(), depth=2, loader=FakeLoader()) as photos:
            photos.take("a.jpg")
            with self.assertRaisesRegex(RuntimeError, "detector failed on BOOM.JPG"):
                photos.take("boom.jpg")
            self.assertEqual(photos.take("c.jpg")[0], "C.JPG")

    def test_before_detect_runs_before_every_detection_in_the_detection_thread(self):
        events = []
        det = FakeDetect()

        def hook():
            events.append(("hook", threading.current_thread().name, len(det.calls)))
        paths = [f"p{i}.jpg" for i in range(6)]
        with DetectionPipeline(paths, det, depth=2, loader=FakeLoader(), before_detect=hook) as photos:
            for p in paths:
                photos.take(p)
        self.assertEqual(events, [("hook", "photo-detect", i) for i in range(6)])

    def test_look_ahead_is_bounded(self):
        paths = [f"p{i}.jpg" for i in range(40)]
        loader = FakeLoader()
        with DetectionPipeline(paths, FakeDetect(), depth=2, loader=loader) as photos:
            for n, p in enumerate(paths[:5], start=1):
                photos.take(p)
                time.sleep(0.15)                                       # slow OCR stage
                self.assertLessEqual(photos.detected, n + 2 + 1)        # queue 2 + 1 being finished by B
                self.assertLessEqual(len(loader.calls), n + 2 + 1 + 2 + 1)   # + A's queue 2 + 1 being decoded

    def test_close_stops_both_stages_and_drops_queued_photos(self):
        paths = [f"p{i}.jpg" for i in range(200)]
        loader, det = FakeLoader(delay=0.01), FakeDetect(delay=0.01)
        photos = DetectionPipeline(paths, det, depth=2, loader=loader)
        photos.take("p0.jpg")
        photos.close()
        self.assertFalse(photos.alive)
        self.assertLess(len(loader.calls), 15)
        self.assertLess(len(det.calls), 10)
        self.assertTrue(photos._queue.empty())
        self.assertTrue(photos._decoded._queue.empty())

    def test_close_while_the_consumer_never_took_anything(self):
        photos = DetectionPipeline([f"p{i}.jpg" for i in range(50)], FakeDetect(), depth=2, loader=FakeLoader())
        time.sleep(0.1)
        photos.close()
        self.assertFalse(photos.alive)

    def test_out_of_order_take_is_refused(self):
        with DetectionPipeline(["a.jpg", "b.jpg"], FakeDetect(), depth=2, loader=FakeLoader()) as photos:
            with self.assertRaisesRegex(RuntimeError, "out of order"):
                photos.take("b.jpg")

    def test_three_stages_run_concurrently(self):
        paths = [f"p{i}.jpg" for i in range(12)]
        t = time.perf_counter()
        with DetectionPipeline(paths, FakeDetect(delay=0.05), depth=2, loader=FakeLoader(delay=0.05)) as photos:
            for p in paths:
                photos.take(p)
                time.sleep(0.05)                                       # "OCR" of the photo
        self.assertLess(time.perf_counter() - t, 1.2)                  # sequential would be 12 x 0.15 = 1.8 s

    def test_thumbnail_hook_runs_in_order(self):
        seen = []
        paths = [f"p{i}.jpg" for i in range(5)]
        with DetectionPipeline(paths, FakeDetect(), lambda p, img: seen.append(p), depth=2,
                               loader=FakeLoader()) as photos:
            for p in paths:
                photos.take(p)
        self.assertEqual(seen, paths)


# ---------------------------------------------------------------------- real engine
GPU_PYTHON = ENGINE_PYTHON
SOURCE = TEST_PHOTOS
N = 12
EVENT = "V1.3 S3 test"


def rows(db):
    c = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    r = c.execute("SELECT p.file_name, d.bib_number, d.numeric_bib, d.confidence, d.det_confidence, "
                  "d.ocr_confidence, d.bbox_x, d.bbox_y, d.bbox_width, d.bbox_height, d.ocr_text, d.source, "
                  "d.readings, d.region_ids, d.letter_digit_fix, d.evidence_json FROM bib_detection d "
                  "JOIN photos p ON p.id = d.photo_id ORDER BY 1, 2").fetchall()
    status = dict(c.execute("SELECT analysis_status, COUNT(*) FROM photos GROUP BY 1").fetchall())
    errors = c.execute("SELECT file_name, analysis_status, analysis_error FROM photos ORDER BY 1").fetchall()
    c.close()
    return [tuple(x) for x in r], status, errors


@unittest.skipUnless(GPU_PYTHON.is_file(), "GPU environment not installed")
@needs(SOURCE, "RUNNERFINDER_TEST_PHOTOS")
class S3RealEngineTest(unittest.TestCase):
    """Tests run in name order: a V1.2 baseline, b S3 identical, c cancel, d resume, e unreadable photo,
    f killed worker recovered."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.photos = t / "photos"
        files = sorted(p for p in SOURCE.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
        cls.picked = files[3::len(files) // N][:N]
        for p in cls.picked:
            dst = cls.photos / p.relative_to(SOURCE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        cls.err_photos = t / "photos_err"                 # 4 good copies + 1 truncated copy
        for p in cls.picked[:4]:
            dst = cls.err_photos / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        data = cls.picked[4].read_bytes()
        (cls.err_photos / "zz_truncated.jpg").write_bytes(data[: len(data) // 3])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_worker(self, db, pipeline, cancel_after=None, folder=None):
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
        job.start(["analyse", "--folder", str(folder or self.photos), "--event", EVENT, "--db", str(db),
                   "--thumbnail-cache", str(Path(self.tmp.name) / f"thumbs_{Path(db).stem}"),
                   "--prefetch", "2", "--pipeline", pipeline])
        end = time.monotonic() + 600
        while not done and time.monotonic() < end:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(done, msgs[-3:])
        self.assertEqual(done[0][0], 0, done[0][1][-2000:])
        return msgs[-1]

    def test_a_v12_pipeline_baseline(self):
        final = self.run_worker(Path(self.tmp.name) / "v12.db", "prefetch")
        self.assertEqual((final["status"], final["processed"], final["failed"], final["pipeline"]),
                         ("finished", N, 0, "prefetch"))
        self.__class__.baseline = rows(Path(self.tmp.name) / "v12.db")
        self.assertTrue(self.baseline[0])

    def test_b_s3_identical_results(self):
        final = self.run_worker(Path(self.tmp.name) / "s3.db", "s3")
        self.assertEqual((final["status"], final["processed"], final["failed"], final["pipeline"]),
                         ("finished", N, 0, "s3"))
        self.assertEqual(rows(Path(self.tmp.name) / "s3.db")[0], self.baseline[0])     # byte-identical

    def test_c_cancel_with_photos_queued_commits_nothing_extra(self):
        db = Path(self.tmp.name) / "cancel.db"
        final = self.run_worker(db, "s3", cancel_after=3)
        self.assertEqual((final["status"], final["processed"]), ("cancelled", 3))
        _, status, _ = rows(db)
        self.assertEqual(status, {"done": 3, "pending": N - 3})      # up to 6 photos were decoded/detected, not stored
        c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        self.assertEqual(c.execute("SELECT status, photos_processed FROM analysis_runs").fetchall(), [("cancelled", 3)])
        c.close()

    def test_d_resume_after_cancel_equals_uninterrupted_run(self):
        db = Path(self.tmp.name) / "cancel.db"
        final = self.run_worker(db, "s3")
        self.assertEqual((final["status"], final["processed"], final["skipped"]), ("finished", N - 3, 3))
        resumed, status, _ = rows(db)
        self.assertEqual(status, {"done": N})
        self.assert_identical_within_det_tolerance(resumed, self.baseline[0])

    def test_e_unreadable_photo_stored_exactly_as_v12(self):
        r = {}
        for pipeline in ("prefetch", "s3"):
            db = Path(self.tmp.name) / f"err_{pipeline}.db"
            final = self.run_worker(db, pipeline, folder=self.err_photos)
            self.assertEqual((final["status"], final["processed"], final["failed"]), ("finished", 5, 1))
            r[pipeline] = rows(db)
        self.assertEqual(r["s3"][0], r["prefetch"][0])
        self.assertEqual(r["s3"][2], r["prefetch"][2])                 # same failed photo, same error text
        self.assertIn(("zz_truncated.jpg", "error"), [(e[0], e[1]) for e in r["s3"][2]])

    def test_f_killed_worker_is_recovered(self):
        db = Path(self.tmp.name) / "killed.db"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen([str(GPU_PYTHON), "-u", "-m", "desktop.services.analysis_worker", "analyse",
                              "--folder", str(self.photos), "--event", EVENT, "--db", str(db),
                              "--thumbnail-cache", str(Path(self.tmp.name) / "thumbs_killed"),
                              "--prefetch", "2", "--pipeline", "s3"],
                             cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, env=env)
        import json
        for line in p.stdout:                                           # kill hard while photo 6 is analysed
            t = line.decode("utf-8", "replace").strip()
            if t.startswith("{") and json.loads(t).get("status") == "analysing" and json.loads(t).get("current") == 6:
                p.kill()
                break
        p.wait(30)
        p.stdin.close()
        p.stdout.close()
        _, status, _ = rows(db)
        self.assertEqual(status.get("done", 0) + status.get("pending", 0), N)
        done_before = status.get("done", 0)
        self.assertGreaterEqual(done_before, 4)
        final = self.run_worker(db, "s3")
        self.assertEqual((final["status"], final["recovered_runs"]), ("finished", 1))
        self.assertEqual(final["processed"], N - done_before)
        recovered, status, _ = rows(db)
        self.assertEqual(status, {"done": N})
        self.assert_identical_within_det_tolerance(recovered, self.baseline[0])
        c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        self.assertEqual([r[0] for r in c.execute("SELECT status FROM analysis_runs ORDER BY id")],
                         ["cancelled", "finished"])
        self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        c.close()

    def assert_identical_within_det_tolerance(self, got, want):
        # As in test_v12_prefetch: every column identical except the YOLO person confidence (column 4) of the
        # first photo of a new worker session, which can differ in the 7th decimal (pre-existing V1.1 behaviour).
        self.assertEqual(len(got), len(want))
        DET = 4
        for a, b in zip(got, want):
            self.assertEqual(a[:DET] + a[DET + 1:], b[:DET] + b[DET + 1:], (a[0], a[1]))
            self.assertAlmostEqual(a[DET], b[DET], delta=1e-6, msg=(a[0], a[1]))
        db_dupes = [x for x in got if got.count(x) > 1]
        self.assertEqual(db_dupes, [])


if __name__ == "__main__":
    unittest.main()
