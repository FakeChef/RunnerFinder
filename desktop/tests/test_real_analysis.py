"""Phase 7.3B: the REAL V0.3.1 engine in the worker process (YOLO + PaddleOCR on the GPU, SQLite).

A 12-photo COPY of TestPhotos and fresh temporary databases / thumbnail folders:
the v02 index, its thumbnail cache and the originals are never written.
Needs the GPU environment (RUNNERFINDER_WORKER_PYTHON or .venv-gpu) and the NVIDIA GPU; about 2 minutes.
Run from the repository folder:
    <python with PySide6 + Pillow> -m unittest desktop.tests.test_real_analysis -v
"""
import gc
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.app import build  # noqa: E402
from desktop.main_window import PROGRESS, SEARCH  # noqa: E402
from desktop.services import progress_protocol as proto  # noqa: E402
from desktop.services.analysis_job import AnalysisJob  # noqa: E402
from desktop.services.real_analysis_service import RealAnalysisService  # noqa: E402
from desktop.tests.private_data import ENGINE_PYTHON, INDEX_DB, TEST_PHOTOS, needs  # noqa: E402

SOURCE = TEST_PHOTOS
SOURCE_DB = INDEX_DB
N = 12
EVENT = "7.3B real engine test"
NOT_IN_GUI = {"paddle", "paddleocr", "paddlex", "torch", "ultralytics", "cv2", "ocr", "detection"}
APP = QApplication.instance() or QApplication(sys.argv)


def pump_until(condition, timeout=300.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ro(db):
    conn = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def bib_rows(db, event):
    """Everything the analysis stored for the event, keyed by file name (comparable across databases)."""
    conn = ro(db)
    rows = conn.execute(
        "SELECT p.file_name, d.bib_number, d.numeric_bib, ROUND(d.confidence, 4) AS conf, d.readings, "
        "json_extract(d.evidence_json, '$.filter.accepted') AS accepted "
        "FROM bib_detection d JOIN photos p ON p.id = d.photo_id JOIN events e ON e.id = p.event_id "
        "WHERE e.name = ? ORDER BY 1, 2", (event,)).fetchall()
    conn.close()
    return [tuple(r) for r in rows]


@unittest.skipUnless(ENGINE_PYTHON.is_file(), "GPU environment not installed (RUNNERFINDER_WORKER_PYTHON)")
@needs(SOURCE, "RUNNERFINDER_TEST_PHOTOS")
@needs(SOURCE_DB, "RUNNERFINDER_TEST_INDEX")
class RealAnalysisTest(unittest.TestCase):
    """Tests run in name order: a cancel, b resume, c up to date, d = CLI, e GUI, z isolation."""

    @classmethod
    def setUpClass(cls):
        cls.source_hash = sha(SOURCE_DB)
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.photos = t / "photos"
        files = sorted(p for p in SOURCE.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
        for p in files[::len(files) // N][:N]:
            dst = cls.photos / p.relative_to(SOURCE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        cls.db, cls.thumbs = t / "worker.db", t / "thumbs"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_worker(self, cancel_after=None, extra=()):
        job = AnalysisJob(cancel_grace_ms=120000)
        msgs, done = [], []
        job.message.connect(msgs.append)
        job.finished.connect(lambda code, err: done.append((code, err)))

        def on_msg(m):
            if cancel_after and m["status"] == proto.ANALYSING and m["current"] == cancel_after:
                job.cancel()
        job.message.connect(on_msg)
        job.start(["analyse", "--folder", str(self.photos), "--event", EVENT, "--db", str(self.db),
                   "--thumbnail-cache", str(self.thumbs), *extra])
        self.assertTrue(pump_until(lambda: done, 600), msgs[-3:])
        self.assertEqual(done[0][0], 0, done[0][1][-2000:])
        return msgs

    # ------------------------------------------------------------------ worker + real engine + SQLite
    def test_a_cancel_stops_after_the_current_photo(self):
        msgs = self.run_worker(cancel_after=3)
        statuses = [m["status"] for m in msgs]
        self.assertIn(proto.LOADING_MODELS, statuses)
        self.assertEqual(next(m for m in msgs if m["status"] == proto.MODELS_LOADED)["ocr_device"], "gpu:0")
        final = msgs[-1]
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(final["processed"], 3)              # photo 3 was finished and committed
        conn = ro(self.db)
        status = dict(conn.execute("SELECT analysis_status, COUNT(*) FROM photos GROUP BY 1").fetchall())
        runs = conn.execute("SELECT status, photos_processed FROM analysis_runs").fetchall()
        conn.close()
        self.assertEqual(status, {"done": 3, "pending": N - 3})
        self.assertEqual([tuple(r) for r in runs], [("cancelled", 3)])

    def test_b_resume_analyses_only_the_rest(self):
        msgs = self.run_worker()
        final = msgs[-1]
        self.assertEqual((final["status"], final["processed"], final["skipped"], final["failed"]),
                         ("finished", N - 3, 3, 0))
        self.assertEqual(final["device"], "gpu:0")
        conn = ro(self.db)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM photos WHERE analysis_status = 'done'").fetchone()[0], N)
        dup = conn.execute("SELECT COUNT(*) FROM (SELECT photo_id, bib_number, COUNT(*) n FROM bib_detection "
                           "GROUP BY 1, 2 HAVING n > 1)").fetchone()[0]
        self.assertEqual(dup, 0)                                 # no duplicate rows
        fps = [r[0] for r in conn.execute("SELECT fingerprint FROM photos")]
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM bib_detection").fetchone()[0], 0)
        conn.close()
        for fp in fps:                                           # thumbnails made during the analysis
            self.assertTrue((self.thumbs / fp[:2] / f"{fp}.jpg").is_file(), fp)
        main = ro(SOURCE_DB)                                     # same settings as the validated V0.3.1 runs
        hashes = {r[0] for r in main.execute("SELECT DISTINCT settings_hash FROM analysis_runs "
                                             "WHERE source = 'ocr' AND settings_hash IS NOT NULL")}
        main.close()
        self.assertIn(final["settings_hash"], hashes)

    def test_c_up_to_date_loads_no_models(self):
        msgs = self.run_worker()
        self.assertNotIn(proto.LOADING_MODELS, [m["status"] for m in msgs])
        self.assertEqual((msgs[-1]["status"], msgs[-1]["processed"], msgs[-1]["skipped"]), ("finished", 0, N))

    def test_d_results_equal_the_cli(self):
        cli_db = Path(self.tmp.name) / "cli.db"
        out = subprocess.run([str(ENGINE_PYTHON), "index_cli.py", "analyze", str(self.photos), "--event", EVENT,
                              "--ocr-device", "gpu", "--db", str(cli_db)], cwd=str(ROOT), capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=900)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:] + out.stderr[-2000:])
        worker, cli = bib_rows(self.db, EVENT), bib_rows(cli_db, EVENT)
        self.assertTrue(worker)
        self.assertEqual(worker, cli)

    # ------------------------------------------------------------------ GUI with the real engine
    def test_e_gui_real_run_then_search(self):
        db = Path(self.tmp.name) / "gui.db"
        _, window, controller = build(APP, lambda parent: RealAnalysisService(
            parent, db_path=db, thumbnail_cache=Path(self.tmp.name) / "gui_thumbs"))
        new, prog, search = window.new_event_page, window.progress_page, window.search_page
        try:
            controller.start()
            self.assertTrue(pump_until(lambda: new.gpu_label.text().endswith("✓"), 120), new.gpu_label.text())
            new.set_folder(str(self.photos))
            self.assertTrue(pump_until(lambda: new.count_label.text() == str(N), 60))
            new.event_edit.setText("GUI real run")
            new.start_button.click()
            self.assertEqual(window.current_page(), PROGRESS)
            seen = set()
            self.assertTrue(pump_until(lambda: seen.add(prog.status_label.text()) or
                                       prog.event_label.text().startswith(("Analysis completed", "Error")), 600),
                            prog.current_label.text())
            self.assertTrue(prog.event_label.text().startswith("Analysis completed"), prog.current_label.text())
            self.assertTrue(any(s.startswith("Loading models") for s in seen), seen)
            self.assertTrue(any(s.startswith("Analysing") for s in seen), seen)
            self.assertEqual(prog.photos_label.text(), f"{N} / {N}")
            self.assertEqual(prog.failed_label.text(), "0")
            self.assertNotEqual(prog.duration_label.text(), "-")
            self.assertTrue(prog.search_button.isEnabled())
            bib = bib_rows(db, "GUI real run")[0][2]             # a number the engine read
            prog.search_button.click()
            self.assertEqual(window.current_page(), SEARCH)
            self.assertTrue(pump_until(lambda: search.event_combo.findText("GUI real run") >= 0, 30))
            self.assertEqual(search.event_combo.currentText(), "GUI real run")
            search.bib_edit.setText(bib)
            search.search_button.click()
            self.assertTrue(pump_until(lambda: search.result_label.text().startswith(("Results", "No photos")), 60))
            self.assertTrue(search.result_label.text().startswith("Results"), search.result_label.text())
            self.assertIn(bib, search.model.item(0).text())     # card: file name, bib · event
        finally:
            controller.service.shutdown()
            window.close()
            window.deleteLater()
            APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            APP.processEvents()
            gc.collect()

    def test_z_gui_process_isolated_and_v02_untouched(self):
        loaded = {m.split(".")[0] for m in sys.modules}
        self.assertFalse(loaded & NOT_IN_GUI, loaded & NOT_IN_GUI)
        self.assertNotIn("indexing.analyze", sys.modules)
        self.assertEqual(sha(SOURCE_DB), self.source_hash)


if __name__ == "__main__":
    unittest.main()
