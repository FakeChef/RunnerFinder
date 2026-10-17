"""Phase 7.3A GUI tests: the real window with RealAnalysisService (worker process, dry run).
(7.3B: the dry run is the test-only simulate_seconds option; the real engine: test_real_analysis.py.)

Uses a COPY of the v02 index (prepare_event only reads it) and real photo folders
(read-only). No OCR, nothing written.
Run from the repository folder:
    <python with PySide6 + Pillow> -m unittest desktop.tests.test_analysis_gui -v
"""
import gc
import hashlib
import os
import shutil
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
from desktop.main_window import NEW_EVENT, PROGRESS  # noqa: E402
from desktop.services.real_analysis_service import RealAnalysisService  # noqa: E402
from desktop.tests.private_data import ENGINE_PYTHON, INDEX_DB, REAL_EVENT, TEST_PHOTOS, needs  # noqa: E402

SOURCE_DB = INDEX_DB
PHOTOS = TEST_PHOTOS                                    # 370 photos
GPU_VENV = ENGINE_PYTHON
NOT_IN_GUI = {"paddle", "paddleocr", "paddlex", "torch", "ultralytics", "cv2", "ocr", "detection"}
APP = QApplication.instance() or QApplication(sys.argv)


def pump_until(condition, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_state(folder):
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in Path(folder).rglob("*") if p.is_file()}


@needs(SOURCE_DB, "RUNNERFINDER_TEST_INDEX")
class AnalysisGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_hash = sha(SOURCE_DB)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "index.db"
        shutil.copy2(SOURCE_DB, cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.copy_hash = sha(self.db)
        _, self.window, self.controller = build(APP, lambda parent: RealAnalysisService(
            parent, db_path=self.db, thumbnail_cache=Path(self.tmp.name) / "thumbs", simulate_seconds=0.02))
        self.service = self.controller.service
        self.new, self.prog = self.window.new_event_page, self.window.progress_page
        self.controller.start()

    def tearDown(self):
        self.service.shutdown()
        self.window.close()
        self.window.deleteLater()
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        gc.collect()
        self.assertEqual(sha(self.db), self.copy_hash)                 # index not modified

    def wait_gpu(self):
        self.assertTrue(pump_until(lambda: self.new.gpu_label.text() != "checking…", 120))

    def select(self, folder, count):
        self.new.set_folder(str(folder))
        # V1.3: a folder with videos shows "6,974 photos · 53 videos" (the real event folder has a V subfolder with 53 clips)
        self.assertTrue(pump_until(lambda: self.new.count_label.text().split(" photos")[0] == f"{count:,}", 120),
                        self.new.count_label.text() + " " + self.new.message_label.text())

    # ------------------------------------------------------------------ page 1
    def test_gpu_status_is_displayed(self):
        self.wait_gpu()
        if GPU_VENV.is_file():
            self.assertTrue(self.new.gpu_label.text().endswith("✓"), self.new.gpu_label.text())
            self.assertIn("RTX", self.new.gpu_label.text())
            self.assertTrue(self.new.cuda_label.text().endswith("✓"), self.new.cuda_label.text())
            self.assertTrue(self.new.paddle_label.text().startswith("available"), self.new.paddle_label.text())
        else:
            self.assertIn("not available", self.new.gpu_label.text())

    @needs(PHOTOS, "RUNNERFINDER_TEST_PHOTOS")
    def test_select_real_folder_counts_photos_and_estimates(self):
        self.select(PHOTOS, 370)
        self.assertEqual(self.new.estimate_label.text(), "about 4 min (GPU)")       # 370 x 0.8 s
        self.assertEqual(self.service.estimate_duration(6974), 6974 * 0.8)

    @needs(REAL_EVENT, "RUNNERFINDER_REAL_EVENT")
    def test_real_event_folder_count(self):
        before = tree_state(REAL_EVENT / "C2")
        self.select(REAL_EVENT, 6974)
        # V1.3: 6,974 photos x 0.8 s + 53 videos x 3.2 s (the V folder of the event)
        self.assertEqual(self.new.estimate_label.text(), "about 1 h 35 min (GPU)")
        self.assertEqual(tree_state(REAL_EVENT / "C2"), before)            # originals untouched

    def test_start_requires_the_gpu(self):
        self.assertFalse(self.new._allow_without_gpu)                     # real service: no demo bypass

    # ------------------------------------------------------------------ page 2
    @needs(PHOTOS, "RUNNERFINDER_TEST_PHOTOS")
    def test_start_switches_to_progress_and_cancel_returns_safely(self):
        self.wait_gpu()
        if not GPU_VENV.is_file():
            self.skipTest("no GPU environment")
        before = tree_state(PHOTOS)
        self.select(PHOTOS, 370)
        self.new.event_edit.setText("Phase 7.3A dry run")
        self.assertTrue(self.new.start_button.isEnabled())
        self.new.start_button.click()
        self.assertEqual(self.window.current_page(), PROGRESS)
        self.assertTrue(self.service.busy)
        self.assertFalse(self.new.start_button.isEnabled())
        self.assertTrue(pump_until(lambda: self.prog.total_label.text() == "370", 60))
        self.assertTrue(pump_until(lambda: "sec/photo" in self.prog.speed_label.text(), 60))
        self.assertEqual(self.prog.event_label.text(), "Analysing: Phase 7.3A dry run")
        self.assertEqual(self.prog.folder_label.text(), str(PHOTOS))
        self.assertNotEqual(self.prog.time_label.text(), "-")
        self.assertTrue(self.prog.file_label.text().lower().endswith(".jpg"))
        self.assertTrue(self.prog.cancel_button.isEnabled())
        pid = self.service._job.pid
        self.prog.cancel_button.click()
        self.assertTrue(pump_until(lambda: self.prog.event_label.text().startswith("Cancelled"), 15))
        self.assertFalse(self.service.busy)
        self.assertFalse(self.prog.cancel_button.isEnabled())
        self.assertTrue(self.new.start_button.isEnabled())                # can start again
        self.assertIn("continue where it stopped", self.prog.current_label.text())
        tasks = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                               errors="replace").stdout
        alive = str(pid) in tasks
        self.assertFalse(alive, "worker process still running")
        self.assertEqual(tree_state(PHOTOS), before)

    @needs(PHOTOS, "RUNNERFINDER_TEST_PHOTOS")
    def test_complete_dry_run(self):
        self.service.simulate_seconds = 0.001
        self.new.allow_without_gpu(True)                                  # GPU not needed for the dry run itself
        self.select(PHOTOS, 370)
        self.new.event_edit.setText("dry")
        self.new.start_button.click()
        self.assertTrue(pump_until(lambda: self.prog.event_label.text().startswith("Analysis completed"), 120),
                        self.prog.event_label.text() + " " + self.prog.current_label.text())
        self.assertEqual(self.prog.photos_label.text(), "370 / 370")
        self.assertEqual(self.prog.failed_label.text(), "0")
        self.assertEqual(self.window.new_event_page.start_button.isEnabled(), True)

    def test_missing_folder_reported_on_new_event_page(self):
        self.window.show_page(NEW_EVENT)
        self.service.start_analysis("x", str(Path(self.tmp.name) / "missing"))
        self.assertIn("Folder not found", self.new.message_label.text())
        self.assertFalse(self.service.busy)

    # ------------------------------------------------------------------ isolation
    def test_zz_no_ocr_module_in_the_gui_process(self):
        loaded = {m.split(".")[0] for m in sys.modules}
        self.assertFalse(loaded & NOT_IN_GUI, loaded & NOT_IN_GUI)
        self.assertNotIn("indexing.analyze", sys.modules)
        self.assertEqual(sha(SOURCE_DB), self.source_hash)


if __name__ == "__main__":
    unittest.main()
