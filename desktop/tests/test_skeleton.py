"""Phase 7.1 skeleton tests (unittest, Qt off-screen; no engine, no OCR).

Run from the repository folder:
    <python with PySide6> -m unittest discover -s desktop\\tests -t . -v
"""
import gc
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop import os_actions  # noqa: E402
from desktop.app import build  # noqa: E402
from desktop.main_window import DASHBOARD, NEW_EVENT, PAGE_NAMES, PROGRESS, SEARCH  # noqa: E402
from desktop.services.models import GpuStatus, SearchResult, format_duration  # noqa: E402

ENGINE_MODULES = {"config", "database", "indexing", "ocr", "search", "scanner", "detection",
                  "utils", "paddle", "paddleocr", "ultralytics", "torch", "streamlit"}
APP = QApplication.instance() or QApplication(sys.argv)


def dispose(window):
    """Destroy a test window NOW, in the GUI thread. Left to Python's garbage collector, its Qt
    objects can be deleted from a background thread (crash seen in Phase 7.3A)."""
    window.close()
    window.deleteLater()
    APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    APP.processEvents()
    gc.collect()


def wait_until(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def tree_state(folder):
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in Path(folder).rglob("*") if p.is_file()}


class SkeletonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.photos = root / "Race 2026"
        (self.photos / "Camera1").mkdir(parents=True)
        (self.photos / ".hidden").mkdir()
        for name in ("Camera1/a.jpg", "Camera1/b.JPG", "c.jpeg", "notes.txt", ".hidden/x.jpg"):
            (self.photos / name).write_bytes(b"not really a photo")
        _, self.window, self.controller = build(APP)
        self.controller.service._photos_per_tick = 1          # slow enough to cancel
        self.controller.start()
        self.new = self.window.new_event_page
        self.prog = self.window.progress_page
        self.search = self.window.search_page

    def tearDown(self):
        self.controller.service._counter.wait_idle()
        dispose(self.window)
        self.tmp.cleanup()

    # ------------------------------------------------------------------ structure
    def test_three_pages_and_navigation(self):
        # Phase 8.2: Dashboard added as the first page (4 pages); the three workflow pages are unchanged
        self.assertEqual([self.window.nav.item(i).text() for i in range(self.window.nav.count())],
                         list(PAGE_NAMES))
        self.assertEqual(PAGE_NAMES[1:], ("New Event", "Analysis Progress", "Search"))
        self.assertEqual(self.window.stack.count(), 4)
        self.assertEqual(self.window.current_page(), DASHBOARD)              # start page
        for page in (SEARCH, PROGRESS, NEW_EVENT, DASHBOARD):
            self.window.nav.setCurrentRow(page)
            self.assertEqual(self.window.current_page(), page)
        self.assertIn("Demo mode", self.window.mode_label.text())

    def test_no_engine_module_is_imported(self):
        # checked in a fresh process: other test modules (Phase 7.2) load the engine on purpose
        import subprocess
        code = ("import os, sys; os.environ['QT_QPA_PLATFORM']='offscreen'; "
                "from desktop.app import build; _, w, c = build(); c.start(); "
                "print(','.join(sorted({m.split('.')[0] for m in sys.modules})))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(Path(__file__).resolve().parents[2]), timeout=120)
        loaded = set(out.stdout.strip().split(","))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertFalse(loaded & ENGINE_MODULES, loaded & ENGINE_MODULES)

    # ------------------------------------------------------------------ page 1
    def test_folder_selection_counts_photos_and_enables_start(self):
        self.assertFalse(self.new.start_button.isEnabled())
        self.assertIn("not connected", self.new.gpu_label.text())
        self.new.set_folder(str(self.photos))
        self.assertEqual(self.new.event_edit.text(), "Race 2026")            # pre-filled
        self.assertTrue(wait_until(lambda: self.new.count_label.text() == "3"))   # jpg/JPG/jpeg, no hidden
        self.assertIn("about", self.new.estimate_label.text())
        self.assertTrue(self.new.start_button.isEnabled())                   # demo: no GPU needed
        self.new.event_edit.setText("")
        self.assertFalse(self.new.start_button.isEnabled())

    def test_missing_folder_is_reported(self):
        self.new.set_folder(str(Path(self.tmp.name) / "does not exist"))
        self.assertTrue(wait_until(lambda: "not found" in self.new.message_label.text()))
        self.assertFalse(self.new.start_button.isEnabled())

    def test_real_mode_requires_the_gpu(self):
        self.new.allow_without_gpu(False)
        self.new.set_folder(str(self.photos))
        wait_until(lambda: self.new.count_label.text() == "3")
        self.assertFalse(self.new.start_button.isEnabled())
        self.new.show_gpu_status(GpuStatus(True, "NVIDIA GeForce RTX 3060"))
        self.assertEqual(self.new.gpu_label.text(), "NVIDIA GeForce RTX 3060 ✓")
        self.assertTrue(self.new.start_button.isEnabled())

    # ------------------------------------------------------------------ page 2
    def _start(self):
        self.new.set_folder(str(self.photos))
        wait_until(lambda: self.new.start_button.isEnabled())
        self.new.event_edit.setText("XTERRA Taihu 2026")
        self.new.start_button.click()

    def test_simulated_analysis_reaches_progress_page_and_search(self):
        before = tree_state(self.photos)
        self._start()
        self.assertEqual(self.window.current_page(), PROGRESS)
        self.assertFalse(self.new.start_button.isEnabled())                  # no second run
        self.assertTrue(wait_until(lambda: self.prog.event_label.text().startswith("Analysis completed")))
        self.assertEqual(self.prog.bar.value(), 100)
        self.assertEqual(self.prog.photos_label.text(), "3 / 3")
        self.assertTrue(self.prog.search_button.isEnabled())
        self.prog.search_button.click()
        self.assertEqual(self.window.current_page(), SEARCH)
        self.assertEqual(self.search.event_combo.currentText(), "XTERRA Taihu 2026")
        self.assertEqual(tree_state(self.photos), before)                    # nothing read/written

    def test_cancel(self):
        self.controller.service._timer.setInterval(500)
        self._start()
        self.assertTrue(wait_until(lambda: self.prog.cancel_button.isEnabled()))
        self.prog.cancel_button.click()
        self.assertTrue(wait_until(lambda: self.prog.event_label.text().startswith("Cancelled")))
        self.assertFalse(self.prog.cancel_button.isEnabled())
        self.assertTrue(self.new.start_button.isEnabled())                   # can start again

    # ------------------------------------------------------------------ page 3
    def test_search_validation_and_empty_result(self):
        self.search.bib_edit.setText("60A14")                 # V1.3.1: full bibs (A0175) are allowed,
        self.search.search_button.click()                    # digits split by a letter are not
        self.assertIn("Please enter a bib number", self.search.result_label.text())
        self.search.bib_edit.setText("60914")
        self.search.search_button.click()
        self.assertTrue(wait_until(lambda: "No photos or videos found for bib 60914" in self.search.result_label.text()))

    def test_results_show_thumbnails_and_open_actions(self):
        thumb = Path(self.tmp.name) / "t.jpg"
        pix = QPixmap(64, 48)
        pix.fill(QColor("red"))
        pix.save(str(thumb))
        photo = str(self.photos / "Camera1" / "a.jpg")
        results = [SearchResult("Race", "60914", photo, str(thumb)),
                   SearchResult("Race", "60914", str(self.photos / "c.jpeg"), None)]
        self.search.show_results("Race", "60914", results)
        self.assertEqual(self.search.model.rowCount(), 2)
        self.assertIn("Results: 2 photos", self.search.result_label.text())
        self.assertEqual(self.search.model.item(0).text().splitlines()[0], "a.jpg")   # 7.2: + bib/event lines
        self.assertFalse(self.search.open_button.isEnabled())
        opened, shown = [], []
        orig = os_actions.open_photo, os_actions.show_in_folder
        os_actions.open_photo, os_actions.show_in_folder = (lambda p: opened.append(p) or True,
                                                            lambda p: shown.append(p) or True)
        try:
            self.search.results.setCurrentIndex(self.search.model.index(0, 0))
            self.assertTrue(self.search.open_button.isEnabled())
            self.search.open_button.click()
            self.search.folder_button.click()
        finally:
            os_actions.open_photo, os_actions.show_in_folder = orig
        self.assertEqual((opened, shown), ([photo], [photo]))

    def test_format_duration(self):
        self.assertEqual([format_duration(x) for x in (None, 45, 720, 5592)], ["-", "45 s", "12 min", "1 h 33 min"])


if __name__ == "__main__":
    unittest.main()
