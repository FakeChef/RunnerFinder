"""Phase 8.2 UI tests: dashboard, theme, app icon, search experience (preview, recent searches,
thumbnail size, keyboard shortcuts). Real service on a COPY of the v02 index; no OCR.
Run from the repository folder:
    <python with PySide6 + Pillow> -m unittest desktop.tests.test_ui_polish -v
"""
import gc
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

from PySide6.QtCore import QEvent, QSettings, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop import os_actions, theme  # noqa: E402
from desktop.app import build  # noqa: E402
from desktop.main_window import DASHBOARD, NEW_EVENT, SEARCH  # noqa: E402
from desktop.services.real_analysis_service import RealAnalysisService  # noqa: E402
from desktop.tests.private_data import INDEX_DB, TEST_PHOTOS, THUMBNAILS, needs  # noqa: E402

SOURCE_DB = INDEX_DB
EVENT = "0927 Race"
APP = QApplication.instance() or QApplication(sys.argv)


def pump_until(condition, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


@needs(SOURCE_DB, "RUNNERFINDER_TEST_INDEX")
class UiPolishTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "index.db"
        shutil.copy2(SOURCE_DB, cls.db)
        cls.thumbs = Path(cls.tmp.name) / "thumbs"
        if THUMBNAILS.is_dir():
            shutil.copytree(THUMBNAILS, cls.thumbs)
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)          # test settings: a temp ini file
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        QSettings.setDefaultFormat(QSettings.Format.NativeFormat)
        cls.tmp.cleanup()

    def setUp(self):
        _, self.window, self.controller = build(APP, lambda parent: RealAnalysisService(
            parent, db_path=self.db, thumbnail_cache=self.thumbs, simulate_seconds=0.01))
        self.dash, self.search = self.window.dashboard_page, self.window.search_page
        self.controller.start()

    def tearDown(self):
        self.controller.service.shutdown()
        self.window.close()
        self.window.deleteLater()
        APP.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        gc.collect()

    # ------------------------------------------------------------------ dashboard
    def test_dashboard_shows_the_index(self):
        self.assertEqual(self.window.current_page(), DASHBOARD)
        self.assertTrue(pump_until(lambda: self.dash.table.rowCount() > 0))
        conn = sqlite3.connect(f"file:{self.db.as_posix()}?mode=ro", uri=True)
        events = {n: c for n, c in conn.execute("SELECT e.name, COUNT(p.id) FROM events e LEFT JOIN photos p "
                                                "ON p.event_id = e.id GROUP BY e.id")}
        conn.close()
        rows = {self.dash.table.item(r, 0).text(): self.dash.table.item(r, 1).text()
                for r in range(self.dash.table.rowCount())}
        self.assertEqual(rows, {n: f"{c:,}" for n, c in events.items()})
        self.assertEqual(self.dash.events_tile.value.text(), str(len(events)))
        self.assertEqual(self.dash.photos_tile.value.text(), f"{sum(events.values()):,}")
        self.assertTrue(pump_until(lambda: "✓" in self.dash.gpu_chip.text() or "not" in self.dash.gpu_chip.text(), 120))

    def test_dashboard_actions_lead_to_search_and_new_event(self):
        pump_until(lambda: self.dash.table.rowCount() > 0)
        row = next(r for r in range(self.dash.table.rowCount()) if self.dash.table.item(r, 0).text() == EVENT)
        self.dash.table.selectRow(row)
        self.dash.search_button.click()
        self.assertEqual(self.window.current_page(), SEARCH)
        self.assertEqual(self.search.event_combo.currentText(), EVENT)
        self.window.show_page(DASHBOARD)
        self.dash.add_button.click()
        self.assertEqual(self.window.current_page(), NEW_EVENT)
        self.assertEqual(self.window.new_event_page.event_edit.text(), EVENT)

    @needs(TEST_PHOTOS, "RUNNERFINDER_TEST_PHOTOS")
    def test_dashboard_updates_after_an_analysis(self):
        pump_until(lambda: self.dash.table.rowCount() > 0)
        before = self.dash.table.rowCount()
        new = self.window.new_event_page
        new.allow_without_gpu(True)
        new.set_folder(str(TEST_PHOTOS))
        new.event_edit.setText("dashboard dry run")
        self.assertTrue(pump_until(lambda: new.start_button.isEnabled(), 60))
        new.start_button.click()
        prog = self.window.progress_page
        self.assertTrue(pump_until(lambda: prog.event_label.text().startswith("Analysis completed"), 120))
        # a dry run writes nothing: the dashboard is refreshed and still shows the same events
        self.assertTrue(pump_until(lambda: self.dash.table.rowCount() == before, 10))
        self.assertTrue(prog.percent_label.text().startswith("✓"))

    # ------------------------------------------------------------------ theme and icon
    def test_theme_toggle_is_applied_and_remembered(self):
        start = theme.current()
        self.window.toggle_theme()
        other = theme.current()
        self.assertNotEqual(start, other)
        self.assertEqual(QSettings().value("ui/theme"), other)
        self.assertIn(theme.PALETTES[other]["bg"], APP.styleSheet())
        self.window.toggle_theme()
        self.assertEqual(theme.current(), start)

    def test_application_and_window_icon(self):
        self.assertTrue(theme.APP_ICON.is_file())
        self.assertFalse(self.window.windowIcon().isNull())
        self.assertFalse(APP.windowIcon().isNull())

    # ------------------------------------------------------------------ search experience
    def _search(self, bib):
        self.window.show_page(SEARCH)
        pump_until(lambda: self.search.event_combo.findText(EVENT) >= 0)
        self.search.select_event(EVENT)
        self.search.bib_edit.setText(bib)
        self.search.search_button.click()
        self.assertTrue(pump_until(lambda: self.search.result_label.text().startswith(("Results", "No photos"))))

    def test_preview_recent_searches_and_size(self):
        self._search("69932")
        self.assertEqual(self.search.results_stack.currentWidget(), self.search.results)
        self.assertEqual(self.search.preview.text(), "No photo selected")
        self.search.results.setCurrentIndex(self.search.model.index(0, 0))
        r = self.search.selected_result()
        self.assertEqual(self.search.preview_title.text(), r.file_name)
        self.assertEqual(self.search.detail_bibs.text(), "69932")
        self.assertEqual(self.search.detail_status.text(), "Found by the analysis")
        self.assertFalse(self.search.preview.pixmap().isNull())
        self._search("67657")
        # no result: empty state, still remembered (V1.3.1: "123" now finds "0123" as an alias, so a
        # number that is in no index is used)
        self._search("8888888")
        self.assertEqual(self.search.results_stack.currentWidget(), self.search.empty_label)
        self.assertIn("No photos or videos found for bib 8888888", self.search.empty_label.text())
        self.assertEqual(self.search.recent_searches()[:3], ["8888888", "67657", "69932"])
        self.search._recent_buttons[2].click()                 # recent chip: search 69932 again
        self.assertTrue(pump_until(lambda: "bib 69932" in self.search.result_label.text()))
        self.search.size_slider.setValue(300)
        self.assertEqual(self.search.results.iconSize().width(), 300)

    def test_keyboard_shortcuts(self):
        self._search("69932")
        self.window.show()
        self.window.activateWindow()                           # shortcuts work in the active window
        self.assertTrue(QTest.qWaitForWindowActive(self.window, 5000))
        self.search.results.setFocus()
        self.search.results.setCurrentIndex(self.search.model.index(0, 0))
        opened, folders = [], []
        orig = os_actions.open_photo, os_actions.show_in_folder
        os_actions.open_photo, os_actions.show_in_folder = (lambda p: opened.append(p) or True,
                                                            lambda p: folders.append(p) or True)
        try:
            QTest.keyClick(self.search.results, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
            QTest.keyClick(self.search.results, Qt.Key.Key_E, Qt.KeyboardModifier.ControlModifier)
        finally:
            os_actions.open_photo, os_actions.show_in_folder = orig
        path = self.search.selected_result().file_path
        self.assertEqual((opened, folders), ([path], [path]))
        n = self.search.model.rowCount()
        QTest.keyClick(self.search.results, Qt.Key.Key_Delete)          # reject
        self.assertTrue(pump_until(lambda: self.search.model.rowCount() == n - 1))
        self.search.show_rejected_check.setChecked(True)
        self.assertTrue(pump_until(lambda: self.search.model.rowCount() == n))
        row = next(i for i in range(n) if "rejected" in self.search.model.item(i).text())
        self.search.results.setCurrentIndex(self.search.model.index(row, 0))
        self.assertTrue(self.search.detail_status.text().startswith("Rejected"))
        QTest.keyClick(self.search.results, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)   # restore
        self.assertTrue(pump_until(lambda: self.search.result_label.text().startswith("Restored")))
        self.search.show_rejected_check.setChecked(False)
        self.assertTrue(pump_until(lambda: self.search.model.rowCount() == n))
        QTest.keyClick(self.search.results, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
        self.assertTrue(self.search.bib_edit.hasFocus() or self.search.bib_edit.selectedText() == "69932")
        conn = sqlite3.connect(f"file:{self.db.as_posix()}?mode=ro", uri=True)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM detection_feedback").fetchone()[0], 0)
        conn.close()


if __name__ == "__main__":
    unittest.main()
