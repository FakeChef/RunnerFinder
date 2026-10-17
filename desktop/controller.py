"""
Connects the pages to the EngineService with signals and slots.

Pages know nothing about the service; the service knows nothing about the
pages. Replacing DemoEngineService by the real engine service in a later
phase changes nothing here.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QMessageBox

from desktop import os_actions
from desktop.main_window import DASHBOARD, NEW_EVENT, PROGRESS, SEARCH, MainWindow
from desktop.services.engine_service import EngineService
from desktop.services.models import AnalysisSummary, GpuStatus


class AppController(QObject):
    def __init__(self, window: MainWindow, service: EngineService):
        super().__init__(window)
        self.window, self.service = window, service
        new, prog, search = window.new_event_page, window.progress_page, window.search_page
        dash = window.dashboard_page

        # page -> service
        new.folder_selected.connect(service.inspect_folder)
        new.start_requested.connect(self.start_analysis)
        prog.cancel_requested.connect(service.cancel_analysis)
        prog.open_search_requested.connect(self.open_search)
        search.search_requested.connect(self.search)
        search.open_photo_requested.connect(self.open_photo)
        search.show_in_folder_requested.connect(self.show_in_folder)
        search.reject_requested.connect(service.reject)
        search.restore_requested.connect(service.restore)
        dash.new_event_requested.connect(lambda: self.add_photos(""))
        dash.search_event_requested.connect(self.open_search)
        dash.add_photos_requested.connect(self.add_photos)
        dash.show_folder_requested.connect(self.show_folder)

        # service -> pages
        service.gpu_status_changed.connect(new.show_gpu_status)
        service.gpu_status_changed.connect(dash.show_gpu_status)
        service.event_summaries_changed.connect(dash.show_summaries)
        service.gpu_status_changed.connect(self._gpu_for_progress)
        service.folder_inspected.connect(new.show_folder_info)
        service.analysis_progress.connect(prog.show_progress)
        service.analysis_finished.connect(prog.show_summary)
        service.analysis_finished.connect(self._analysis_finished)
        service.events_changed.connect(search.set_events)
        service.search_finished.connect(search.show_results)
        service.search_hidden.connect(search.show_hidden_count)
        service.review_finished.connect(self._review_finished)
        service.error.connect(self._error)

        demo = not service.connected
        window.set_mode_text(service.mode_description)
        new.allow_without_gpu(demo)

    def start(self) -> None:
        """Initial requests after the window is shown."""
        self.service.check_gpu()
        self.service.list_events()
        self.service.list_event_summaries()

    # ---------------------------------------------------------------- slots
    @Slot(str, str)
    def start_analysis(self, folder: str, event: str) -> None:
        if self.service.busy:
            self._error("An analysis is already running.")
            return
        self.window.new_event_page.set_running(True)
        self.window.show_page(PROGRESS)
        self.service.start_analysis(event, folder)

    @Slot(object)
    def _analysis_finished(self, summary: AnalysisSummary) -> None:
        self.window.new_event_page.set_running(False)
        self.service.list_events()
        self.service.list_event_summaries()

    @Slot(str, str)
    def search(self, event: str, bib: str) -> None:
        page = self.window.search_page
        self.service.search(event, bib, include_rejected=page.show_rejected, include_noise=page.show_filtered,
                            include_ambiguous=page.show_ambiguous)

    @Slot(str)
    def _review_finished(self, message: str) -> None:
        page = self.window.search_page
        page.refresh()                     # same search again: the card changes or disappears
        page.show_message(message)
        self.service.list_event_summaries()

    @Slot(str)
    def open_search(self, event: str) -> None:
        self.window.search_page.select_event(event)
        self.window.show_page(SEARCH)

    @Slot(str)
    def add_photos(self, event: str) -> None:
        """Dashboard: New Event (optionally for an existing event name)."""
        if event:
            self.window.new_event_page.set_event_name(event)
        self.window.show_page(NEW_EVENT)

    @Slot(str)
    def show_folder(self, folder: str) -> None:
        if not os_actions.open_photo(folder):
            self._error(f"Folder not found:\n{folder}")

    @Slot(object)
    def _gpu_for_progress(self, status: GpuStatus) -> None:
        self.window.progress_page.set_gpu(status.name if status.available else f"not available ({status.detail})")

    @Slot(str)
    def open_photo(self, path: str) -> None:
        # V1.3: videos too - the Windows default player plays the original (read-only, nothing embedded)
        if not os_actions.open_photo(path):
            kind = "video" if path.lower().endswith((".mp4", ".mov", ".m4v")) else "photo"
            self._error(f"Cannot open the {kind}:\n{path}")

    @Slot(str)
    def show_in_folder(self, path: str) -> None:
        if not os_actions.show_in_folder(path):
            self._error(f"File not found:\n{path}")

    @Slot(str)
    def _error(self, message: str) -> None:
        page = self.window.current_page()
        if page in (NEW_EVENT, DASHBOARD):
            self.window.new_event_page.show_message(message)
            self.window.dashboard_page.show_message(message)
        elif page == SEARCH:
            self.window.search_page.show_message(message)
        else:
            QMessageBox.warning(self.window, "RunnerFinder", message)
