"""Builds and runs the desktop application (no command-line interaction)."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from desktop import theme
from desktop.controller import AppController
from desktop.main_window import MainWindow
from desktop.services.demo_service import DemoEngineService


def build(app: QApplication | None = None, service_factory=None):
    """Create window, service and controller. Returns (app, window, controller).
    service_factory(parent) -> EngineService; default: the demo service."""
    app = app or QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("RunnerFinder")
    app.setOrganizationName("RunnerFinder")
    app.setWindowIcon(theme.app_icon())
    theme.apply(app)                       # saved choice, dark by default
    window = MainWindow()
    service = (service_factory or DemoEngineService)(window)
    controller = AppController(window, service)
    return app, window, controller


def real_service(parent):
    """Phase 7.2: real search / review on the RunnerFinder index; analysis still simulated."""
    from desktop.services.runnerfinder_service import RunnerFinderSearchService
    return RunnerFinderSearchService(parent)


def real_analysis_service(parent):
    """Phase 7.3A: real search + analysis worker process (dry run, no OCR)."""
    from desktop.services.real_analysis_service import RealAnalysisService
    return RealAnalysisService(parent)


def _windows_taskbar_identity() -> None:
    """Own taskbar entry with the RunnerFinder icon (instead of pythonw's)."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RunnerFinder.Desktop")
        except (AttributeError, OSError):
            pass


def main() -> int:
    _windows_taskbar_identity()
    app, window, controller = build(service_factory=real_analysis_service)
    if hasattr(controller.service, "shutdown"):
        app.aboutToQuit.connect(controller.service.shutdown)
    window.show()
    controller.start()
    return app.exec()
