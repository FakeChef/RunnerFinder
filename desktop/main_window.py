"""Main window: sidebar (brand, navigation, theme), the four pages, status bar (Phase 8.2 design)."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from desktop import __version__, theme
from desktop.pages.dashboard_page import DashboardPage
from desktop.pages.new_event_page import NewEventPage
from desktop.pages.progress_page import ProgressPage
from desktop.pages.search_page import SearchPage

DASHBOARD, NEW_EVENT, PROGRESS, SEARCH = range(4)
PAGE_NAMES = ("Dashboard", "New Event", "Analysis Progress", "Search")
NAV_GLYPHS = ("dashboard", "new", "progress", "search")


class MainWindow(QMainWindow):
    theme_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RunnerFinder")
        self.setWindowIcon(theme.app_icon())
        self.resize(1280, 800)
        self.setMinimumSize(QSize(980, 640))

        self.dashboard_page = DashboardPage()
        self.new_event_page = NewEventPage()
        self.progress_page = ProgressPage()
        self.search_page = SearchPage()

        # ------------------------------------------------------------ sidebar
        logo = QLabel()
        logo.setPixmap(theme.app_icon().pixmap(40, 40))
        brand = QLabel("RunnerFinder")
        brand.setObjectName("brand")
        sub = QLabel("Race photos by bib number")
        sub.setObjectName("brandSub")
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(brand)
        names.addWidget(sub)
        head = QHBoxLayout()
        head.setContentsMargins(14, 16, 10, 14)
        head.addWidget(logo)
        head.addLayout(names, 1)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setIconSize(QSize(18, 18))
        for name, glyph in zip(PAGE_NAMES, NAV_GLYPHS):
            self.nav.addItem(QListWidgetItem(theme.glyph_icon(glyph, size=36), name))

        self.theme_button = QPushButton()
        self.theme_button.setObjectName("themeButton")
        self.theme_button.setIcon(theme.glyph_icon("theme"))
        self.theme_button.clicked.connect(self.toggle_theme)
        foot = QLabel(f"Version {__version__}")
        foot.setObjectName("sidebarFoot")

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(212)
        sv = QVBoxLayout(sidebar)
        sv.setContentsMargins(0, 0, 0, 12)
        sv.setSpacing(6)
        sv.addLayout(head)
        sv.addWidget(self.nav, 1)
        tb = QHBoxLayout()
        tb.setContentsMargins(12, 0, 12, 0)
        tb.addWidget(self.theme_button)
        sv.addLayout(tb)
        fb = QHBoxLayout()
        fb.setContentsMargins(16, 0, 12, 0)
        fb.addWidget(foot)
        sv.addLayout(fb)

        # ------------------------------------------------------------ pages
        self.stack = QStackedWidget()
        self.stack.setObjectName("pages")
        for page in (self.dashboard_page, self.new_event_page, self.progress_page, self.search_page):
            page.setContentsMargins(26, 20, 26, 18)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(DASHBOARD)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(sidebar)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.mode_label = QLabel("")
        self.statusBar().addPermanentWidget(QLabel(f"RunnerFinder Desktop {__version__}"))
        self.statusBar().addWidget(self.mode_label)
        self._update_theme_button()

    def show_page(self, index: int) -> None:
        self.nav.setCurrentRow(index)

    def current_page(self) -> int:
        return self.stack.currentIndex()

    def set_mode_text(self, text: str) -> None:
        self.mode_label.setText(text)

    # ---------------------------------------------------------------- theme
    def toggle_theme(self) -> None:
        from PySide6.QtWidgets import QApplication
        name = theme.apply(QApplication.instance(), "light" if theme.current() == "dark" else "dark")
        theme.save(name)
        self._update_theme_button()
        self.search_page.refresh()                 # card colours follow the theme
        self.theme_changed.emit(name)

    def _update_theme_button(self) -> None:
        self.theme_button.setText("  Light theme" if theme.current() == "dark" else "  Dark theme")
        self.theme_button.setToolTip("Switch between the dark and the light theme (remembered)")
