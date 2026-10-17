"""Analysis Progress: replaces the console output of the analysis (Phase 8.2 layout)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
                               QWidget)

from desktop import theme
from desktop.services.models import AnalysisSummary, ProgressInfo, format_duration
from desktop.widgets import Card, StatTile, field_label, page_header


class ProgressPage(QWidget):
    cancel_requested = Signal()
    open_search_requested = Signal(str)          # event name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("page")
        self._event = ""

        self.event_label = QLabel("No analysis running.")
        self.event_label.setObjectName("eventName")
        self.status_label = QLabel("-")
        self.status_label.setObjectName("message")
        self.status_label.setWordWrap(True)
        self.percent_label = QLabel("")
        self.percent_label.setObjectName("bigPercent")
        self.percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.valueChanged.connect(lambda v: self.percent_label.setText(f"{v} %" if v or self._event else ""))

        # key figures
        self.photos_label = QLabel("-")
        self.speed_label = QLabel("-")
        self.time_label = QLabel("-")
        self.failed_label = QLabel("-")
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        for caption, lab in (("Processed", self.photos_label), ("Speed", self.speed_label),
                             ("Time remaining", self.time_label), ("Failed", self.failed_label)):
            tiles.addWidget(StatTile(caption, lab), 1)

        # details
        self.folder_label = QLabel("-")
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.total_label = QLabel("-")
        self.file_label = QLabel("-")
        self.remaining_photos_label = QLabel("-")
        self.duration_label = QLabel("-")
        self.gpu_label = QLabel("-")
        details = Card("Details")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        rows = (("Folder", self.folder_label), ("Total photos", self.total_label),
                ("Current photo", self.file_label), ("Remaining photos", self.remaining_photos_label),
                ("Duration", self.duration_label), ("GPU", self.gpu_label))
        for r, (name, lab) in enumerate(rows):
            grid.addWidget(field_label(name), r, 0)
            grid.addWidget(lab, r, 1)
        grid.setColumnStretch(1, 1)
        details.body.addLayout(grid)

        self.current_label = QLabel("")
        self.current_label.setObjectName("message")
        self.current_label.setWordWrap(True)

        self.cancel_button = QPushButton("  Cancel")
        self.cancel_button.setIcon(theme.glyph_icon("cancel"))
        self.cancel_button.setToolTip("Stops after the current photo; finished photos are kept. "
                                      "Start the same folder and event again to continue.")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.search_button = QPushButton("  Open Search")
        self.search_button.setObjectName("primary")
        self.search_button.setIcon(theme.glyph_icon("search", "#ffffff"))
        self.search_button.setMinimumHeight(38)
        self.search_button.setEnabled(False)
        self.search_button.clicked.connect(lambda: self.open_search_requested.emit(self._event))
        buttons = QHBoxLayout()
        buttons.addWidget(self.current_label, 1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.search_button)

        status_card = Card()
        top = QHBoxLayout()
        names = QVBoxLayout()
        names.setSpacing(2)
        names.addWidget(self.event_label)
        names.addWidget(self.status_label)
        top.addLayout(names, 1)
        top.addWidget(self.percent_label)
        status_card.body.addLayout(top)
        status_card.body.addWidget(self.bar)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addWidget(page_header("Analysis Progress", "The analysis runs in its own process: you can keep "
                                                          "using RunnerFinder, and Cancel is always safe."))
        layout.addWidget(status_card)
        layout.addLayout(tiles)
        layout.addWidget(details)
        layout.addStretch(1)
        layout.addLayout(buttons)

    def set_gpu(self, text: str) -> None:
        self.gpu_label.setText(text)

    @Slot(object)
    def show_progress(self, p: ProgressInfo) -> None:
        self._event = p.event
        self.event_label.setText(f"Analysing: {p.event}")
        if p.folder:
            self.folder_label.setText(p.folder)
        self.total_label.setText(f"{p.total:,}" if p.total else "-")
        self.photos_label.setText(f"{p.done:,} / {p.total:,}")
        self.remaining_photos_label.setText(f"{max(0, p.total - p.done):,}")
        unit = "video" if p.status.startswith("Analysing videos") else "photo"     # V1.3 video stage
        self.speed_label.setText(f"{1 / p.photos_per_second:.2f} sec/{unit}" if p.photos_per_second else "-")
        self.file_label.setText(p.current_file or "-")
        self.failed_label.setText("-")
        theme.set_state(self.failed_label, "")
        self.duration_label.setText("-")
        self.time_label.setText(format_duration(p.remaining_seconds))
        self.status_label.setText(p.status)
        self.current_label.setText("")
        self.bar.setValue(p.percent)
        self.percent_label.setText(f"{p.percent} %")
        self.cancel_button.setEnabled(True)
        self.search_button.setEnabled(False)

    @Slot(object)
    def show_summary(self, s: AnalysisSummary) -> None:
        self._event = s.event
        words = {"done": "Analysis completed", "cancelled": "Cancelled", "error": "Error"}
        self.event_label.setText(f"{words.get(s.status, s.status)}: {s.event}")
        self.photos_label.setText(f"{s.processed:,} / {s.total:,}")
        self.remaining_photos_label.setText(f"{max(0, s.total - s.processed):,}")
        self.failed_label.setText(f"{s.failed:,}")
        theme.set_state(self.failed_label, "bad" if s.failed else "ok")
        self.duration_label.setText(format_duration(s.elapsed_seconds))
        self.time_label.setText("-")
        self.file_label.setText("-")
        self.status_label.setText(words.get(s.status, s.status)
                                  + (f" — {s.skipped:,} already analysed (skipped)" if s.skipped else ""))
        self.current_label.setText(s.message)
        if s.status == "done":
            self.bar.setValue(100)
        self.percent_label.setText({"done": "✓ 100 %", "cancelled": f"{self.bar.value()} %",
                                    "error": "!"}.get(s.status, ""))
        self.cancel_button.setEnabled(False)
        self.search_button.setEnabled(s.status != "error")
