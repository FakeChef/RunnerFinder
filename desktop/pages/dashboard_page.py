"""Dashboard (Phase 8.2): overview of the photo index - events, counts, GPU status, quick actions."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from desktop import theme
from desktop.services.models import EventSummary, GpuStatus
from desktop.widgets import Card, StatTile, chip, page_header

COLUMNS = ("Event", "Photos", "Analysed", "Failed", "Bib numbers", "Last analysis")


class DashboardPage(QWidget):
    new_event_requested = Signal()
    search_event_requested = Signal(str)          # event name
    add_photos_requested = Signal(str)            # event name: New Event with this name
    show_folder_requested = Signal(str)           # a folder analysed into the event

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("page")
        self._summaries: list[EventSummary] = []

        self.gpu_chip = chip("GPU: checking…", "neutral")
        self.new_button = QPushButton("  New Event")
        self.new_button.setObjectName("primary")
        self.new_button.setIcon(theme.glyph_icon("add", "#ffffff"))
        self.new_button.clicked.connect(self.new_event_requested)
        head = QHBoxLayout()
        head.addWidget(page_header("Dashboard", "Your race photo index: analysed events, photos and bib numbers."), 1)
        head.addWidget(self.gpu_chip, 0, Qt.AlignmentFlag.AlignTop)
        head.addSpacing(8)
        head.addWidget(self.new_button, 0, Qt.AlignmentFlag.AlignTop)

        self.events_tile = StatTile("Events")
        self.photos_tile = StatTile("Photos indexed")
        self.numbers_tile = StatTile("Bib numbers found")
        self.last_tile = StatTile("Last analysis")
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        for t in (self.events_tile, self.photos_tile, self.numbers_tile, self.last_tile):
            tiles.addWidget(t, 1)

        # events table / empty state
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for col in range(1, len(COLUMNS)):
            self.table.horizontalHeaderItem(col).setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(COLUMNS)):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.cellDoubleClicked.connect(lambda row, _col: self._emit_selected(self.search_event_requested))
        self.table.itemSelectionChanged.connect(self._update_buttons)

        self.empty = QLabel("No events yet.\n\nClick “New Event”, choose the photo folder of a camera card and "
                            "start the analysis.\nAfterwards you can search every runner by bib number.")
        self.empty.setObjectName("emptyState")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        self.table_stack = QStackedWidget()
        self.table_stack.addWidget(self.empty)
        self.table_stack.addWidget(self.table)

        self.search_button = QPushButton("  Search event")
        self.search_button.setIcon(theme.glyph_icon("search"))
        self.search_button.clicked.connect(lambda: self._emit_selected(self.search_event_requested))
        self.add_button = QPushButton("  Add photos…")
        self.add_button.setIcon(theme.glyph_icon("add"))
        self.add_button.setToolTip("New Event with this event name: only the new photos of a folder are analysed")
        self.add_button.clicked.connect(lambda: self._emit_selected(self.add_photos_requested))
        self.folder_button = QPushButton("  Show folder")
        self.folder_button.setIcon(theme.glyph_icon("folder"))
        self.folder_button.clicked.connect(self._show_folder)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("hint")
        self.detail_label.setWordWrap(True)
        actions = QHBoxLayout()
        actions.addWidget(self.detail_label, 1)
        for b in (self.folder_button, self.add_button, self.search_button):
            actions.addWidget(b)

        events_card = Card("Events")
        events_card.body.addWidget(self.table_stack, 1)
        events_card.body.addLayout(actions)

        self.message_label = QLabel("")
        self.message_label.setObjectName("message")
        self.message_label.setWordWrap(True)
        tip = QLabel("Tip: add every camera card to the same event name - only new photos are analysed. "
                     "Search with the digits of the bib (B60914 → 60914).")
        tip.setObjectName("hint")
        tip.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addLayout(head)
        layout.addLayout(tiles)
        layout.addWidget(events_card, 1)
        layout.addWidget(self.message_label)
        layout.addWidget(tip)
        self.show_summaries([])

    # ---------------------------------------------------------------- slots
    @Slot(list)
    def show_summaries(self, summaries: list) -> None:
        selected = self.selected_event()
        self._summaries = list(summaries)
        self.table.setRowCount(len(self._summaries))
        for row, s in enumerate(self._summaries):
            values = (s.name, f"{s.photos:,}", f"{s.analysed:,}", f"{s.failed:,}", f"{s.bib_numbers:,}",
                      s.last_analysis or "-")
            c = theme.colors()
            for col, v in enumerate(values):
                item = QTableWidgetItem(v)
                if col:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 3 and s.failed:
                    item.setForeground(QColor(c["bad"]))
                tip = "\n".join(s.folders) if s.folders else s.name
                if col == 1 and s.videos:                   # V1.3: videos are not counted as photos
                    tip = f"{s.photos:,} photos · {s.videos:,} videos"
                if col == 2 and s.pending:                  # not complete: photos waiting
                    item.setForeground(QColor(c["warn"]))
                    tip = f"{s.pending:,} photos waiting - start the folder again with the same event name to continue"
                item.setToolTip(tip)
                self.table.setItem(row, col, item)
            if s.name == selected:
                self.table.selectRow(row)
        self.table_stack.setCurrentIndex(1 if self._summaries else 0)
        self.events_tile.value.setText(f"{len(self._summaries):,}")
        self.photos_tile.value.setText(f"{sum(s.photos for s in self._summaries):,}")
        self.numbers_tile.value.setText(f"{sum(s.bib_numbers for s in self._summaries):,}")
        last = max((s.last_analysis for s in self._summaries if s.last_analysis), default="")
        self.last_tile.value.setText(last[:10] if last else "-")
        self.last_tile.value.setToolTip(last)
        if self._summaries and self.table.currentRow() < 0:
            self.table.selectRow(0)
        self._update_buttons()

    @Slot(object)
    def show_gpu_status(self, status: GpuStatus) -> None:
        if status.available:
            text = f"GPU: {status.name.replace('NVIDIA GeForce ', '')} ✓" + (f"  ·  CUDA {status.cuda}" if status.cuda else "")
            theme.set_chip(self.gpu_chip, "ok")
        else:
            text = "GPU: not available"
            theme.set_chip(self.gpu_chip, "bad" if status.detail else "neutral")
        self.gpu_chip.setText(text)
        self.gpu_chip.setToolTip(status.detail or status.name)

    @Slot(str)
    def show_message(self, text: str) -> None:
        self.message_label.setText(text)

    # ---------------------------------------------------------------- helpers
    def selected_event(self) -> str | None:
        row = self.table.currentRow()
        return self._summaries[row].name if 0 <= row < len(self._summaries) and self.table.selectedItems() else None

    def _emit_selected(self, signal) -> None:
        name = self.selected_event()
        if name:
            signal.emit(name)

    def _show_folder(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._summaries) and self._summaries[row].folders:
            self.show_folder_requested.emit(self._summaries[row].folders[-1])

    def _update_buttons(self) -> None:
        row = self.table.currentRow()
        s = self._summaries[row] if 0 <= row < len(self._summaries) and self.table.selectedItems() else None
        self.search_button.setEnabled(s is not None)
        self.add_button.setEnabled(s is not None)
        self.folder_button.setEnabled(bool(s and s.folders))
        if s:
            parts = [f"{len(s.folders)} folder{'s' if len(s.folders) != 1 else ''} analysed"] if s.folders else []
            if s.missing:
                parts.append(f"{s.missing:,} originals not found")
            if s.pending:
                parts.append(f"{s.pending:,} photos waiting - start the folder again to continue")
            if s.videos:                                   # V1.3
                parts.append(f"{s.videos:,} videos: {s.videos_verified:,} with a bib, {s.videos_ambiguous:,} ambiguous, "
                             f"{s.videos_unindexed:,} not indexed"
                             + (f", {s.videos_pending:,} waiting" if s.videos_pending else "")
                             + (f", {s.videos_failed:,} unreadable" if s.videos_failed else ""))
            self.detail_label.setText(" · ".join(parts))
        else:
            self.detail_label.setText("")
