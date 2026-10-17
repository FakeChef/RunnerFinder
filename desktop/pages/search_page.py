"""Search: event, bib number, thumbnail results, preview, open photo / file location, reject / restore.

V1.3: the same list also shows the event's VIDEOS (accepted video bibs; ambiguous video candidates only
with "Show ambiguous videos"); every card and the details panel say PHOTO or VIDEO.

Phase 8.2 layout: big search field, recent searches, thumbnail size, preview panel, keyboard shortcuts.
Keys: Ctrl+F search field · Enter / Ctrl+O open photo · Ctrl+E show in folder · Del reject · Ctrl+Z restore.
"""
from __future__ import annotations

import os
import re

from PySide6.QtCore import QSize, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QIcon, QKeySequence, QPixmap, QShortcut, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListView,
                               QPushButton, QSizePolicy, QSlider, QSplitter, QStackedWidget, QVBoxLayout, QWidget)

from desktop import theme
from desktop.services.models import SearchResult
from desktop.widgets import Card, field_label, page_header

THUMB = QSize(200, 200)                      # default thumbnail size in the grid
THUMB_MIN, THUMB_MAX = 120, 320
PATH_ROLE = Qt.ItemDataRole.UserRole + 1
RESULT_ROLE = Qt.ItemDataRole.UserRole + 2
MAX_RECENT = 8
# V1.3.1 search input (same form as search.db_search.parse_query; the GUI page imports no engine code)
BIB_QUERY = re.compile(r"[A-Z]{0,3}[\s\-_.#·:/]*\d+[\s\-_.#·:/]*[A-Z]{0,3}")


class SearchPage(QWidget):
    search_requested = Signal(str, str)          # event, bib (digits)
    open_photo_requested = Signal(str)           # original photo path
    show_in_folder_requested = Signal(str)       # original photo path
    reject_requested = Signal(object)            # SearchResult
    restore_requested = Signal(object)           # SearchResult

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("page")

        # ------------------------------------------------------------ search bar
        self.event_combo = QComboBox()
        self.event_combo.setMinimumWidth(240)
        self.event_combo.setPlaceholderText("no events yet")
        self.bib_edit = QLineEdit()
        self.bib_edit.setObjectName("bigSearch")
        self.bib_edit.setPlaceholderText("Bib number - digits only, e.g. 60914")
        self.bib_edit.setClearButtonEnabled(True)
        self.bib_edit.addAction(theme.glyph_icon("search"), QLineEdit.ActionPosition.LeadingPosition)
        self.bib_edit.returnPressed.connect(self._search)
        self.bib_edit.textChanged.connect(self._update_buttons)
        self.search_button = QPushButton("Search")
        self.search_button.setObjectName("primary")
        self.search_button.setMinimumHeight(44)
        self.search_button.setMinimumWidth(110)
        self.search_button.clicked.connect(self._search)
        bar = Card(margins=(16, 12, 16, 12))
        row = QHBoxLayout()
        row.setSpacing(10)
        event_box = QVBoxLayout()
        event_box.setSpacing(2)
        event_box.addWidget(field_label("Event"))
        event_box.addWidget(self.event_combo)
        row.addLayout(event_box)
        bib_box = QVBoxLayout()
        bib_box.setSpacing(2)
        bib_box.addWidget(field_label("Search Bib"))
        bib_box.addWidget(self.bib_edit)
        row.addLayout(bib_box, 1)
        row.addWidget(self.search_button, 0, Qt.AlignmentFlag.AlignBottom)
        bar.body.addLayout(row)

        self.recent_box = QHBoxLayout()
        self.recent_box.setSpacing(6)
        self.recent_label = field_label("Recent:")
        self.recent_box.addWidget(self.recent_label)
        self.recent_box.addStretch(1)
        self._recent: list[str] = []
        self._recent_buttons: list[QPushButton] = []
        bar.body.addLayout(self.recent_box)

        # ------------------------------------------------------------ options
        self.show_filtered_check = QCheckBox("Show filtered candidates (with reason)")
        self.show_filtered_check.setToolTip("Numbers the noise filter hides, e.g. bag-check numbers, 10KM, 2026")
        self.show_rejected_check = QCheckBox("Show rejected")
        self.show_ambiguous_check = QCheckBox("Show ambiguous videos")
        self.show_ambiguous_check.setToolTip("Videos where several runners or conflicting numbers were read: "
                                             "shown as unconfirmed candidates, never as matches")
        self.show_ambiguous_check.toggled.connect(self._options_changed)
        self.show_filtered_check.toggled.connect(self._options_changed)
        self.show_rejected_check.toggled.connect(self._options_changed)
        self.hidden_label = QLabel("")
        self.hidden_label.setObjectName("message")
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(THUMB_MIN, THUMB_MAX)
        self.size_slider.setValue(THUMB.width())
        self.size_slider.setFixedWidth(120)
        self.size_slider.setToolTip("Thumbnail size")
        self.size_slider.valueChanged.connect(self._set_thumb_size)
        options = QHBoxLayout()
        options.addWidget(self.show_filtered_check)
        options.addSpacing(8)
        options.addWidget(self.show_rejected_check)
        options.addSpacing(8)
        options.addWidget(self.show_ambiguous_check)
        options.addSpacing(12)
        options.addWidget(self.hidden_label)
        options.addStretch(1)
        options.addWidget(field_label("Size"))
        options.addWidget(self.size_slider)

        # ------------------------------------------------------------ results grid
        self.result_label = QLabel("Enter a bib number.")
        self.result_label.setObjectName("sectionTitle")
        self.model = QStandardItemModel(self)
        self.results = QListView()
        self.results.setObjectName("results")
        self.results.setModel(self.model)
        self.results.setViewMode(QListView.ViewMode.IconMode)
        self.results.setResizeMode(QListView.ResizeMode.Adjust)
        self.results.setUniformItemSizes(True)          # fast layout for many results
        self.results.setMovement(QListView.Movement.Static)
        self.results.setWordWrap(True)
        self.results.setSpacing(4)
        self.results.doubleClicked.connect(lambda idx: self.open_photo_requested.emit(idx.data(PATH_ROLE)))
        self.results.activated.connect(lambda idx: self.open_photo_requested.emit(idx.data(PATH_ROLE)))
        self.results.selectionModel().selectionChanged.connect(self._update_buttons)
        self.empty_label = QLabel()
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.results_stack = QStackedWidget()
        self.results_stack.addWidget(self.empty_label)
        self.results_stack.addWidget(self.results)
        results_box = QWidget()
        rv = QVBoxLayout(results_box)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(self.result_label)
        rv.addWidget(self.results_stack, 1)

        # ------------------------------------------------------------ preview panel
        self.preview = QLabel("No photo selected")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(260, 220)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_title = QLabel("")
        self.preview_title.setObjectName("previewTitle")
        self.preview_title.setWordWrap(True)
        self.detail_type = QLabel("-")
        self.detail_bibs = QLabel("-")
        self.detail_event = QLabel("-")
        self.detail_status = QLabel("-")
        self.detail_status.setWordWrap(True)
        self.detail_path = QLabel("-")
        self.detail_path.setWordWrap(True)
        self.detail_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail_path.setObjectName("hint")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        for r, (name, lab) in enumerate((("Type", self.detail_type), ("Bib", self.detail_bibs),
                                         ("Event", self.detail_event),
                                         ("Status", self.detail_status), ("File", self.detail_path))):
            grid.addWidget(field_label(name), r, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(lab, r, 1)
        grid.setColumnStretch(1, 1)

        self.open_button = QPushButton("  Open Photo")
        self.open_button.setObjectName("primary")
        self.open_button.setIcon(theme.glyph_icon("open", "#ffffff"))
        self.open_button.setToolTip("Open the original photo (Enter, Ctrl+O)")
        self.open_button.clicked.connect(lambda: self._emit_for_selection(self.open_photo_requested))
        self.folder_button = QPushButton("  Show in Folder")
        self.folder_button.setIcon(theme.glyph_icon("folder"))
        self.folder_button.setToolTip("Show the original in Windows Explorer (Ctrl+E)")
        self.folder_button.clicked.connect(lambda: self._emit_for_selection(self.show_in_folder_requested))
        self.reject_button = QPushButton("  Reject")
        self.reject_button.setObjectName("danger")
        self.reject_button.setIcon(theme.glyph_icon("reject"))
        self.reject_button.setToolTip("This photo does not show this bib: hide it from the results (Del)")
        self.reject_button.clicked.connect(lambda: self._emit_result(self.reject_requested))
        self.restore_button = QPushButton("  Restore")
        self.restore_button.setIcon(theme.glyph_icon("restore"))
        self.restore_button.setToolTip("Undo a reject (tick 'Show rejected' to see rejected photos) (Ctrl+Z)")
        self.restore_button.clicked.connect(lambda: self._emit_result(self.restore_requested))
        keys = QLabel("Enter open · Ctrl+E folder · Del reject · Ctrl+Z restore · Ctrl+F search")
        keys.setObjectName("hint")
        keys.setWordWrap(True)

        panel = Card(margins=(14, 14, 14, 14))
        panel.setMinimumWidth(360)
        panel.body.addWidget(self.preview, 1)
        panel.body.addWidget(self.preview_title)
        panel.body.addLayout(grid)
        b1 = QHBoxLayout()
        b1.addWidget(self.open_button, 1)
        b1.addWidget(self.folder_button, 1)
        b2 = QHBoxLayout()
        b2.addWidget(self.reject_button, 1)
        b2.addWidget(self.restore_button, 1)
        panel.body.addLayout(b1)
        panel.body.addLayout(b2)
        panel.body.addWidget(keys)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(results_box)
        self.splitter.addWidget(panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([680, 370])
        self.splitter.setChildrenCollapsible(False)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.addWidget(page_header("Search", "Find a runner's photos and videos by bib number."))
        layout.addWidget(bar)
        layout.addLayout(options)
        layout.addWidget(self.splitter, 1)

        # keyboard
        QShortcut(QKeySequence.StandardKey.Find, self, activated=self.focus_search)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=lambda: self._emit_for_selection(self.open_photo_requested))
        QShortcut(QKeySequence("Ctrl+E"), self,
                  activated=lambda: self._emit_for_selection(self.show_in_folder_requested))
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self.results,
                  activated=lambda: self.reject_button.isEnabled() and self._emit_result(self.reject_requested))
        QShortcut(QKeySequence.StandardKey.Undo, self.results,
                  activated=lambda: self.restore_button.isEnabled() and self._emit_result(self.restore_requested))

        self._placeholder = self._make_placeholder()
        self._last_query: tuple[str, str] | None = None
        self._keep_key: tuple | None = None
        self._set_thumb_size(THUMB.width())
        self._show_empty("Type a bib number and press Enter.\n\nThe runner's photos appear here as thumbnails; "
                         "select one to see it larger and open the original.")
        self._update_buttons()

    # ---------------------------------------------------------------- user actions
    def focus_search(self) -> None:
        self.bib_edit.setFocus()
        self.bib_edit.selectAll()

    def _search(self) -> None:
        bib = self.bib_edit.text().strip()
        # V1.3.1: digits ("0175") or a full bib ("A0175", "A-0175"); never partial matching
        if not BIB_QUERY.fullmatch(bib.upper()):
            self.result_label.setText("Please enter a bib number, e.g. 0175 or A0175 (exact match; "
                                      "leading zeros are kept).")
            return
        self.result_label.setText("Searching…")
        self._last_query = (self.event_combo.currentText(), bib)
        self._remember(bib)
        self.search_requested.emit(self.event_combo.currentText(), bib)

    def _remember(self, bib: str) -> None:
        if bib in self._recent:
            self._recent.remove(bib)
        self._recent.insert(0, bib)
        del self._recent[MAX_RECENT:]
        for b in self._recent_buttons:
            self.recent_box.removeWidget(b)
            b.deleteLater()
        self._recent_buttons = []
        for i, value in enumerate(self._recent):
            b = QPushButton(value)
            b.setObjectName("chipButton")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(f"Search {value} again")
            b.clicked.connect(lambda _c=False, v=value: self._search_again(v))
            self.recent_box.insertWidget(1 + i, b)
            self._recent_buttons.append(b)

    def _search_again(self, bib: str) -> None:
        self.bib_edit.setText(bib)
        self._search()

    def recent_searches(self) -> list[str]:
        return list(self._recent)

    def _options_changed(self, *_args) -> None:
        if self._last_query:                       # same search again with the new options
            self.refresh()

    def refresh(self) -> None:
        """Repeat the last search (after a reject/restore or an option change); keep the selection."""
        if self._last_query:
            result = self.selected_result()
            self._keep_key = self._key(result) if result else None
            self.search_requested.emit(*self._last_query)

    @property
    def show_filtered(self) -> bool:
        return self.show_filtered_check.isChecked()

    @property
    def show_rejected(self) -> bool:
        return self.show_rejected_check.isChecked()

    @property
    def show_ambiguous(self) -> bool:
        return self.show_ambiguous_check.isChecked()

    @staticmethod
    def _key(r: SearchResult) -> tuple:
        return ("VIDEO", r.video_id, r.bib) if r.is_video else ("PHOTO", r.photo_id)

    def _selected_path(self) -> str | None:
        idx = self.results.selectionModel().selectedIndexes()
        return idx[0].data(PATH_ROLE) if idx else None

    def selected_result(self) -> SearchResult | None:
        idx = self.results.selectionModel().selectedIndexes()
        return idx[0].data(RESULT_ROLE) if idx else None

    def _emit_result(self, signal) -> None:
        result = self.selected_result()
        if result is not None:
            signal.emit(result)

    def _emit_for_selection(self, signal) -> None:
        path = self._selected_path()
        if path:
            signal.emit(path)

    def _set_thumb_size(self, size: int) -> None:
        self.results.setIconSize(QSize(size, size))
        self.results.setGridSize(QSize(size + 40, size + 84))

    def _update_buttons(self, *_args) -> None:
        result = self.selected_result()
        has = self._selected_path() is not None
        self.open_button.setEnabled(has and not (result and result.missing))
        self.folder_button.setEnabled(has and not (result and result.missing))
        reviewable = bool(result and (result.photo_id is not None or result.video_id is not None))
        self.reject_button.setEnabled(reviewable and bool(result.active_bibs))
        self.restore_button.setEnabled(reviewable and bool(result.rejected_bibs))
        video = bool(result and result.is_video)
        self.open_button.setText("  Open Video" if video else "  Open Photo")
        self.open_button.setToolTip(("Play the original video in the Windows default player" if video
                                     else "Open the original photo") + " (Enter, Ctrl+O)")
        self.search_button.setEnabled(bool(self.bib_edit.text().strip()))
        self._show_preview(result)

    def _show_preview(self, r: SearchResult | None) -> None:
        if r is None:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No photo selected")
            self.preview_title.setText("")
            for lab in (self.detail_type, self.detail_bibs, self.detail_event, self.detail_status, self.detail_path):
                lab.setText("-")
            theme.set_state(self.detail_status, "")
            return
        pix = QPixmap(r.thumbnail_path) if r.thumbnail_path else QPixmap()
        if pix.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No preview" + (" - original not found" if r.missing else ""))
        else:
            box = self.preview.size()
            self.preview.setPixmap(pix.scaled(max(box.width() - 8, 120), max(box.height() - 8, 120),
                                              Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))
        self.preview_title.setText(r.file_name)
        self.detail_type.setText("VIDEO (representative frame)" if r.is_video else "PHOTO")
        self.detail_bibs.setText(", ".join(r.bibs) or r.bib)
        self.detail_event.setText(r.event)
        if r.missing:
            status, state = "Original not found", "bad"
        elif r.is_video and r.rejected_bibs and not r.active_bibs:
            status, state = f"Rejected ({', '.join(r.rejected_bibs)})", "bad"
        elif r.is_video and not r.confirmed:
            status, state = (f"AMBIGUOUS video - unconfirmed candidate (read in {r.votes} of {r.frames} frames; "
                             "several runners or conflicting numbers)"), "warn"
        elif r.is_video:
            status, state = f"Found in the video (read in {r.votes} of {r.frames} sampled frames)", "ok"
        elif r.rejected_bibs and not r.active_bibs:
            status, state = f"Rejected ({', '.join(r.rejected_bibs)})", "bad"
        elif r.filtered:
            status, state = f"Filtered: {r.filter_reason}", "warn"
        elif r.match == "normalized":
            status, state = f"Normalized match: OCR read “{r.matched_text}”", "warn"
        elif r.match == "alias":
            status, state = (f"Leading-zero alias: bib {', '.join(r.bibs) or r.bib} "
                             "(not the exact number searched)"), "warn"
        elif r.low_confidence:
            status, state = f"Low confidence: {r.low_confidence}", "warn"
        else:
            status, state = "Found by the analysis", "ok"
        self.detail_status.setText(status)
        theme.set_state(self.detail_status, state)
        self.detail_path.setText(os.path.dirname(r.file_path))

    def resizeEvent(self, event) -> None:  # keep the preview sharp at the new size
        super().resizeEvent(event)
        self._show_preview(self.selected_result())

    def _show_empty(self, text: str) -> None:
        self.empty_label.setText(text)
        self.results_stack.setCurrentIndex(0)

    # ---------------------------------------------------------------- slots
    @Slot(list)
    def set_events(self, names: list) -> None:
        current = self.event_combo.currentText()
        self.event_combo.clear()
        self.event_combo.addItems(names)
        if current in names:
            self.event_combo.setCurrentText(current)

    def select_event(self, name: str) -> None:
        if name and self.event_combo.findText(name) < 0:
            self.event_combo.addItem(name)
        self.event_combo.setCurrentText(name)

    @Slot(str, str, list)
    def show_results(self, event: str, bib: str, results: list) -> None:
        self.model.clear()
        for r in results:
            self.model.appendRow(self._item(r))
        if self._keep_key is not None:             # re-select the card after a refresh
            for row in range(self.model.rowCount()):
                r = self.model.item(row).data(RESULT_ROLE)
                if r is not None and self._key(r) == self._keep_key:
                    self.results.setCurrentIndex(self.model.index(row, 0))
                    break
            self._keep_key = None
        if results:
            n_v = sum(1 for r in results if r.is_video)
            n_p = len(results) - n_v
            parts = [f"{n_p} photo{'s' if n_p != 1 else ''}"]
            if n_v:
                parts.append(f"{n_v} video{'s' if n_v != 1 else ''}")
            self.result_label.setText(f"Results: {', '.join(parts)} for bib {bib} in {event}")
            self.results_stack.setCurrentIndex(1)
        else:
            self.result_label.setText(f"No photos or videos found for bib {bib}"
                                      + (f" in {event}." if event else "."))
            self._show_empty(f"No photos or videos found for bib {bib}.\n\nCheck the number, or tick “Show filtered "
                             "candidates” to see numbers the filter hides (distances, years, bag-check numbers).")
        self._update_buttons()

    @Slot(str)
    def show_message(self, text: str) -> None:
        self.result_label.setText(text)

    @Slot(int)
    def show_hidden_count(self, n: int) -> None:
        self.hidden_label.setText(f"{n} filtered candidate{'s' if n != 1 else ''} hidden"
                                  if n and not self.show_filtered else "")

    def _item(self, r: SearchResult) -> QStandardItem:
        pix = QPixmap(r.thumbnail_path) if r.thumbnail_path else QPixmap()
        icon = QIcon(pix.scaled(THUMB_MAX, THUMB_MAX, Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)) if not pix.isNull() else self._placeholder
        lines = [r.file_name, ("▶ VIDEO · " if r.is_video else "PHOTO · ") + f"{', '.join(r.bibs) or r.bib} · {r.event}"]
        if r.is_video:
            lines.append(f"read in {r.votes}/{r.frames} frames" if r.confirmed
                         else f"⚠ ambiguous - unconfirmed ({r.votes}/{r.frames})")
        if r.filtered:
            lines.append(f"⚠ filtered: {r.filter_reason}")
        if r.match == "normalized":
            lines.append(f"≈ OCR read {r.matched_text}")
        elif r.match == "alias":
            lines.append(f"≈ leading-zero alias of {r.bib}")
        if r.low_confidence:
            lines.append(f"? low confidence: {r.low_confidence}")
        if r.rejected_bibs and not r.active_bibs:
            lines.append("✕ rejected")
        if r.missing:
            lines.append("original not found")
        item = QStandardItem(icon, "\n".join(lines))
        item.setData(r.file_path, PATH_ROLE)
        item.setData(r, RESULT_ROLE)
        item.setToolTip(f"{r.file_path}\nbib {', '.join(r.bibs) or r.bib} · {r.event}"
                        + (f"\nfilter: {r.filter_reason}" if r.filtered else "")
                        + (f"\nrejected: {', '.join(r.rejected_bibs)}" if r.rejected_bibs else ""))
        c = theme.colors()
        if r.missing:
            item.setForeground(QBrush(QColor(c["bad"])))
        elif r.rejected_bibs and not r.active_bibs:
            item.setForeground(QBrush(QColor(c["rejected"])))
        elif r.filtered or r.low_confidence or r.match == "alias" or (r.is_video and not r.confirmed):
            item.setForeground(QBrush(QColor(c["warn"])))
        item.setEditable(False)
        return item

    def _make_placeholder(self) -> QIcon:
        pix = QPixmap(THUMB)
        pix.fill(QColor(theme.colors()["surface2"]))
        return QIcon(pix)
