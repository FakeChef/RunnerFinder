"""New Event: photo folder, event name, GPU, photo count, estimate, Start (Phase 8.2 layout)."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QVBoxLayout, QWidget)

from desktop import theme
from desktop.services.models import FolderInfo, GpuStatus, format_duration
from desktop.widgets import Card, field_label, page_header


class NewEventPage(QWidget):
    folder_selected = Signal(str)                 # a folder was chosen -> inspect it
    start_requested = Signal(str, str)            # folder, event name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("page")
        self._gpu_ok = False
        self._allow_without_gpu = False           # True only in demo mode
        self._photo_count: int | None = None

        # 1. photo folder
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText(r"for example D:\RacePhotos\Race\C1  (one camera card)")
        self.folder_edit.editingFinished.connect(self._folder_edited)
        self.browse_button = QPushButton("  Browse…")
        self.browse_button.setIcon(theme.glyph_icon("folder"))
        self.browse_button.clicked.connect(self._browse)
        self.count_label = QLabel("-")
        self.count_label.setObjectName("tileValue")
        self.estimate_label = QLabel("-")
        self.estimate_label.setObjectName("tileValue")
        folder_card = Card("1   Photo folder")
        row = QHBoxLayout()
        row.addWidget(self.folder_edit, 1)
        row.addWidget(self.browse_button)
        folder_card.body.addLayout(row)
        stats = QGridLayout()
        stats.setHorizontalSpacing(40)
        stats.addWidget(field_label("Photos found"), 0, 0)
        stats.addWidget(field_label("Estimated time"), 0, 1)
        stats.addWidget(self.count_label, 1, 0)
        stats.addWidget(self.estimate_label, 1, 1)
        stats.setColumnStretch(2, 1)
        folder_card.body.addLayout(stats)
        hint = QLabel("Sub-folders are included. Photos and videos (MP4, MOV, M4V) are only read - never changed, "
                      "moved or copied.")
        hint.setObjectName("hint")
        folder_card.body.addWidget(hint)

        # 2. event
        self.event_edit = QLineEdit()
        self.event_edit.setPlaceholderText("for example 2026 Race")
        self.event_edit.textChanged.connect(lambda _text: self._update_start())
        event_card = Card("2   Event name")
        event_card.body.addWidget(self.event_edit)
        hint2 = QLabel("Use the same name for every camera card of a race: photos already analysed are skipped, "
                       "only new photos are analysed.")
        hint2.setObjectName("hint")
        hint2.setWordWrap(True)
        event_card.body.addWidget(hint2)

        # 3. system
        self.gpu_label = QLabel("checking…")
        self.cuda_label = QLabel("-")
        self.paddle_label = QLabel("-")
        system_card = Card("System")
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        for r, (name, lab) in enumerate((("GPU", self.gpu_label), ("CUDA", self.cuda_label),
                                         ("Paddle", self.paddle_label))):
            grid.addWidget(field_label(name), r, 0)
            grid.addWidget(lab, r, 1)
        grid.setColumnStretch(1, 1)
        system_card.body.addLayout(grid)

        self.message_label = QLabel("")
        self.message_label.setObjectName("message")
        self.message_label.setWordWrap(True)

        self.start_button = QPushButton("  Start Analysis")
        self.start_button.setObjectName("primary")
        self.start_button.setIcon(theme.glyph_icon("play", "#ffffff"))
        self.start_button.setMinimumHeight(40)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        left.addWidget(folder_card)
        left.addWidget(event_card)
        left.addStretch(1)
        right = QVBoxLayout()
        right.addWidget(system_card)
        right.addStretch(1)
        columns.addLayout(left, 3)
        columns.addLayout(right, 2)

        bottom = QHBoxLayout()
        bottom.addWidget(self.message_label, 1)
        bottom.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignRight)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addWidget(page_header("New Event", "Analyse the photos of one camera card. "
                                                  "The bib numbers are read on the NVIDIA GPU."))
        layout.addLayout(columns, 1)
        layout.addLayout(bottom)

    # ---------------------------------------------------------------- user actions
    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select race photo folder", self.folder_edit.text())
        if folder:
            self.set_folder(folder)

    def _folder_edited(self) -> None:
        text = self.folder_edit.text().strip().strip('"')
        if text:
            self.set_folder(text)

    def set_folder(self, folder: str) -> None:
        folder = os.path.normpath(folder)
        self.folder_edit.setText(folder)
        if not self.event_edit.text().strip():
            self.event_edit.setText(os.path.basename(folder.rstrip("\\/")))
        self._photo_count = None
        self.count_label.setText("counting…")
        self.estimate_label.setText("-")
        self._update_start()
        self.folder_selected.emit(folder)

    def set_event_name(self, name: str) -> None:
        """Dashboard 'Add photos…': the next folder goes into this event."""
        self.event_edit.setText(name)
        self.event_edit.setFocus()

    def _start(self) -> None:
        self.start_requested.emit(self.folder_edit.text().strip(), self.event_edit.text().strip())

    # ---------------------------------------------------------------- slots (from the service)
    @Slot(object)
    def show_gpu_status(self, status: GpuStatus) -> None:
        self._gpu_ok = status.available
        self.gpu_label.setText(f"{status.name} ✓" if status.available else f"not available — {status.detail}")
        self.cuda_label.setText(f"{status.cuda} ✓" if status.available and status.cuda else "-")
        self.paddle_label.setText(f"available ({status.paddle})" if status.paddle else "-")
        for lab in (self.gpu_label, self.cuda_label, self.paddle_label):
            theme.set_state(lab, "ok" if status.available else ("bad" if status.detail and lab is self.gpu_label else ""))
        self._update_start()

    @Slot(object)
    def show_folder_info(self, info: FolderInfo) -> None:
        if os.path.normcase(info.folder) != os.path.normcase(self.folder_edit.text().strip()):
            return                                         # answer for an older selection
        self._photo_count = info.photo_count + info.video_count    # V1.3: videos are analysed too
        self.count_label.setText(f"{info.photo_count:,}" + (f" photos · {info.video_count:,} videos"
                                                             if info.video_count else ""))
        self.estimate_label.setText(f"about {format_duration(info.estimated_seconds)} (GPU)"
                                    if self._photo_count else "-")
        self._update_start()

    @Slot(str)
    def show_message(self, text: str) -> None:
        self.message_label.setText(text)

    def allow_without_gpu(self, allowed: bool) -> None:
        """Demo mode only: the simulated analysis does not need the GPU."""
        self._allow_without_gpu = allowed
        self._update_start()

    def set_running(self, running: bool) -> None:
        for w in (self.folder_edit, self.browse_button, self.event_edit):
            w.setEnabled(not running)
        self._update_start(running)

    def _update_start(self, running: bool = False) -> None:
        ok = (not running and bool(self.folder_edit.text().strip()) and bool(self.event_edit.text().strip())
              and bool(self._photo_count) and (self._gpu_ok or self._allow_without_gpu))
        self.start_button.setEnabled(ok)
