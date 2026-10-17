"""Small reusable UI building blocks (Phase 8.2): page header, card, stat tile, chip."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


def page_header(title: str, subtitle: str = "", title_label: QLabel | None = None) -> QWidget:
    """Title + muted subtitle. title_label: an existing QLabel to use as the title (keeps its object)."""
    box = QWidget()
    v = QVBoxLayout(box)
    v.setContentsMargins(0, 0, 0, 4)
    v.setSpacing(0)
    t = title_label or QLabel(title)
    t.setText(title)
    t.setObjectName("pageTitle")
    v.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("pageSubtitle")
        s.setWordWrap(True)
        v.addWidget(s)
    return box


class Card(QFrame):
    """Rounded surface with an optional section title; add content to .body (a QVBoxLayout)."""

    def __init__(self, title: str = "", parent: QWidget | None = None, margins=(18, 14, 18, 16)):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(10)
        if title:
            t = QLabel(title)
            t.setObjectName("sectionTitle")
            self.body.addWidget(t)


class StatTile(QFrame):
    """Big value with a caption below; .value is the QLabel holding the value."""

    def __init__(self, caption: str, value: QLabel | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("tile")
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(2)
        self.value = value or QLabel("-")
        self.value.setObjectName("tileValue")
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cap = QLabel(caption)
        cap.setObjectName("tileCaption")
        v.addWidget(self.value)
        v.addWidget(cap)


def chip(text: str = "", state: str = "neutral") -> QLabel:
    lab = QLabel(text)
    lab.setProperty("chip", state)
    return lab


def field_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("fieldLabel")
    return lab


def hbox(*widgets, spacing: int = 10, stretch_last: bool = False) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setSpacing(spacing)
    for w in widgets:
        if w is None:
            h.addStretch(1)
        else:
            h.addWidget(w)
    if stretch_last:
        h.addStretch(1)
    return h
