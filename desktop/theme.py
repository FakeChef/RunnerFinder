"""Visual design of the desktop application (Phase 8.2): dark (default) and light theme.

Fusion style + a QPalette (dialogs, native controls) + one generated style sheet.
The choice is remembered with QSettings ("ui/theme"). Glyph icons come from the
Windows icon font (Segoe MDL2 Assets / Segoe Fluent Icons); without it, no icon.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication

RESOURCES = Path(__file__).resolve().parent / "resources"
APP_ICON = RESOURCES / "runnerfinder.ico"

PALETTES = {
    "dark": {
        "bg": "#1b1c1f", "sidebar": "#141518", "surface": "#24262a", "surface2": "#2d3035", "border": "#3a3d43",
        "text": "#e8e9eb", "muted": "#9aa0a8", "accent": "#3d8bfd", "accent_text": "#ffffff",
        "accent_soft": "#1f3555", "ok": "#3fb950", "warn": "#d6a23a", "bad": "#f0605a", "rejected": "#7d838b",
        "input": "#1f2124", "hover": "#2a3442",
    },
    "light": {
        "bg": "#f4f5f7", "sidebar": "#e9ecf1", "surface": "#ffffff", "surface2": "#f1f3f6", "border": "#d6dbe2",
        "text": "#1d2127", "muted": "#5f6773", "accent": "#2563eb", "accent_text": "#ffffff",
        "accent_soft": "#dce8ff", "ok": "#1a7f37", "warn": "#9a6700", "bad": "#cf222e", "rejected": "#8b929b",
        "input": "#ffffff", "hover": "#e3ebfa",
    },
}

# Segoe MDL2 Assets code points
GLYPHS = {"dashboard": "", "new": "", "progress": "", "search": "", "open": "",
          "folder": "", "reject": "", "restore": "", "theme": "", "photo": "",
          "gpu": "", "cancel": "", "history": "", "add": "", "play": ""}

_current = "dark"
ICON_GREY = "#8b95a3"          # glyph colour that reads well on both themes


def current() -> str:
    return _current


def colors() -> dict:
    return PALETTES[_current]


def saved_theme() -> str:
    value = QSettings().value("ui/theme", "dark")
    return value if value in PALETTES else "dark"


def _icon_font() -> str | None:
    families = set(QFontDatabase.families())
    for name in ("Segoe Fluent Icons", "Segoe MDL2 Assets"):
        if name in families:
            return name
    return None


def glyph_icon(name: str, color: str | None = None, size: int = 32) -> QIcon:
    """A monochrome icon from the Windows icon font (empty icon if the font is missing)."""
    family = _icon_font()
    if family is None or name not in GLYPHS:
        return QIcon()
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    font = QFont(family)
    font.setPixelSize(int(size * 0.72))
    p.setFont(font)
    p.setPen(QColor(color or ICON_GREY))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, GLYPHS[name])
    p.end()
    return QIcon(pix)


def app_icon() -> QIcon:
    return QIcon(str(APP_ICON)) if APP_ICON.exists() else QIcon()


def style_sheet(c: dict) -> str:
    return f"""
QWidget {{ color: {c['text']}; font-family: "Segoe UI"; font-size: 10pt; }}
QMainWindow, QStackedWidget#pages, QWidget#page {{ background: {c['bg']}; }}
QToolTip {{ background: {c['surface2']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px; }}

/* sidebar */
QFrame#sidebar {{ background: {c['sidebar']}; border-right: 1px solid {c['border']}; }}
QLabel#brand {{ font-size: 15pt; font-weight: 700; }}
QLabel#brandSub {{ color: {c['muted']}; font-size: 8.5pt; }}
QListWidget#nav {{ background: transparent; border: none; font-size: 10.5pt; outline: 0; }}
QListWidget#nav::item {{ padding: 9px 10px; margin: 2px 8px; border-radius: 8px; color: {c['muted']}; }}
QListWidget#nav::item:hover {{ background: {c['hover']}; color: {c['text']}; }}
QListWidget#nav::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; font-weight: 600; }}
QPushButton#themeButton {{ background: transparent; border: 1px solid {c['border']}; border-radius: 8px;
                           padding: 6px 10px; color: {c['muted']}; text-align: left; }}
QPushButton#themeButton:hover {{ background: {c['hover']}; color: {c['text']}; }}
QLabel#sidebarFoot {{ color: {c['muted']}; font-size: 8pt; }}

/* page structure */
QLabel#pageTitle {{ font-size: 19pt; font-weight: 700; }}
QLabel#pageSubtitle {{ color: {c['muted']}; font-size: 10pt; margin-bottom: 6px; }}
QLabel#sectionTitle {{ font-size: 11pt; font-weight: 600; }}
QLabel#eventName {{ font-size: 15pt; font-weight: 700; }}
QLabel#message {{ color: {c['muted']}; }}
QLabel#hint {{ color: {c['muted']}; font-size: 8.5pt; }}
QLabel#fieldLabel {{ color: {c['muted']}; }}
QFrame#card {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 12px; }}
QFrame#tile {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 12px; }}
QLabel#tileValue {{ font-size: 20pt; font-weight: 700; }}
QLabel#tileCaption {{ color: {c['muted']}; font-size: 9pt; }}
QLabel#bigPercent {{ font-size: 34pt; font-weight: 700; color: {c['accent']}; }}
QLabel#emptyState {{ color: {c['muted']}; font-size: 12pt; }}
QLabel#previewTitle {{ font-size: 12.5pt; font-weight: 700; }}
QLabel#preview {{ background: {c['surface2']}; border-radius: 10px; color: {c['muted']}; }}

/* status chips */
QLabel[chip="ok"] {{ background: {c['surface2']}; color: {c['ok']}; border: 1px solid {c['border']};
                    border-radius: 11px; padding: 3px 10px; font-weight: 600; }}
QLabel[chip="bad"] {{ background: {c['surface2']}; color: {c['bad']}; border: 1px solid {c['border']};
                     border-radius: 11px; padding: 3px 10px; font-weight: 600; }}
QLabel[chip="neutral"] {{ background: {c['surface2']}; color: {c['muted']}; border: 1px solid {c['border']};
                         border-radius: 11px; padding: 3px 10px; }}
QLabel[state="ok"] {{ color: {c['ok']}; font-weight: 600; }}
QLabel[state="bad"] {{ color: {c['bad']}; font-weight: 600; }}
QLabel[state="warn"] {{ color: {c['warn']}; font-weight: 600; }}

/* inputs */
QLineEdit, QComboBox, QSpinBox {{ background: {c['input']}; border: 1px solid {c['border']}; border-radius: 8px;
                                 padding: 6px 10px; selection-background-color: {c['accent']}; }}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {c['accent']}; }}
QLineEdit#bigSearch {{ font-size: 16pt; font-weight: 600; padding: 8px 12px; letter-spacing: 1px; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {c['surface']}; border: 1px solid {c['border']};
                              selection-background-color: {c['accent_soft']}; selection-color: {c['text']}; }}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {c['muted']}; border-radius: 4px; background: {c['input']}; }}
QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border: 1px solid {c['accent']}; }}

/* buttons */
QPushButton {{ background: {c['surface2']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 7px 16px; }}
QPushButton:hover {{ background: {c['hover']}; }}
QPushButton:pressed {{ background: {c['accent_soft']}; }}
QPushButton:disabled {{ color: {c['muted']}; background: {c['surface']}; border-color: {c['surface2']}; }}
QPushButton#primary {{ background: {c['accent']}; color: {c['accent_text']}; border: 1px solid {c['accent']};
                      font-weight: 600; }}
QPushButton#primary:hover {{ background: {c['accent']}; border-color: {c['text']}; }}
QPushButton#primary:disabled {{ background: {c['surface2']}; color: {c['muted']}; border-color: {c['border']}; }}
QPushButton#danger:enabled {{ color: {c['bad']}; }}
QPushButton#chipButton {{ border-radius: 12px; padding: 3px 12px; font-size: 9pt; }}

/* lists, tables, progress */
QListView#results {{ background: {c['bg']}; border: none; outline: 0; }}
QListView#results::item {{ border-radius: 10px; padding: 6px; margin: 2px; }}
QListView#results::item:hover {{ background: {c['surface']}; }}
QListView#results::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}
QTableWidget {{ background: {c['surface']}; border: none; gridline-color: {c['border']};
               selection-background-color: {c['accent_soft']}; selection-color: {c['text']}; }}
QTableWidget::item {{ padding: 0 12px; }}
QHeaderView::section {{ background: {c['surface']}; color: {c['muted']}; border: none;
                       border-bottom: 1px solid {c['border']}; padding: 6px 12px; font-weight: 600; }}
QProgressBar {{ background: {c['surface2']}; border: none; border-radius: 7px; min-height: 14px; max-height: 14px;
               text-align: right; color: transparent; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 7px; }}
QSplitter::handle {{ background: {c['bg']}; }}
QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {c['accent']}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QStatusBar {{ background: {c['sidebar']}; color: {c['muted']}; border-top: 1px solid {c['border']}; }}
QStatusBar QLabel {{ color: {c['muted']}; font-size: 8.5pt; }}
QMessageBox {{ background: {c['surface']}; }}
"""


def apply(app: QApplication, name: str | None = None) -> str:
    """Apply a theme ("dark" / "light"; default: the saved choice). Returns the applied name."""
    global _current
    _current = name if name in PALETTES else saved_theme()
    c = PALETTES[_current]
    app.setStyle("Fusion")
    pal = QPalette()
    for role, key in ((QPalette.ColorRole.Window, "bg"), (QPalette.ColorRole.WindowText, "text"),
                      (QPalette.ColorRole.Base, "input"), (QPalette.ColorRole.AlternateBase, "surface2"),
                      (QPalette.ColorRole.Text, "text"), (QPalette.ColorRole.Button, "surface2"),
                      (QPalette.ColorRole.ButtonText, "text"), (QPalette.ColorRole.Highlight, "accent"),
                      (QPalette.ColorRole.HighlightedText, "accent_text"), (QPalette.ColorRole.ToolTipBase, "surface2"),
                      (QPalette.ColorRole.ToolTipText, "text"), (QPalette.ColorRole.PlaceholderText, "muted"),
                      (QPalette.ColorRole.Link, "accent")):
        pal.setColor(role, QColor(c[key]))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(c["muted"]))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(c["muted"]))
    app.setPalette(pal)
    app.setStyleSheet(style_sheet(c))
    return _current


def save(name: str) -> None:
    QSettings().setValue("ui/theme", name)


def set_chip(label, state: str) -> None:
    """state: ok / bad / neutral - re-polishes the label so the style sheet applies."""
    label.setProperty("chip", state)
    label.style().unpolish(label)
    label.style().polish(label)


def set_state(label, state: str) -> None:
    """Text colour of a value label: ok / bad / warn / '' (normal)."""
    label.setProperty("state", state)
    label.style().unpolish(label)
    label.style().polish(label)
