"""Windows actions for result cards. Originals are only opened, never written."""
from __future__ import annotations

import os
import subprocess

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


def open_photo(path: str) -> bool:
    """Open the original photo in the default Windows viewer."""
    return os.path.isfile(path) and QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def show_in_folder(path: str) -> bool:
    """Open Windows Explorer with the photo selected."""
    if not os.path.isfile(path):
        return False
    if os.name == "nt":
        subprocess.Popen(["explorer.exe", f"/select,{os.path.normpath(path)}"])  # noqa: S603
        return True
    return QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
