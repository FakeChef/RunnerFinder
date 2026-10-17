"""
Analysis pipeline: scanner -> detection -> OCR -> database.
Shared by the Streamlit UI and index_cli.py. Implemented in Phases 4-6.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[int, int, str], None]   # done, total, current file


def analyze_folder(folder: Path, event_name: str,
                   progress: ProgressCallback | None = None) -> int:
    """Analyse every new/changed photo in folder. Returns the run id."""
    raise NotImplementedError("Phases 4-6")
