"""Double-clickable start file for the desktop GUI (runs with pythonw: no console window).

Start: <python with PySide6>\\pythonw.exe <repository>\\desktop\\RunnerFinder.pyw
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desktop.app import main  # noqa: E402

sys.exit(main())
