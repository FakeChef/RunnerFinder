"""Hardware/software description for reports. Standard library only."""
from __future__ import annotations

import os
import platform
import sys


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 2**30
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            return int(f.readline().split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        return None


def system_summary() -> dict:
    ram = ram_gb()
    return {
        "os": platform.platform(),
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram_gb": round(ram, 1) if ram else None,
        "python": platform.python_version(),
    }


def ai_summary() -> dict:
    """Versions and device of the AI stack, if installed."""
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["torch_threads"] = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        pass
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:  # noqa: BLE001
        pass
    return info
