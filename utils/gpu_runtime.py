"""
Optional NVIDIA GPU support for PaddleOCR (V0.2).

The CPU environment (.venv, hash-locked) is unchanged and never needs this.
GPU OCR runs from a SEPARATE environment (INSTALLATION.md, .venv-gpu) that
contains paddlepaddle-gpu and the NVIDIA CUDA/cuDNN wheels.

Windows does not find those wheel DLLs on its own: Paddle fails with
messages such as "cublasLt64_13.dll not found". register_cuda_dlls() adds
the wheel folders (site-packages/nvidia/*/bin[/x86_64]) to the DLL search
path. It must run before paddle is imported; config.py calls it. In an
environment without those folders (the CPU .venv) it does nothing.

resolve_ocr_device() decides the OCR device ONCE, at start-up:
  "cpu"         -> CPU, Paddle's GPU code is never touched
  "gpu"/"gpu:N" -> GPU, verified with a small tensor; failure is an ERROR
                   (an explicitly requested GPU never silently becomes CPU)
  "auto"        -> GPU if it verifies, otherwise CPU, with the reason kept
                   so it is printed and stored in the run
"""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_dll_handles: list = []          # keep the handles alive for the whole process
registered_dirs: list[str] = []


class GPUUnavailable(RuntimeError):
    """The requested GPU cannot be used by PaddlePaddle."""


def _site_dirs() -> list[Path]:
    dirs = []
    try:
        dirs += [Path(p) for p in site.getsitepackages()]
    except Exception:  # noqa: BLE001 - some embedded Pythons have no site.getsitepackages
        pass
    dirs += [Path(p) for p in sys.path if p.endswith("site-packages")]
    return list(dict.fromkeys(d for d in dirs if d.is_dir()))


def cuda_dll_dirs(site_dirs: list[Path] | None = None) -> list[Path]:
    """nvidia/<pkg>/bin and nvidia/<pkg>/bin/x86_64 folders that contain DLLs."""
    found = []
    for sp in site_dirs if site_dirs is not None else _site_dirs():
        nvidia = sp / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            for cand in (pkg / "bin" / "x86_64", pkg / "bin"):
                if cand.is_dir() and any(cand.glob("*.dll")):
                    found.append(cand)
    return found


def register_cuda_dlls(site_dirs: list[Path] | None = None) -> list[str]:
    """Make the CUDA/cuDNN wheel DLLs findable. Safe to call more than once; never raises."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    try:
        for d in cuda_dll_dirs(site_dirs):
            s = str(d)
            if s in registered_dirs:
                continue
            _dll_handles.append(os.add_dll_directory(s))
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
            registered_dirs.append(s)
    except Exception:  # noqa: BLE001 - a GPU problem must never break the CPU app
        pass
    return list(registered_dirs)


def gpu_status(index: int = 0) -> dict:
    """What PaddlePaddle can do with the GPU. Raises GPUUnavailable with the reason."""
    try:
        import paddle
    except Exception as exc:  # noqa: BLE001
        raise GPUUnavailable(f"PaddlePaddle cannot be imported: {type(exc).__name__}: {exc}") from exc
    info = {"paddle": getattr(paddle, "__version__", "?"),
            "compiled_with_cuda": bool(paddle.device.is_compiled_with_cuda()),
            "dll_dirs": list(registered_dirs)}
    if not info["compiled_with_cuda"]:
        raise GPUUnavailable("this PaddlePaddle is the CPU-only build "
                             "(use the GPU environment, see INSTALLATION.md)")
    try:
        count = paddle.device.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        raise GPUUnavailable(f"CUDA device query failed: {type(exc).__name__}: {exc}") from exc
    if count <= index:
        raise GPUUnavailable(f"no NVIDIA GPU {index} visible to PaddlePaddle (found {count})")
    try:
        paddle.device.set_device(f"gpu:{index}")
        x = paddle.ones([64, 64], dtype="float32")
        ok = float((x @ x).sum()) == 64.0 * 64 * 64
        props = paddle.device.cuda.get_device_properties(index)
        info.update({"device": f"gpu:{index}", "name": getattr(props, "name", "?"),
                     "memory_mib": round(getattr(props, "total_memory", 0) / 2**20),
                     "compute_capability": f"{getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')}",
                     "cuda": paddle.version.cuda(), "cudnn": paddle.version.cudnn()})
    except Exception as exc:  # noqa: BLE001
        raise GPUUnavailable(f"GPU test computation failed: {type(exc).__name__}: {exc}") from exc
    if not ok:
        raise GPUUnavailable("GPU test computation returned a wrong result")
    return info


def resolve_ocr_device(requested: str) -> tuple[str, dict, str | None]:
    """(device for PaddleOCR, GPU info, note). See the module docstring for the rules."""
    req = (requested or "cpu").strip().lower()
    if req == "cpu":
        return "cpu", {}, None
    if req == "auto":
        try:
            info = gpu_status(0)
        except GPUUnavailable as exc:
            return "cpu", {}, f"auto: GPU not usable, using CPU ({exc})"
        return info["device"], info, None
    if req == "gpu" or req.startswith("gpu:"):
        index = int(req.split(":", 1)[1]) if ":" in req else 0
        info = gpu_status(index)          # raises GPUUnavailable: no silent CPU fallback
        return info["device"], info, None
    raise ValueError(f'OCR device must be "cpu", "gpu", "gpu:N" or "auto", not {requested!r}')


def active_paddle_device() -> str | None:
    """Device Paddle is actually using now (checked after PaddleOCR has loaded)."""
    try:
        import paddle
        return str(paddle.device.get_device())
    except Exception:  # noqa: BLE001
        return None
