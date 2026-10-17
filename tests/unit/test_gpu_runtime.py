"""OCR device rules (utils/gpu_runtime.py) with a stand-in for PaddlePaddle."""
import sys
import types

import pytest

from config import AppConfig, OCRConfig
from utils import gpu_runtime
from utils.gpu_runtime import GPUUnavailable, cuda_dll_dirs, resolve_ocr_device


def fake_paddle(monkeypatch, cuda=True, count=1, compute_ok=True, active="gpu:0"):
    class T:
        def __matmul__(self, other):
            return self

        def sum(self):
            return 64.0 ** 3 if compute_ok else 0.0

    cuda_mod = types.SimpleNamespace(
        device_count=lambda: count,
        get_device_properties=lambda i: types.SimpleNamespace(
            name="Fake RTX", total_memory=8 * 2**30, major=8, minor=6))
    device = types.SimpleNamespace(is_compiled_with_cuda=lambda: cuda, cuda=cuda_mod,
                                   set_device=lambda d: None, get_device=lambda: active)
    paddle = types.SimpleNamespace(__version__="3.3.1", device=device,
                                   ones=lambda *a, **k: T(),
                                   version=types.SimpleNamespace(cuda=lambda: "13.0", cudnn=lambda: "9.13.0"))
    monkeypatch.setitem(sys.modules, "paddle", paddle)


def test_cpu_never_touches_paddle(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddle", None)          # import would fail
    assert resolve_ocr_device("cpu") == ("cpu", {}, None)


def test_gpu_verified(monkeypatch):
    fake_paddle(monkeypatch)
    device, info, note = resolve_ocr_device("gpu")
    assert device == "gpu:0" and note is None
    assert info["name"] == "Fake RTX" and info["memory_mib"] == 8192 and info["cuda"] == "13.0"


@pytest.mark.parametrize("kw, reason", [
    ({"cuda": False}, "CPU-only build"),
    ({"count": 0}, "no NVIDIA GPU"),
    ({"compute_ok": False}, "wrong result"),
])
def test_explicit_gpu_failure_is_an_error_not_a_silent_fallback(monkeypatch, kw, reason):
    fake_paddle(monkeypatch, **kw)
    with pytest.raises(GPUUnavailable, match=reason):
        resolve_ocr_device("gpu")


def test_missing_paddle_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddle", None)
    with pytest.raises(GPUUnavailable, match="cannot be imported"):
        resolve_ocr_device("gpu:0")


def test_auto_uses_gpu_or_falls_back_with_reason(monkeypatch):
    fake_paddle(monkeypatch)
    assert resolve_ocr_device("auto")[0] == "gpu:0"
    fake_paddle(monkeypatch, cuda=False)
    device, info, note = resolve_ocr_device("auto")
    assert device == "cpu" and info == {} and "CPU-only build" in note


def test_invalid_device_values():
    with pytest.raises(ValueError):
        resolve_ocr_device("cuda")
    cfg = AppConfig(ocr=OCRConfig(device="cuda:0"))
    with pytest.raises(ValueError, match="ocr.device"):
        cfg.validate()
    for ok in ("cpu", "gpu", "gpu:1", "auto"):
        AppConfig(ocr=OCRConfig(device=ok)).validate()


def test_dll_folders_found_only_where_nvidia_wheels_are(tmp_path):
    sp = tmp_path / "site-packages"
    (sp / "nvidia" / "cu13" / "bin" / "x86_64").mkdir(parents=True)
    (sp / "nvidia" / "cu13" / "bin" / "x86_64" / "cublasLt64_13.dll").write_bytes(b"")
    (sp / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    (sp / "nvidia" / "cudnn" / "bin" / "cudnn64_9.dll").write_bytes(b"")
    (sp / "nvidia" / "empty" / "bin").mkdir(parents=True)                 # no DLLs: ignored
    assert [p.relative_to(sp).as_posix() for p in cuda_dll_dirs([sp])] == [
        "nvidia/cu13/bin/x86_64", "nvidia/cudnn/bin"]
    assert cuda_dll_dirs([tmp_path / "cpu-venv"]) == []


def test_register_is_harmless_without_cuda_wheels(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_runtime.os, "add_dll_directory", lambda p: calls.append(p), raising=False)
    gpu_runtime.register_cuda_dlls([tmp_path])
    assert calls == []
