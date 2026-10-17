"""
Real Ultralytics: detection must not use the network (no usage analytics,
no connectivity probes) once the model is on disk. Needs models/yolo26n.pt.
"""
import socket

import pytest

ultralytics = pytest.importorskip("ultralytics")

from config import get_config  # noqa: E402
from detection.person_detector import YoloPersonDetector  # noqa: E402

WEIGHTS = get_config().paths.models_dir / "yolo26n.pt"
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="models/yolo26n.pt missing")


def test_analytics_switched_off():
    from ultralytics import settings
    settings.update({"sync": True})          # Ultralytics' default on a fresh install
    YoloPersonDetector(WEIGHTS)
    assert settings["sync"] is False
    # An analytics object created now stays disabled even if the PC is online.
    import ultralytics.utils.events as ev
    original = ev.ONLINE
    ev.ONLINE = True
    try:
        assert ev.Events().enabled is False
    finally:
        ev.ONLINE = original


def test_detection_makes_no_network_connections(monkeypatch):
    from PIL import Image
    detector = YoloPersonDetector(WEIGHTS)
    attempts = []

    def no_connect(self, address, *a, **k):
        attempts.append(address)
        raise OSError("network blocked by test")

    def no_dns(host, *a, **k):
        attempts.append(host)
        raise OSError("DNS blocked by test")

    monkeypatch.setattr(socket.socket, "connect", no_connect)
    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    bus = ultralytics.__file__.replace("__init__.py", "assets/bus.jpg")
    with Image.open(bus) as img:
        for _ in range(3):
            detector.detect(img.convert("RGB"))
    assert attempts == []
