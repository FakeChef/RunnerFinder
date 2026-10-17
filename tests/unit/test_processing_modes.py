from PIL import Image

from config import DetectionConfig
from detection.bib_detector import PersonBox, RegionFinder
from detection.geometry import Box, center_crop_box
from detection.ranking import rank_persons
from ocr.numeric_candidates import numeric_identifier, numeric_identifiers


class ExplodingDetector:
    def detect(self, image):
        raise AssertionError("Center mode must bypass YOLO completely")


class FixedDetector:
    def __init__(self, detections):
        self.detections = detections
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.detections


def test_center_crop_geometry_preserves_center_and_size():
    box = center_crop_box(1800, 1200, 0.60, 0.60)
    assert box.xywh() == (360, 240, 1080, 720)


def test_center_mode_does_not_initialize_or_call_detector():
    image = Image.new("RGB", (1000, 800))
    cfg = DetectionConfig(processing_mode="center")
    result = RegionFinder(cfg, ExplodingDetector(), "center").find(image)
    assert result.persons == []
    assert result.regions[0].source == "center_crop"
    assert result.regions[0].box.xywh() == (200, 160, 600, 480)


def test_lite_mode_selects_top_k_ranked_people():
    from detection.person_detector import RawDetection
    detector = FixedDetector([
        RawDetection(0, 0, 200, 300, 0.99),
        RawDetection(350, 250, 950, 1000, 0.70),
        RawDetection(1200, 100, 1450, 500, 0.95),
    ])
    image = Image.new("RGB", (1600, 1200))
    cfg = DetectionConfig(processing_mode="lite2")
    result = RegionFinder(cfg, detector, "lite2").find(image)
    assert detector.calls == 1
    assert len(result.regions) == 2
    assert [r.person_index for r in result.regions] == [2, 1]
    assert result.full_image_fallback_pending is False
    assert all(any(n.startswith("rank_score=") for n in r.notes) for r in result.regions)


def test_ranking_weights_are_normalized_and_deterministic():
    persons = [PersonBox(1, Box(0, 0, 100, 100), 1.0, True),
               PersonBox(2, Box(450, 350, 950, 850), 0.5, True)]
    ranked = rank_persons(persons, (1000, 1000))
    assert ranked[0][0].index == 2
    assert 0 < ranked[0][1] <= 1


def test_numeric_identifiers_preserve_leading_zero_and_do_not_join_runs():
    assert numeric_identifier("A18866") == "18866"
    assert numeric_identifier("UT001") == "001"
    assert numeric_identifiers("B 31087") == ["31087"]
    assert numeric_identifiers("10 258") == ["10", "258"]
    assert numeric_identifier("10 258") is None
