"""
Real YOLO26n person detection (needs the full install and models/yolo26n.pt,
which `check_environment.py --models` downloads). Skipped otherwise.

Uses the sample photo bus.jpg that ships inside the Ultralytics package
(4 people, 810x1080). These tests check that the pipeline WORKS with the
real model; they do not measure accuracy on race photos.
"""
from pathlib import Path

import pytest
from PIL import Image

ultralytics = pytest.importorskip("ultralytics")

from config import DetectionConfig, get_config  # noqa: E402
from detection.bib_detector import RegionFinder  # noqa: E402
from detection.geometry import Box, iou  # noqa: E402
from detection.person_detector import YoloPersonDetector  # noqa: E402
from scanner.photo_scanner import load_image_with_orientation  # noqa: E402

WEIGHTS = get_config().paths.models_dir / "yolo26n.pt"
BUS = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(),
                                reason="models/yolo26n.pt missing: run check_environment.py --models")


@pytest.fixture(scope="module")
def yolo():
    return YoloPersonDetector(WEIGHTS, conf=0.25, imgsz=640, device="cpu")


def match_all(boxes_a, boxes_b, min_iou):
    """Every box in a has a partner in b with IoU >= min_iou."""
    return all(max((iou(a, b) for b in boxes_b), default=0) >= min_iou for a in boxes_a)


def test_people_found_in_sample_photo(yolo):
    image, _ = load_image_with_orientation(BUS)
    res = RegionFinder(DetectionConfig(detection_image_size=640), yolo).find(image)
    assert len(res.kept_persons) >= 3
    w, h = image.size
    for r in res.regions:
        assert r.source == "person_body"
        assert 0 <= r.box.x1 < r.box.x2 <= w and 0 <= r.box.y1 < r.box.y2 <= h


def test_our_mapping_matches_ultralytics_own_mapping(yolo):
    """
    Our path: downscale to 640 ourselves, detect, map back.
    Reference: give Ultralytics the full image; it scales internally and
    returns boxes in original pixels. Both must agree.
    """
    image, _ = load_image_with_orientation(BUS)
    ours = [p.box for p in RegionFinder(DetectionConfig(detection_image_size=640), yolo)
            .find(image).persons]
    ref = yolo.model.predict(image, classes=[0], conf=0.25, imgsz=640, device="cpu",
                             verbose=False)[0].boxes.xyxy.cpu().numpy().tolist()
    ref = [Box(*b) for b in ref]
    assert abs(len(ours) - len(ref)) <= 1
    assert match_all(ours, ref, 0.85) or match_all(ref, ours, 0.85)


def test_exif_rotated_copy_gives_same_boxes(yolo, tmp_path):
    """Pixels stored rotated + EXIF orientation 6 must display - and detect - the same."""
    bus = Image.open(BUS).convert("RGB")
    exif = Image.Exif()
    exif[0x0112] = 6
    rotated = tmp_path / "bus_o6.jpg"
    bus.transpose(Image.Transpose.ROTATE_90).save(rotated, quality=95, exif=exif.tobytes())
    plain = tmp_path / "bus.jpg"
    bus.save(plain, quality=95)
    finder = RegionFinder(DetectionConfig(detection_image_size=640), yolo)
    a = finder.find(load_image_with_orientation(plain)[0])
    b = finder.find(load_image_with_orientation(rotated)[0])
    assert b.image_size == (810, 1080)
    assert match_all([p.box for p in a.persons], [p.box for p in b.persons], 0.9)


def test_runners_cut_off_at_bottom_keep_crop_to_edge(yolo, tmp_path):
    cut = tmp_path / "cut.jpg"
    Image.open(BUS).convert("RGB").crop((0, 0, 810, 560)).save(cut, quality=95)
    res = RegionFinder(DetectionConfig(detection_image_size=640), yolo).find(
        load_image_with_orientation(cut)[0])
    touching = [r for r in res.regions if any("bottom" in n for n in r.notes)]
    assert touching and all(r.box.y2 == 560 for r in touching)
