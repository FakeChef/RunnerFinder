"""
RegionFinder on real image files (decode, EXIF orientation, downscale,
mapping, crops), with a colour-blob stand-in for YOLO so that the expected
positions are exact.
"""
import pytest
from tests.helpers import ColorBlobDetector, draw_people
from PIL import ImageOps

from config import DetectionConfig
from detection.bib_detector import RegionFinder, crop_region
from scanner.photo_scanner import load_image_with_orientation

RED, BLUE, YELLOW = (230, 20, 20), (20, 40, 230), (240, 230, 20)


def finder(detector, **overrides):
    return RegionFinder(DetectionConfig(**overrides), detector)


def close(box, expected, tol):
    return all(abs(a - b) <= tol for a, b in zip((box.x1, box.y1, box.x2, box.y2), expected))


# Expected displayed (oriented) position of a raw point (x, y), for a raw image Wr x Hr.
TRANSFORMS = {
    1: lambda x, y, W, H: (x, y),          2: lambda x, y, W, H: (W - x, y),
    3: lambda x, y, W, H: (W - x, H - y),  4: lambda x, y, W, H: (x, H - y),
    5: lambda x, y, W, H: (y, x),          6: lambda x, y, W, H: (H - y, x),
    7: lambda x, y, W, H: (H - y, W - x),  8: lambda x, y, W, H: (y, W - x),
}


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_and_mapping_to_original(tmp_path, orientation):
    W, H = 1800, 1200
    raw_box = (200, 100, 500, 900)
    p = draw_people(tmp_path / "o.jpg", (W, H), [(RED, raw_box)], orientation=orientation)
    image, o = load_image_with_orientation(p)
    assert o == orientation
    corners = [TRANSFORMS[orientation](x, y, W, H) for x in raw_box[::2] for y in raw_box[1::2]]
    expected = (min(c[0] for c in corners), min(c[1] for c in corners),
                max(c[0] for c in corners), max(c[1] for c in corners))
    # detection copy is 4x smaller than the original: mapping must undo that
    res = finder(ColorBlobDetector([RED]), detection_image_size=450).find(image)
    assert res.detection_size == ((450, 300) if orientation < 5 else (300, 450))
    assert len(res.persons) == 1
    assert close(res.persons[0].box, expected, tol=5), (res.persons[0].box, expected)


def test_24mp_portrait_mapping_accuracy(tmp_path):
    p = draw_people(tmp_path / "big.jpg", (4000, 6000), [(BLUE, (1234, 2345, 2100, 5100))])
    image, _ = load_image_with_orientation(p)
    res = finder(ColorBlobDetector([BLUE])).find(image)
    assert res.detection_size == (853, 1280)
    # one detection-copy pixel is ~4.7 original pixels
    assert close(res.persons[0].box, (1234, 2345, 2100, 5100), tol=6)


def test_multiple_people_each_get_a_crop(tmp_path):
    people = [(YELLOW, (1300, 200, 1600, 1000)), (RED, (100, 150, 400, 1050)),
              (BLUE, (700, 300, 950, 950))]
    p = draw_people(tmp_path / "group.png", (1800, 1200), people, fmt="PNG")
    image, _ = load_image_with_orientation(p)
    res = finder(ColorBlobDetector([RED, BLUE, YELLOW])).find(image)
    assert [r.source for r in res.regions] == ["person_body"] * 3
    # numbered left to right, each crop belongs to its own person
    assert [p.box.x1 for p in res.persons] == sorted(p.box.x1 for p in res.persons)
    assert [r.person_index for r in res.regions] == [1, 2, 3]
    for region, color in zip(res.regions, (RED, BLUE, YELLOW)):
        crop = crop_region(image, region)
        cx, cy = crop.width // 2, crop.height // 2
        assert crop.getpixel((cx, cy)) == color        # centre of crop is that runner


def test_crops_are_full_resolution_pixels(tmp_path):
    p = draw_people(tmp_path / "x.png", (3000, 2000), [(RED, (1000, 400, 1300, 1600))], fmt="PNG")
    image, _ = load_image_with_orientation(p)
    res = finder(ColorBlobDetector([RED])).find(image)
    region = res.regions[0]
    crop = crop_region(image, region)
    assert crop.size == (region.box.width, region.box.height)
    assert crop.tobytes() == image.crop(region.box.pil()).tobytes()
    assert res.detection_size == (1280, 853)          # detection used a smaller copy...
    assert crop.width > 300                           # ...but the crop did not


def test_partly_visible_runner_at_bottom_edge(tmp_path):
    p = draw_people(tmp_path / "cut.png", (1800, 1200), [(RED, (600, 500, 900, 1200))], fmt="PNG")
    image, _ = load_image_with_orientation(p)
    res = finder(ColorBlobDetector([RED])).find(image)
    assert res.regions[0].box.y2 == 1200
    assert any("bottom" in n for n in res.regions[0].notes)


def test_landscape_and_portrait_give_same_relative_crop(tmp_path):
    a = draw_people(tmp_path / "l.png", (1800, 1200), [(RED, (100, 100, 400, 1100))], fmt="PNG")
    b = draw_people(tmp_path / "p.png", (1200, 1800), [(RED, (100, 100, 400, 1100))], fmt="PNG")
    spans = []
    for path in (a, b):
        image, _ = load_image_with_orientation(path)
        r = finder(ColorBlobDetector([RED])).find(image).regions[0]
        spans.append((r.box.y1, r.box.y2))
    # 10 %..90 % of the 1000 px person; rounding outwards may add 1-2 px
    assert all(abs(a - b) <= 2 for a, b in zip(spans[0], spans[1]))
    assert abs(spans[0][0] - 200) <= 2 and abs(spans[0][1] - 1000) <= 2


# ---------------------------------------------------------------- no person / fallback

@pytest.mark.parametrize("mode, expected_regions", [
    ("never", 0), ("no_person", 1), ("no_valid_bib", 1)])
def test_zero_person_fallback(tmp_path, mode, expected_regions):
    p = draw_people(tmp_path / "empty.jpg", (1800, 1200), [])
    image, _ = load_image_with_orientation(p)
    res = finder(ColorBlobDetector([RED]), full_image_fallback=mode).find(image)
    assert res.persons == [] and len(res.regions) == expected_regions
    assert not res.full_image_fallback_pending
    if expected_regions:
        r = res.regions[0]
        assert r.source == "full_image" and r.box.xywh() == (0, 0, 1800, 1200)


def test_zero_person_large_photo_uses_tiles(tmp_path):
    p = draw_people(tmp_path / "empty_big.jpg", (6000, 4000), [])
    image, _ = load_image_with_orientation(p)
    res = finder(ColorBlobDetector([RED])).find(image)
    assert len(res.regions) == 6 and all(r.source == "full_image" for r in res.regions)


def test_fallback_pending_when_people_found(tmp_path):
    p = draw_people(tmp_path / "one.png", (1800, 1200), [(RED, (100, 100, 400, 1100))], fmt="PNG")
    image, _ = load_image_with_orientation(p)
    assert finder(ColorBlobDetector([RED])).find(image).full_image_fallback_pending
    assert not finder(ColorBlobDetector([RED]),
                      full_image_fallback="no_person").find(image).full_image_fallback_pending


def test_too_small_people_are_skipped_and_trigger_fallback(tmp_path):
    p = draw_people(tmp_path / "far.png", (1800, 1200), [(RED, (100, 100, 140, 180))], fmt="PNG")
    image, _ = load_image_with_orientation(p)
    res = finder(ColorBlobDetector([RED])).find(image)
    assert len(res.persons) == 1 and not res.persons[0].kept
    assert "too small" in res.persons[0].skip_reason
    assert [r.source for r in res.regions] == ["full_image"]
    assert res.fallback_reason == "no person kept"


def test_bib_model_strategy_is_reserved():
    with pytest.raises(NotImplementedError):
        RegionFinder(DetectionConfig(strategy="bib_model"), ColorBlobDetector([RED]))
