import pytest

from config import CROP_PRESETS, DetectionConfig
from detection.benchmark import select_files
from detection.geometry import (Box, body_crop, crop_to_original, detection_size,
                                fraction_inside, full_image_tiles, inside_any, iou,
                                map_to_original)


# ---------------------------------------------------------------- downscale + mapping

@pytest.mark.parametrize("size, max_side, expected", [
    ((6000, 4000), 1280, (1280, 853)),     # landscape 24 MP
    ((4000, 6000), 1280, (853, 1280)),     # portrait
    ((1000, 800), 1280, (1000, 800)),      # small: never upscaled
    ((5000, 5000), 1280, (1280, 1280)),
])
def test_detection_size(size, max_side, expected):
    assert detection_size(*size, max_side) == expected


def test_full_frame_box_maps_to_full_image():
    b = map_to_original(Box(0, 0, 1280, 853), (1280, 853), (6000, 4000))
    assert b == Box(0, 0, 6000, 4000)


def test_mapping_uses_separate_x_and_y_scales():
    # 6001x4001 -> copy 1280x853: x and y factors differ slightly.
    det = detection_size(6001, 4001, 1280)
    b = map_to_original(Box(640, 426.5, 1280, 853), det, (6001, 4001))
    assert b.x1 == pytest.approx(3000.5, abs=0.01) and b.x2 == pytest.approx(6001)
    assert b.y1 == pytest.approx(426.5 * 4001 / det[1]) and b.y2 == pytest.approx(4001)


def test_mapping_clamps_to_image():
    b = map_to_original(Box(-5, -5, 1300, 900), (1280, 853), (6000, 4000))
    assert b == Box(0, 0, 6000, 4000)


# ---------------------------------------------------------------- body crops

def test_default_crop_covers_chest_to_upper_legs():
    top, bottom, side = CROP_PRESETS["body"]
    person = Box(1000, 500, 1400, 1500)                 # 400 x 1000 px, standing runner
    c = body_crop(person, 6000, 4000, top, bottom, side)
    assert c.box == Box(940, 600, 1460, 1400) and c.notes == ()
    # chest ~25 %, abdomen ~40 %, waist ~50 %, upper legs ~60-75 % of body height
    for frac in (0.25, 0.40, 0.50, 0.60, 0.75):
        assert c.box.y1 <= person.y1 + frac * person.height <= c.box.y2


@pytest.mark.parametrize("preset", list(CROP_PRESETS))
def test_presets_stay_inside_image_and_are_integers(preset):
    top, bottom, side = CROP_PRESETS[preset]
    for person in (Box(0, 0, 50, 300), Box(5950, 3500, 6000, 4000), Box(10.4, 20.6, 990.2, 3999.9)):
        b = body_crop(person, 6000, 4000, top, bottom, side).box
        assert 0 <= b.x1 < b.x2 <= 6000 and 0 <= b.y1 < b.y2 <= 4000
        assert all(float(v).is_integer() for v in (b.x1, b.y1, b.x2, b.y2))


def test_crop_rounds_outwards():
    b = body_crop(Box(100.6, 200.6, 300.4, 1200.4), 6000, 4000, 0.1, 0.9, 0.0).box
    assert (b.x1, b.x2) == (100, 301)


def test_runner_cut_off_at_bottom_edge_keeps_waist_and_thighs():
    # Box ends at the image bottom: the legs continue outside the photo.
    person = Box(1000, 3000, 1400, 4000)
    c = body_crop(person, 6000, 4000, 0.10, 0.90, 0.15)
    assert c.box.y2 == 4000
    assert "bottom" in c.notes[0]
    # Without edge extension the bottom 10 % (where the waist is) would be lost.
    c2 = body_crop(person, 6000, 4000, 0.10, 0.90, 0.15, extend_at_edges=False)
    assert c2.box.y2 == 3900


def test_runner_cut_off_at_top_edge_starts_at_top():
    c = body_crop(Box(1000, 0, 1400, 900), 6000, 4000, 0.10, 0.90, 0.15)
    assert c.box.y1 == 0 and any("top" in n for n in c.notes)


def test_runner_at_side_edge_is_clamped_and_noted():
    c = body_crop(Box(0, 1000, 300, 2000), 6000, 4000, 0.10, 0.90, 0.15)
    assert c.box.x1 == 0 and any("side" in n for n in c.notes)


def test_side_view_margin_widens_narrow_box():
    person = Box(1000, 500, 1150, 1500)                   # side view: narrow box
    c = body_crop(person, 6000, 4000, 0.10, 0.90, 0.15)
    assert c.box.width == pytest.approx(150 * 1.3, abs=1)


def test_custom_crop_strategy():
    cfg = DetectionConfig(crop_strategy="custom", crop_top=0.0, crop_bottom=0.7, crop_side_margin=0.3)
    assert cfg.crop_fractions() == (0.0, 0.7, 0.3)
    assert DetectionConfig(crop_strategy="torso").crop_fractions() == CROP_PRESETS["torso"]


# ---------------------------------------------------------------- tiles

def _covers_every_pixel(tiles, w, h):
    xs = sorted({(t.x1, t.x2) for t in tiles})
    ys = sorted({(t.y1, t.y2) for t in tiles})
    for spans, length in ((xs, w), (ys, h)):
        assert spans[0][0] == 0 and spans[-1][1] == length
        for (a1, a2), (b1, b2) in zip(spans, spans[1:]):
            assert b1 <= a2, "gap between tiles"
    return True


def test_tiles_for_24mp_photo():
    tiles = full_image_tiles(6000, 4000, 2560, 256)
    assert len(tiles) == 6
    assert all(t.width <= 2560 and t.height <= 2560 for t in tiles)
    assert _covers_every_pixel(tiles, 6000, 4000)
    xs = sorted({t.x1 for t in tiles})
    assert xs[1] - xs[0] <= 2560 - 256                      # overlap >= 256 px


def test_small_photo_is_one_tile():
    assert full_image_tiles(1600, 1067, 2560, 256) == [Box(0, 0, 1600, 1067)]


@pytest.mark.parametrize("w, h", [(2561, 100), (8000, 5333), (3000, 9000)])
def test_tiles_cover_odd_sizes(w, h):
    assert _covers_every_pixel(full_image_tiles(w, h, 2560, 256), w, h)


# ---------------------------------------------------------------- OCR interface helpers

def test_crop_to_original():
    assert crop_to_original(Box(940, 600, 1460, 1400), Box(10, 20, 110, 70)) == \
        Box(950, 620, 1050, 670)


def test_inside_any_and_iou():
    person = Box(100, 100, 300, 600)
    assert fraction_inside(Box(150, 300, 250, 350), person) == 1.0
    assert inside_any(Box(250, 300, 350, 350), [person])          # half inside
    assert not inside_any(Box(500, 50, 900, 90), [person])        # banner text
    assert iou(person, person) == 1.0 and iou(person, Box(400, 0, 500, 50)) == 0.0


def test_select_files_spreads_over_whole_event():
    files = list(range(1000))
    assert select_files(files, 5) == [0, 250, 500, 749, 999]
    assert select_files(files, 3, spread=False) == [0, 1, 2]
    assert select_files(files[:4], 30) == [0, 1, 2, 3]
