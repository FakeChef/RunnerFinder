import pytest

from scanner.photo_scanner import _orientation, parse_exif_datetime, parse_offset, parse_subsec


@pytest.mark.parametrize("value, subsec, expected", [
    ("2026:04:12 08:31:22", None, "2026-04-12T08:31:22"),
    ("2026:04:12 08:31:22", "45", "2026-04-12T08:31:22.450"),      # Canon/Nikon: 2 digits
    ("2026:04:12 08:31:22", "457", "2026-04-12T08:31:22.457"),     # Sony: 3 digits
    ("2026:04:12 08:31:22", "4571234", "2026-04-12T08:31:22.457123"),  # capped at 6
    ("2026:04:12 08:31:22", "04", "2026-04-12T08:31:22.040"),      # leading zero kept
    ("2026:04:12 08:31:22\x00", " 45\x00", "2026-04-12T08:31:22.450"),  # NUL padding
    (b"2026:04:12 08:31:22", None, "2026-04-12T08:31:22"),
    ("2026:04:12 08:31:22.33", None, "2026-04-12T08:31:22.330"),
    ("0000:00:00 00:00:00", None, None),                           # unset clock
    ("    :  :     :  :  ", None, None),
    ("2026:13:45 08:31:22", None, None),                           # impossible date
    ("", None, None), (None, "45", None),
])
def test_parse_exif_datetime(value, subsec, expected):
    assert parse_exif_datetime(value, subsec) == expected


def test_subsecond_times_sort_correctly_as_text():
    times = [parse_exif_datetime("2026:04:12 08:31:22", s) for s in ("9", "45", "1", "450", None)]
    times = [t for t in times if t]
    assert sorted(times) == ["2026-04-12T08:31:22", "2026-04-12T08:31:22.100",
                             "2026-04-12T08:31:22.450", "2026-04-12T08:31:22.450",
                             "2026-04-12T08:31:22.900"]


@pytest.mark.parametrize("value, expected", [("+08:00", "+08:00"), ("-05:30", "-05:30"),
                                             ("8", None), ("", None), (None, None)])
def test_parse_offset(value, expected):
    assert parse_offset(value) == expected


@pytest.mark.parametrize("value, expected", [(1, 1), (6, 6), (8, 8), (0, None), (9, None),
                                             ("x", None), (None, None)])
def test_orientation_values(value, expected):
    assert _orientation(value) == expected


def test_subsec_garbage():
    assert parse_subsec("  ") is None and parse_subsec("ab") is None
