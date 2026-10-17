import ntpath
from pathlib import Path

from export.exporter import bib_export_folder, safe_folder_name
from scanner.photo_scanner import is_under, make_path_key


def test_windows_path_keys_ignore_case_and_slash_style():
    a = make_path_key(r"D:\RacePhotos\Taihu\IMG_0001.JPG", pathmod=ntpath)
    b = make_path_key("d:/racephotos/taihu/img_0001.jpg", pathmod=ntpath)
    assert a == b


def test_same_name_in_different_folders_gives_different_keys():
    a = make_path_key(r"D:\Race\CamA\IMG_0001.JPG", pathmod=ntpath)
    b = make_path_key(r"D:\Race\CamB\IMG_0001.JPG", pathmod=ntpath)
    assert a != b


def test_is_under_does_not_match_name_prefixes():
    root = make_path_key(r"D:\Race", pathmod=ntpath)
    assert is_under(make_path_key(r"D:\Race\a.jpg", pathmod=ntpath), root, sep="\\")
    assert not is_under(make_path_key(r"D:\Race2\a.jpg", pathmod=ntpath), root, sep="\\")


def test_export_layout_keeps_leading_zeros():
    p = bib_export_folder(Path("Export"), "2026_Xterra_Taihu", "UT001")
    assert p == Path("Export") / "2026_Xterra_Taihu" / "Bib_UT001"
    assert bib_export_folder(Path("E"), "R", "00123").name == "Bib_00123"


def test_unsafe_event_names_are_cleaned():
    assert safe_folder_name("Xterra: Taihu 2026?") == "Xterra__Taihu_2026_"
    assert safe_folder_name("CON") == "_CON"
    assert safe_folder_name("nul.txt") == "_nul.txt"
    assert safe_folder_name("   ") == "Unnamed"
    assert safe_folder_name("Race...") == "Race"
    assert safe_folder_name("2026 太湖越野") == "2026_太湖越野"
