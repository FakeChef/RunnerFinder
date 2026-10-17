from export.exporter import unique_destination


def test_duplicate_file_names_get_unique_export_names(tmp_path):
    taken: set[str] = set()
    a = unique_destination(tmp_path, "IMG_0001.JPG", taken)
    b = unique_destination(tmp_path, "IMG_0001.JPG", taken)
    c = unique_destination(tmp_path, "img_0001.jpg", taken)   # Windows: same name
    assert [a.name, b.name, c.name] == ["IMG_0001.JPG", "IMG_0001_2.JPG", "img_0001_3.jpg"]


def test_existing_export_files_are_not_overwritten(tmp_path):
    (tmp_path / "IMG_0001.JPG").write_bytes(b"x")
    assert unique_destination(tmp_path, "IMG_0001.JPG", set()).name == "IMG_0001_2.JPG"
