import pytest

from config import AppConfig


def test_defaults_are_valid():
    cfg = AppConfig()
    cfg.validate()
    assert cfg.detection.strategy == "person_body"
    assert cfg.detection.full_image_fallback == "no_valid_bib"
    # Default crop must reach well below the chest (waist / thigh bibs).
    assert cfg.detection.crop_strategy == "body"
    assert cfg.detection.crop_fractions()[1] >= 0.85


@pytest.mark.parametrize("field, value", [
    ("strategy", "chest_only"), ("full_image_fallback", "sometimes"),
    ("crop_strategy", "chest_only"), ("person_conf_threshold", 1.5),
    ("full_image_tile_overlap", 2000),
])
def test_invalid_detection_settings_rejected(field, value):
    cfg = AppConfig()
    setattr(cfg.detection, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_custom_fractions_rejected():
    cfg = AppConfig()
    cfg.detection.crop_strategy = "custom"
    cfg.detection.crop_top, cfg.detection.crop_bottom = 0.8, 0.3
    with pytest.raises(ValueError):
        cfg.validate()


def test_default_export_root_is_not_a_source_package():
    """On Windows "Export" and "export" are the same folder: the export root must not be the export/ package."""
    from pathlib import Path

    import config
    project = Path(config.__file__).resolve().parent
    root = AppConfig().paths.default_export_root
    packages = {p.name.lower() for p in project.iterdir() if (p / "__init__.py").is_file()}
    assert root.parent == project
    assert root.name.lower() not in packages, f"{root} is the source package {root.name.lower()}/"
