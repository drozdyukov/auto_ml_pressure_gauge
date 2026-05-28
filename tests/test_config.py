from pathlib import Path

from pressure_gauge_ml.config import load_config, override_config


def test_load_config_resolves_project_paths() -> None:
    config = load_config("configs/train.yaml")

    assert config.dataset_root == Path("data/guage_read_coco").resolve()
    assert config.imgsz == 640
    assert config.batch == 4
    assert config.model == "rtdetr-l.pt"
    assert config.reports_root == Path("reports").resolve()
    assert config.holdout_count == 10
    assert config.degrees == 3.0
    assert config.scale == 0.15
    assert config.hsv_v == 0.2
    assert config.flipud == 0.0
    assert config.mosaic == 0.0
    assert config.mixup == 0.0
    assert "3060" in config.expected_gpu


def test_override_config_replaces_selected_values(tmp_path: Path) -> None:
    config = load_config("configs/train.yaml")
    overridden = override_config(
        config,
        dataset_root=tmp_path / "dataset",
        prepared_root=tmp_path / "prepared",
        model="rtdetr-x.pt",
        epochs=3,
        batch=2,
        name="smoke",
        degrees=5.0,
        scale=0.3,
        perspective=0.001,
        hsv_v=0.1,
        fliplr=0.25,
        mosaic=0.2,
    )

    assert overridden.dataset_root == (tmp_path / "dataset").resolve()
    assert overridden.prepared_root == (tmp_path / "prepared").resolve()
    assert overridden.model == "rtdetr-x.pt"
    assert overridden.epochs == 3
    assert overridden.batch == 2
    assert overridden.name == "smoke"
    assert overridden.degrees == 5.0
    assert overridden.scale == 0.3
    assert overridden.perspective == 0.001
    assert overridden.hsv_v == 0.1
    assert overridden.fliplr == 0.25
    assert overridden.mosaic == 0.2
    assert overridden.runs_root == config.runs_root
