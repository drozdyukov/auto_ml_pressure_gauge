import json
from pathlib import Path

from pressure_gauge_ml.config import load_config
from pressure_gauge_ml.train import _run_name, _write_run_artifacts


def test_finetune_run_name_gets_prefix() -> None:
    assert _run_name("camera_a", "finetune", "20260101_120000").startswith("finetune_camera_a_")
    assert _run_name("finetune_camera_a", "finetune", "20260101_120000").startswith(
        "finetune_camera_a_"
    )


def test_finetune_manifest_contains_source_model_and_metrics(tmp_path: Path) -> None:
    config = load_config("configs/train.yaml")
    data_yaml = tmp_path / "data.yaml"
    holdout_yaml = tmp_path / "holdout_data.yaml"
    data_yaml.write_text("names:\n  0: base\n", encoding="utf-8")
    holdout_yaml.write_text("names:\n  0: base\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    best_model = weights_dir / "best.pt"
    last_model = weights_dir / "last.pt"
    best_model.write_bytes(b"best")
    last_model.write_bytes(b"last")

    _write_run_artifacts(
        config,
        data_yaml,
        run_dir,
        "finetune_example_20260101_120000",
        best_model,
        last_model,
        "20260101_120000",
        "finetune",
        "runs/rtdetr/base/weights/best.pt",
        {"box_map50": 0.5},
        {"box_map50": 0.4},
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_type"] == "finetune"
    assert manifest["source_model"] == "runs/rtdetr/base/weights/best.pt"
    assert manifest["test_metrics"]["box_map50"] == 0.5
    assert manifest["holdout_metrics"]["box_map50"] == 0.4
    assert manifest["augmentation"]["scale"] == config.scale
    assert (run_dir / "training_config.yaml").exists()
