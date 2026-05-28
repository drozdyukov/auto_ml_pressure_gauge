import json
from pathlib import Path

from PIL import Image

from pressure_gauge_ml.data import prepare_yolo_dataset


def _write_image(path: Path) -> None:
    Image.new("RGB", (100, 80), color=(240, 240, 240)).save(path)


def _write_split(root: Path, split: str, count: int) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "base"},
        {"id": 2, "name": "maximum"},
        {"id": 3, "name": "minimum"},
        {"id": 4, "name": "tip"},
    ]
    for index in range(count):
        file_name = f"{split}_{index}.jpg"
        _write_image(split_dir / file_name)
        images.append({"id": index, "file_name": file_name, "width": 100, "height": 80})
        for class_id in range(1, 5):
            annotations.append(
                {
                    "id": index * 10 + class_id,
                    "image_id": index,
                    "category_id": class_id,
                    "bbox": [10 * class_id, 20, 8, 8],
                }
            )
    coco = {"images": images, "categories": categories, "annotations": annotations}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def _write_dataset(root: Path) -> None:
    _write_split(root, "train", 12)
    _write_split(root, "valid", 2)
    _write_split(root, "test", 1)


def test_prepare_yolo_dataset_writes_yaml_labels_and_holdout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    data_yaml = prepare_yolo_dataset(
        dataset_root,
        tmp_path / "gauge_yolo",
        copy_images=True,
        reports_root=tmp_path / "reports",
        holdout_count=2,
        seed=42,
    )

    assert data_yaml.exists()
    assert (tmp_path / "gauge_yolo" / "holdout_data.yaml").exists()
    assert (tmp_path / "gauge_yolo" / "labels" / "train").exists()
    assert len(list((tmp_path / "gauge_yolo" / "images" / "holdout").glob("*.jpg"))) == 2

    first_label = next((tmp_path / "gauge_yolo" / "labels" / "train").glob("*.txt"))
    values = first_label.read_text(encoding="utf-8").strip().splitlines()[0].split()
    assert len(values) == 5

    yaml_text = data_yaml.read_text(encoding="utf-8")
    assert "base" in yaml_text
    assert "tip" in yaml_text
    assert (tmp_path / "reports" / "holdout_manifest.json").exists()
