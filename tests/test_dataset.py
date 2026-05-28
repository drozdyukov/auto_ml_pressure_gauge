import json
from pathlib import Path

from PIL import Image

from pressure_gauge_ml.data import (
    build_dataset_quality_report,
    prepare_yolo_dataset,
    summarize_coco_dataset,
    validate_coco_dataset,
)


CLASSES = [
    {"id": 1, "name": "base"},
    {"id": 2, "name": "maximum"},
    {"id": 3, "name": "minimum"},
    {"id": 4, "name": "tip"},
]


def _write_image(path: Path) -> None:
    Image.new("RGB", (100, 80), color=(240, 240, 240)).save(path)


def _annotations(image_id: int, *, missing_class: int | None = None) -> list[dict]:
    rows = []
    for class_id in range(1, 5):
        if class_id == missing_class:
            continue
        rows.append(
            {
                "id": image_id * 10 + class_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [10 * class_id, 10, 8, 8],
            }
        )
    return rows


def _write_split(root: Path, split: str, image_count: int, *, invalid: bool = False) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    images = []
    annotations = []
    for index in range(image_count):
        file_name = f"{split}_{index}.jpg"
        _write_image(split_dir / file_name)
        images.append({"id": index, "file_name": file_name, "width": 100, "height": 80})
        annotations.extend(_annotations(index, missing_class=2 if invalid and index == 0 else None))
    coco = {"images": images, "categories": CLASSES, "annotations": annotations}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")


def _write_dataset(root: Path, *, invalid_train: bool = False, invalid_valid: bool = False) -> None:
    _write_split(root, "train", 12, invalid=invalid_train)
    _write_split(root, "valid", 2, invalid=invalid_valid)
    _write_split(root, "test", 1)


def test_dataset_has_expected_splits(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    summary = summarize_coco_dataset(dataset_root)

    assert set(summary) == {"train", "valid", "test"}
    assert summary["train"]["images"] > summary["valid"]["images"] > summary["test"]["images"]
    assert summary["train"]["annotations"] > 0


def test_coco_annotations_are_consistent_with_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    errors = validate_coco_dataset(dataset_root, tmp_path / "reports")

    assert errors == []
    assert (tmp_path / "reports" / "dataset_quality_report.json").exists()
    assert (tmp_path / "reports" / "dataset_quality_report.csv").exists()


def test_train_invalid_images_are_excluded_and_quarantined(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root, invalid_train=True)

    prepare_yolo_dataset(
        dataset_root,
        tmp_path / "prepared",
        copy_images=True,
        reports_root=tmp_path / "reports",
        holdout_count=2,
        seed=42,
    )

    assert not (tmp_path / "prepared" / "images" / "train" / "train_0.jpg").exists()
    assert (tmp_path / "reports" / "quarantine" / "train" / "train_0" / "train_0.jpg").exists()
    assert (tmp_path / "prepared" / "images" / "holdout").exists()
    assert len(list((tmp_path / "prepared" / "images" / "holdout").glob("*.jpg"))) == 2


def test_invalid_valid_split_fails_strict_validation(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root, invalid_valid=True)

    errors = validate_coco_dataset(dataset_root, tmp_path / "reports")

    assert errors
    assert "valid" in errors[0]


def test_quality_report_flags_bad_bbox_and_corrupt_image(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_dataset(dataset_root)
    (dataset_root / "train" / "train_0.jpg").write_bytes(b"bad")
    coco_path = dataset_root / "train" / "_annotations.coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco["annotations"][0]["bbox"] = [-1, 0, 200, 200]
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    report = build_dataset_quality_report(dataset_root)
    reasons = [
        reason
        for issue in report["issues"]
        if issue["split"] == "train" and issue["file_name"] == "train_0.jpg"
        for reason in issue["reasons"]
    ]

    assert any("unreadable" in reason for reason in reasons)
    assert any("outside image bounds" in reason for reason in reasons)
