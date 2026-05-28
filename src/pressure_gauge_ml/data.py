from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECTION_CLASS_ORDER = ("base", "maximum", "minimum", "tip")
EXPECTED_CLASS_SET = set(DETECTION_CLASS_ORDER)


def load_coco(annotation_path: Path) -> dict:
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def summarize_coco_dataset(dataset_root: Path) -> dict[str, dict[str, int | list[str]]]:
    summary: dict[str, dict[str, int | list[str]]] = {}
    for split in SPLITS:
        coco = load_coco(dataset_root / split / "_annotations.coco.json")
        category_names = [category["name"] for category in coco.get("categories", [])]
        summary[split] = {
            "images": len(coco.get("images", [])),
            "annotations": len(coco.get("annotations", [])),
            "categories": category_names,
        }
    return summary


def _image_is_readable(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _category_names(coco: dict) -> dict[int, str]:
    return {category["id"]: category["name"].lower() for category in coco.get("categories", [])}


def _annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        grouped[annotation.get("image_id")].append(annotation)
    return grouped


def _bbox_reasons(annotation: dict, image: dict) -> list[str]:
    bbox = annotation.get("bbox", [])
    if len(bbox) != 4:
        return [f"annotation {annotation.get('id')} has invalid bbox shape"]

    x, y, width, height = bbox
    reasons = []
    if width <= 0 or height <= 0:
        reasons.append(f"annotation {annotation.get('id')} has non-positive bbox size")
    if x < 0 or y < 0 or x + width > image["width"] or y + height > image["height"]:
        reasons.append(f"annotation {annotation.get('id')} bbox is outside image bounds")
    return reasons


def _image_quality_issues(
    split_dir: Path,
    image: dict,
    annotations: list[dict],
    category_names: dict[int, str],
    category_ids: set[int],
) -> list[str]:
    issues = []
    image_path = split_dir / image["file_name"]
    if not image_path.exists():
        issues.append("image file is missing")
    elif image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        issues.append("unsupported image extension")
    elif not _image_is_readable(image_path):
        issues.append("image file is corrupt or unreadable")

    if not annotations:
        issues.append("image has no annotations")

    seen_classes = set()
    for annotation in annotations:
        category_id = annotation.get("category_id")
        if category_id not in category_ids:
            issues.append(f"annotation {annotation.get('id')} references unknown category")
            continue
        class_name = category_names[category_id]
        if class_name not in EXPECTED_CLASS_SET:
            issues.append(f"annotation {annotation.get('id')} uses unexpected class '{class_name}'")
            continue
        seen_classes.add(class_name)
        issues.extend(_bbox_reasons(annotation, image))

    missing_classes = sorted(EXPECTED_CLASS_SET - seen_classes)
    if missing_classes:
        issues.append(f"image is missing required classes: {', '.join(missing_classes)}")
    return issues


def build_dataset_quality_report(dataset_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "expected_classes": list(DETECTION_CLASS_ORDER),
        "splits": {},
        "issues": [],
    }
    for split in SPLITS:
        split_dir = dataset_root / split
        annotation_path = split_dir / "_annotations.coco.json"
        split_summary = {
            "images": 0,
            "annotations": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "empty_images": 0,
            "class_distribution": {},
        }
        report["splits"][split] = split_summary

        if not split_dir.exists():
            report["issues"].append(
                {"split": split, "image_id": None, "file_name": None, "reasons": ["split directory is missing"]}
            )
            continue
        if not annotation_path.exists():
            report["issues"].append(
                {"split": split, "image_id": None, "file_name": None, "reasons": ["annotation file is missing"]}
            )
            continue

        coco = load_coco(annotation_path)
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        category_names = _category_names(coco)
        category_ids = set(category_names)
        annotations_by_image = _annotations_by_image(coco)
        image_ids = {image["id"] for image in images}
        split_summary["images"] = len(images)
        split_summary["annotations"] = len(annotations)

        class_counter = Counter(
            category_names.get(annotation.get("category_id"), "unknown")
            for annotation in annotations
        )
        split_summary["class_distribution"] = dict(class_counter)

        for annotation in annotations:
            if annotation.get("image_id") not in image_ids:
                report["issues"].append(
                    {
                        "split": split,
                        "image_id": annotation.get("image_id"),
                        "file_name": None,
                        "reasons": [f"annotation {annotation.get('id')} references missing image"],
                    }
                )

        for image in images:
            image_annotations = annotations_by_image.get(image["id"], [])
            reasons = _image_quality_issues(
                split_dir, image, image_annotations, category_names, category_ids
            )
            if not image_annotations:
                split_summary["empty_images"] += 1
            if reasons:
                split_summary["invalid_images"] += 1
                report["issues"].append(
                    {
                        "split": split,
                        "image_id": image["id"],
                        "file_name": image["file_name"],
                        "reasons": reasons,
                    }
                )
            else:
                split_summary["valid_images"] += 1
    return report


def _write_quality_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "image_id", "file_name", "reasons"])
        writer.writeheader()
        for issue in report["issues"]:
            writer.writerow(
                {
                    "split": issue["split"],
                    "image_id": issue["image_id"],
                    "file_name": issue["file_name"],
                    "reasons": " | ".join(issue["reasons"]),
                }
            )


def write_dataset_quality_report(
    dataset_root: Path,
    reports_root: Path,
    quarantine_train: bool = True,
) -> dict[str, Any]:
    report = build_dataset_quality_report(dataset_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "dataset_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_quality_csv(report, reports_root / "dataset_quality_report.csv")
    if quarantine_train:
        _write_train_quarantine(dataset_root, reports_root, report)
    return report


def _write_train_quarantine(dataset_root: Path, reports_root: Path, report: dict[str, Any]) -> None:
    quarantine_root = reports_root / "quarantine" / "train"
    shutil.rmtree(quarantine_root, ignore_errors=True)
    quarantine_root.mkdir(parents=True, exist_ok=True)

    for issue in report["issues"]:
        if issue["split"] != "train" or not issue["file_name"]:
            continue
        source_image = dataset_root / "train" / issue["file_name"]
        target_dir = quarantine_root / Path(issue["file_name"]).stem
        target_dir.mkdir(parents=True, exist_ok=True)
        if source_image.exists():
            shutil.copy2(source_image, target_dir / source_image.name)
        (target_dir / "reasons.json").write_text(
            json.dumps(issue, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def validate_coco_dataset(dataset_root: Path, reports_root: Path | None = None) -> list[str]:
    report = build_dataset_quality_report(dataset_root)
    if reports_root is not None:
        write_dataset_quality_report(dataset_root, reports_root)

    errors = []
    for issue in report["issues"]:
        if issue["split"] in {"valid", "test"}:
            errors.append(
                f"{issue['split']}: {issue['file_name'] or issue['image_id']}: "
                + "; ".join(issue["reasons"])
            )
    return errors


def _used_category_names(coco: dict) -> set[str]:
    category_names = _category_names(coco)
    return {
        category_names[ann["category_id"]]
        for ann in coco["annotations"]
        if ann.get("category_id") in category_names
    }


def _category_mapping(coco: dict) -> tuple[dict[int, int], dict[int, str]]:
    category_names = _category_names(coco)
    used_names = _used_category_names(coco)
    ordered_names = [name for name in DETECTION_CLASS_ORDER if name in used_names]
    extra_names = sorted(used_names - set(ordered_names))
    names = ordered_names + extra_names
    name_to_class = {name: index for index, name in enumerate(names)}
    category_to_class = {
        category_id: name_to_class[name]
        for category_id, name in category_names.items()
        if name in name_to_class
    }
    class_names = {index: name for name, index in name_to_class.items()}
    return category_to_class, class_names


def _to_yolo_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, box_w, box_h = bbox
    x_center = (x + box_w / 2) / width
    y_center = (y + box_h / 2) / height
    return x_center, y_center, box_w / width, box_h / height


def _link_or_copy(source_image: Path, target_image: Path, copy_images: bool) -> None:
    if target_image.exists():
        return
    if copy_images:
        shutil.copy2(source_image, target_image)
        return
    try:
        target_image.symlink_to(source_image.resolve())
    except OSError:
        try:
            target_image.hardlink_to(source_image.resolve())
        except OSError:
            shutil.copy2(source_image, target_image)


def _invalid_image_names(report: dict[str, Any], split: str) -> set[str]:
    return {issue["file_name"] for issue in report["issues"] if issue["split"] == split and issue["file_name"]}


def _select_holdout_names(valid_train_names: list[str], holdout_count: int, seed: int) -> list[str]:
    if holdout_count <= 0:
        return []
    ordered = sorted(valid_train_names)
    rng = random.Random(seed)
    sample_size = min(holdout_count, len(ordered))
    return sorted(rng.sample(ordered, sample_size))


def _write_yolo_split(
    *,
    coco: dict,
    source_dir: Path,
    image_out_dir: Path,
    label_out_dir: Path,
    image_names: set[str],
    copy_images: bool,
) -> None:
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    category_to_class, _ = _category_mapping(coco)
    annotations_by_image = _annotations_by_image(coco)

    for image in coco["images"]:
        if image["file_name"] not in image_names:
            continue
        source_image = source_dir / image["file_name"]
        target_image = image_out_dir / image["file_name"]
        _link_or_copy(source_image, target_image, copy_images)

        label_path = label_out_dir / f"{Path(image['file_name']).stem}.txt"
        rows = []
        for annotation in annotations_by_image.get(image["id"], []):
            if annotation["category_id"] not in category_to_class:
                continue
            class_id = category_to_class[annotation["category_id"]]
            bbox = _to_yolo_bbox(annotation["bbox"], image["width"], image["height"])
            rows.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in bbox))
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_data_yaml(output_root: Path, class_names: dict[int, str], filename: str = "data.yaml") -> Path:
    data_yaml = output_root / filename
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()),
                "train": "images/train",
                "val": "images/valid",
                "test": "images/test" if filename == "data.yaml" else "images/holdout",
                "names": class_names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data_yaml


def prepare_yolo_dataset(
    dataset_root: Path,
    output_root: Path,
    copy_images: bool = False,
    *,
    reports_root: Path | None = None,
    holdout_count: int = 10,
    seed: int = 42,
) -> Path:
    reports_root = reports_root or Path("reports")
    report = write_dataset_quality_report(dataset_root, reports_root)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_names = {index: name for index, name in enumerate(DETECTION_CLASS_ORDER)}
    holdout_names: list[str] = []
    for split in SPLITS:
        coco = load_coco(dataset_root / split / "_annotations.coco.json")
        split_dir = dataset_root / split
        invalid_names = _invalid_image_names(report, split) if split == "train" else set()
        valid_names = [image["file_name"] for image in coco["images"] if image["file_name"] not in invalid_names]
        if split == "train":
            holdout_names = _select_holdout_names(valid_names, holdout_count, seed)
            valid_names = [name for name in valid_names if name not in set(holdout_names)]
        _write_yolo_split(
            coco=coco,
            source_dir=split_dir,
            image_out_dir=output_root / "images" / split,
            label_out_dir=output_root / "labels" / split,
            image_names=set(valid_names),
            copy_images=copy_images,
        )

    train_coco = load_coco(dataset_root / "train" / "_annotations.coco.json")
    _write_yolo_split(
        coco=train_coco,
        source_dir=dataset_root / "train",
        image_out_dir=output_root / "images" / "holdout",
        label_out_dir=output_root / "labels" / "holdout",
        image_names=set(holdout_names),
        copy_images=copy_images,
    )
    holdout_manifest = {
        "seed": seed,
        "holdout_count": len(holdout_names),
        "files": holdout_names,
    }
    (reports_root / "holdout_manifest.json").write_text(
        json.dumps(holdout_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_data_yaml(output_root, class_names, "holdout_data.yaml")
    return _write_data_yaml(output_root, class_names)


def annotation_class_distribution(dataset_root: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        coco = load_coco(dataset_root / split / "_annotations.coco.json")
        category_names = _category_names(coco)
        counter = Counter(category_names[ann["category_id"]] for ann in coco["annotations"])
        result[split] = dict(counter)
    return result
