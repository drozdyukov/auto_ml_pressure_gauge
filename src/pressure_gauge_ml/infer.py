from __future__ import annotations

import json
from pathlib import Path


def _is_rtdetr_path(path: Path) -> bool:
    return any(part.lower().startswith("rtdetr") for part in path.parts)


def run_inference(
    model_path: Path,
    source: Path,
    output_dir: Path,
    conf: float = 0.25,
    iou: float = 0.7,
    predictions_json: Path | None = None,
) -> Path:
    from ultralytics import RTDETR, YOLO

    model_cls = RTDETR if _is_rtdetr_path(model_path) else YOLO
    model = model_cls(str(model_path))
    results = model.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )
    if predictions_json is not None:
        predictions_json.parent.mkdir(parents=True, exist_ok=True)
        predictions_json.write_text(
            json.dumps(_results_to_records(results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return output_dir


def _results_to_records(results: list) -> list[dict]:
    records = []
    for result in results:
        names = result.names
        boxes = result.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        for bbox, confidence, class_id in zip(xyxy, confidences, classes, strict=True):
            class_index = int(class_id)
            records.append(
                {
                    "image": str(result.path),
                    "class": names.get(class_index, str(class_index)),
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(value) for value in bbox],
                }
            )
    return records
