from __future__ import annotations

import json
from pathlib import Path


def _is_rtdetr_path(path: Path) -> bool:
    return any(part.lower().startswith("rtdetr") for part in path.parts)


def evaluate_model(model_path: Path, data_yaml: Path, output_path: Path, imgsz: int = 640) -> dict:
    from ultralytics import RTDETR, YOLO

    model_cls = RTDETR if _is_rtdetr_path(model_path) else YOLO
    model = model_cls(str(model_path))
    metrics = model.val(data=str(data_yaml), split="test", imgsz=imgsz)
    result = {
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "box_precision": float(metrics.box.mp),
        "box_recall": float(metrics.box.mr),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
