from __future__ import annotations

import json
import time
from pathlib import Path

from pressure_gauge_ml.data import annotation_class_distribution, summarize_coco_dataset
from pressure_gauge_ml.hardware import get_nvidia_smi_summary


def write_monitoring_snapshot(dataset_root: Path, output_path: Path) -> dict:
    import psutil

    snapshot = {
        "created_at_unix": time.time(),
        "dataset": summarize_coco_dataset(dataset_root),
        "class_distribution": annotation_class_distribution(dataset_root),
        "infrastructure": {
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "memory_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
            "gpu": get_nvidia_smi_summary(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot
