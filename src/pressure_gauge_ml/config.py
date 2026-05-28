from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: Path
    prepared_root: Path
    runs_root: Path
    reports_root: Path
    experiment_name: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    device: str | int
    workers: int
    patience: int
    seed: int
    optimizer: str
    cache: bool
    amp: bool
    project: Path
    name: str
    exist_ok: bool
    confidence_threshold: float
    iou_threshold: float
    expected_gpu: str
    holdout_count: int
    degrees: float
    translate: float
    scale: float
    shear: float
    perspective: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    fliplr: float
    flipud: float
    mosaic: float
    mixup: float

    @property
    def data_yaml(self) -> Path:
        return self.prepared_root / "data.yaml"

    @property
    def holdout_data_yaml(self) -> Path:
        return self.prepared_root / "holdout_data.yaml"


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(path: str | Path = "configs/train.yaml") -> TrainConfig:
    config_path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent.parent
    return TrainConfig(
        dataset_root=_resolve_path(raw["dataset_root"], base_dir),
        prepared_root=_resolve_path(raw["prepared_root"], base_dir),
        runs_root=_resolve_path(raw["runs_root"], base_dir),
        reports_root=_resolve_path(raw.get("reports_root", "reports"), base_dir),
        experiment_name=str(raw["experiment_name"]),
        model=str(raw["model"]),
        epochs=int(raw["epochs"]),
        imgsz=int(raw["imgsz"]),
        batch=int(raw["batch"]),
        device=raw["device"],
        workers=int(raw["workers"]),
        patience=int(raw["patience"]),
        seed=int(raw["seed"]),
        optimizer=str(raw["optimizer"]),
        cache=bool(raw["cache"]),
        amp=bool(raw["amp"]),
        project=_resolve_path(raw["project"], base_dir),
        name=str(raw["name"]),
        exist_ok=bool(raw["exist_ok"]),
        confidence_threshold=float(raw["confidence_threshold"]),
        iou_threshold=float(raw["iou_threshold"]),
        expected_gpu=str(raw["expected_gpu"]),
        holdout_count=int(raw.get("holdout_count", 10)),
        degrees=float(raw.get("degrees", 0.0)),
        translate=float(raw.get("translate", 0.1)),
        scale=float(raw.get("scale", 0.2)),
        shear=float(raw.get("shear", 0.0)),
        perspective=float(raw.get("perspective", 0.0)),
        hsv_h=float(raw.get("hsv_h", 0.015)),
        hsv_s=float(raw.get("hsv_s", 0.2)),
        hsv_v=float(raw.get("hsv_v", 0.2)),
        fliplr=float(raw.get("fliplr", 0.0)),
        flipud=float(raw.get("flipud", 0.0)),
        mosaic=float(raw.get("mosaic", 0.0)),
        mixup=float(raw.get("mixup", 0.0)),
    )


def override_config(
    config: TrainConfig,
    *,
    dataset_root: str | Path | None = None,
    prepared_root: str | Path | None = None,
    model: str | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    name: str | None = None,
    degrees: float | None = None,
    scale: float | None = None,
    perspective: float | None = None,
    hsv_v: float | None = None,
    fliplr: float | None = None,
    mosaic: float | None = None,
) -> TrainConfig:
    updates: dict[str, Any] = {}
    if dataset_root is not None:
        updates["dataset_root"] = _resolve_path(dataset_root, Path.cwd())
    if prepared_root is not None:
        updates["prepared_root"] = _resolve_path(prepared_root, Path.cwd())
    if model is not None:
        updates["model"] = model
    if epochs is not None:
        updates["epochs"] = epochs
    if batch is not None:
        updates["batch"] = batch
    if name is not None:
        updates["name"] = name
    if degrees is not None:
        updates["degrees"] = degrees
    if scale is not None:
        updates["scale"] = scale
    if perspective is not None:
        updates["perspective"] = perspective
    if hsv_v is not None:
        updates["hsv_v"] = hsv_v
    if fliplr is not None:
        updates["fliplr"] = fliplr
    if mosaic is not None:
        updates["mosaic"] = mosaic
    return replace(config, **updates)
