from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from pressure_gauge_ml.config import TrainConfig
from pressure_gauge_ml.data import prepare_yolo_dataset, summarize_coco_dataset
from pressure_gauge_ml.evaluate import evaluate_model
from pressure_gauge_ml.hardware import assert_cuda_ready, get_nvidia_smi_summary


@dataclass(frozen=True)
class TrainingRunResult:
    best_model: Path
    run_dir: Path
    test_metrics: dict
    holdout_metrics: dict


def train_model(
    config: TrainConfig,
    require_cuda: bool = True,
    *,
    run_type: str = "train",
    source_model: Path | None = None,
) -> TrainingRunResult:
    import mlflow
    from ultralytics import RTDETR, YOLO

    if require_cuda:
        assert_cuda_ready(config.expected_gpu)

    data_yaml = prepare_yolo_dataset(
        config.dataset_root,
        config.prepared_root,
        reports_root=config.reports_root,
        holdout_count=config.holdout_count,
        seed=config.seed,
    )
    gpu = get_nvidia_smi_summary()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _run_name(config.name, run_type, timestamp)
    model_source = str(source_model or config.model)

    mlflow.set_tracking_uri((config.runs_root / "mlruns").resolve().as_uri())
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                **_training_params(config),
                **_augmentation_params(config),
                "run_type": run_type,
                "source_model": model_source,
                "expected_gpu": config.expected_gpu,
                "actual_gpu": gpu["name"] if gpu else "unknown",
            }
        )
        mlflow.log_dict(summarize_coco_dataset(config.dataset_root), "dataset_summary.json")

        model_cls = RTDETR if _is_rtdetr_source(model_source) else YOLO
        model = model_cls(model_source)
        results = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            device=config.device,
            workers=config.workers,
            patience=config.patience,
            seed=config.seed,
            optimizer=config.optimizer,
            cache=config.cache,
            amp=config.amp,
            project=str(config.project),
            name=run_name,
            exist_ok=False,
            **_augmentation_params(config),
        )
        run_dir = Path(results.save_dir)
        best_model = run_dir / "weights" / "best.pt"
        last_model = run_dir / "weights" / "last.pt"
        test_metrics, holdout_metrics = _evaluate_after_training(config, best_model, run_dir)
        _write_run_artifacts(
            config,
            data_yaml,
            run_dir,
            run_name,
            best_model,
            last_model,
            timestamp,
            run_type,
            model_source,
            test_metrics,
            holdout_metrics,
        )
        if best_model.exists():
            mlflow.log_artifact(str(best_model), artifact_path="model")
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metrics({f"holdout_{key}": value for key, value in holdout_metrics.items()})
        return TrainingRunResult(best_model, run_dir, test_metrics, holdout_metrics)


def _run_name(base_name: str, run_type: str, timestamp: str) -> str:
    if run_type == "finetune" and not base_name.startswith("finetune_"):
        base_name = f"finetune_{base_name}"
    return f"{base_name}_{timestamp}"


def _is_rtdetr_source(model_source: str) -> bool:
    path = Path(model_source)
    return model_source.lower().startswith("rtdetr") or any(
        part.lower().startswith("rtdetr") for part in path.parts
    )


def _training_params(config: TrainConfig) -> dict:
    return {
        "model": config.model,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": config.device,
        "workers": config.workers,
        "optimizer": config.optimizer,
        "amp": config.amp,
    }


def _augmentation_params(config: TrainConfig) -> dict:
    return {
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "shear": config.shear,
        "perspective": config.perspective,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        "fliplr": config.fliplr,
        "flipud": config.flipud,
        "mosaic": config.mosaic,
        "mixup": config.mixup,
    }


def _evaluate_after_training(
    config: TrainConfig,
    best_model: Path,
    run_dir: Path,
) -> tuple[dict, dict]:
    test_metrics = evaluate_model(best_model, config.data_yaml, run_dir / "test_metrics.json", imgsz=config.imgsz)
    holdout_metrics = evaluate_model(
        best_model,
        config.holdout_data_yaml,
        run_dir / "holdout_metrics.json",
        imgsz=config.imgsz,
    )
    config.reports_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "test_metrics.json", config.reports_root / "test_metrics.json")
    shutil.copy2(run_dir / "holdout_metrics.json", config.reports_root / "holdout_metrics.json")
    return test_metrics, holdout_metrics


def _write_run_artifacts(
    config: TrainConfig,
    data_yaml: Path,
    run_dir: Path,
    run_name: str,
    best_model: Path,
    last_model: Path,
    timestamp: str,
    run_type: str,
    source_model: str,
    test_metrics: dict,
    holdout_metrics: dict,
) -> None:
    config_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    config_dict["resolved_run_name"] = run_name
    config_dict["created_at"] = timestamp
    config_dict["run_type"] = run_type
    config_dict["source_model"] = source_model
    config_dict["augmentation"] = _augmentation_params(config)
    (run_dir / "training_config.yaml").write_text(
        yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    shutil.copy2(data_yaml, run_dir / "data.yaml")
    if (data_yaml.parent / "holdout_data.yaml").exists():
        shutil.copy2(data_yaml.parent / "holdout_data.yaml", run_dir / "holdout_data.yaml")
    class_list = ["base", "maximum", "minimum", "tip"]
    (run_dir / "classes.json").write_text(
        json.dumps(class_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "run_name": run_name,
        "run_type": run_type,
        "source_model": source_model,
        "created_at": timestamp,
        "best_model": str(best_model),
        "last_model": str(last_model),
        "data_yaml": str(run_dir / "data.yaml"),
        "holdout_data_yaml": str(run_dir / "holdout_data.yaml"),
        "test_metrics": test_metrics,
        "holdout_metrics": holdout_metrics,
        "augmentation": _augmentation_params(config),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
