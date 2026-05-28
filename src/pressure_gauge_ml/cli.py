from __future__ import annotations

import argparse
from pathlib import Path

from pressure_gauge_ml.config import load_config, override_config
from pressure_gauge_ml.data import (
    prepare_yolo_dataset,
    summarize_coco_dataset,
    validate_coco_dataset,
)
from pressure_gauge_ml.evaluate import evaluate_model
from pressure_gauge_ml.infer import run_inference
from pressure_gauge_ml.monitor import write_monitoring_snapshot
from pressure_gauge_ml.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Pressure gauge ML pipeline")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to train config YAML")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override dataset root")
    parser.add_argument("--prepared-root", type=Path, default=None, help="Override prepared dataset root")
    parser.add_argument("--train-model", default=None, help="Override train model, for example rtdetr-l.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epoch count")
    parser.add_argument("--batch", type=int, default=None, help="Override training batch size")
    parser.add_argument("--name", default=None, help="Override run name")
    parser.add_argument("--degrees", type=float, default=None, help="Override rotation augmentation")
    parser.add_argument("--scale", type=float, default=None, help="Override scale augmentation")
    parser.add_argument("--perspective", type=float, default=None, help="Override perspective augmentation")
    parser.add_argument("--hsv-v", type=float, default=None, help="Override brightness/value augmentation")
    parser.add_argument("--fliplr", type=float, default=None, help="Override horizontal flip probability")
    parser.add_argument("--mosaic", type=float, default=None, help="Override mosaic augmentation probability")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Print COCO dataset summary")
    subparsers.add_parser("validate-data", help="Validate COCO annotations and image files")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Convert COCO annotations to Ultralytics detection format",
    )
    prepare_parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")

    train_parser = subparsers.add_parser("train", help="Train object detection model")
    train_parser.add_argument("--allow-cpu", action="store_true", help="Do not require CUDA before training")

    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune from an existing model checkpoint")
    finetune_parser.add_argument("--model", required=True, type=Path, help="Path to source .pt checkpoint")
    finetune_parser.add_argument("--allow-cpu", action="store_true", help="Do not require CUDA before training")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model on test split")
    eval_parser.add_argument("--model", required=True, type=Path)
    eval_parser.add_argument("--output", default="reports/test_metrics.json", type=Path)
    eval_parser.add_argument("--data-yaml", default=None, type=Path)

    infer_parser = subparsers.add_parser("infer", help="Run inference on images or a folder")
    infer_parser.add_argument("--model", required=True, type=Path)
    infer_parser.add_argument("--source", required=True, type=Path)
    infer_parser.add_argument("--output-dir", default="runs/predict/gauge", type=Path)
    infer_parser.add_argument("--predictions-json", default=None, type=Path)

    demo_parser = subparsers.add_parser("infer-demo", help="Run inference on holdout images")
    demo_parser.add_argument("--model", required=True, type=Path)
    demo_parser.add_argument("--output-dir", default="runs/predict/holdout_demo", type=Path)
    demo_parser.add_argument("--predictions-json", default="reports/holdout_predictions.json", type=Path)

    monitor_parser = subparsers.add_parser("monitor", help="Write monitoring snapshot")
    monitor_parser.add_argument("--output", default="reports/monitoring_snapshot.json", type=Path)

    args = parser.parse_args()
    config = override_config(
        load_config(args.config),
        dataset_root=args.dataset_root,
        prepared_root=args.prepared_root,
        model=args.train_model,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        degrees=args.degrees,
        scale=args.scale,
        perspective=args.perspective,
        hsv_v=args.hsv_v,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
    )

    if args.command == "summary":
        print(summarize_coco_dataset(config.dataset_root))
    elif args.command == "validate-data":
        errors = validate_coco_dataset(config.dataset_root, config.reports_root)
        if errors:
            raise SystemExit("\n".join(errors))
        print("Dataset validation passed.")
    elif args.command == "prepare":
        data_yaml = prepare_yolo_dataset(
            config.dataset_root,
            config.prepared_root,
            args.copy_images,
            reports_root=config.reports_root,
            holdout_count=config.holdout_count,
            seed=config.seed,
        )
        print(f"Ultralytics data config written to {data_yaml}")
    elif args.command == "train":
        result = train_model(config, require_cuda=not args.allow_cpu)
        print(f"Best model: {result.best_model}")
        print({"test_metrics": result.test_metrics, "holdout_metrics": result.holdout_metrics})
    elif args.command == "finetune":
        result = train_model(
            config,
            require_cuda=not args.allow_cpu,
            run_type="finetune",
            source_model=args.model,
        )
        print(f"Best model: {result.best_model}")
        print({"test_metrics": result.test_metrics, "holdout_metrics": result.holdout_metrics})
    elif args.command == "evaluate":
        data_yaml = args.data_yaml or config.data_yaml
        metrics = evaluate_model(args.model, data_yaml, args.output, imgsz=config.imgsz)
        print(metrics)
    elif args.command == "infer":
        output_dir = run_inference(
            args.model,
            args.source,
            args.output_dir,
            config.confidence_threshold,
            config.iou_threshold,
            args.predictions_json,
        )
        print(f"Predictions saved to {output_dir}")
    elif args.command == "infer-demo":
        output_dir = run_inference(
            args.model,
            config.prepared_root / "images" / "holdout",
            args.output_dir,
            config.confidence_threshold,
            config.iou_threshold,
            args.predictions_json,
        )
        print(f"Holdout demo predictions saved to {output_dir}")
    elif args.command == "monitor":
        snapshot = write_monitoring_snapshot(config.dataset_root, args.output)
        print(snapshot)


if __name__ == "__main__":
    main()
