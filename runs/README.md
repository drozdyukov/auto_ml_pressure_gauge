# runs

This folder is reserved for local training and inference outputs.

Expected local contents:

- `rtdetr/<run_name>/` - RT-DETR training runs, weights, metrics and Ultralytics plots.
- `predict/` - inference-demo outputs.

Run artifacts and `.pt` weights are intentionally not committed to Git because they are large. Final compact metrics are stored in `reports/`.
