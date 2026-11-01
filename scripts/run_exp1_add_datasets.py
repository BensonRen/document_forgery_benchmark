#!/usr/bin/env python3
"""
Add new-dataset results to existing E1 results.json without re-running
already-completed datasets.

Usage:
    python scripts/run_exp1_add_datasets.py \
        --methods TruFor MVSS-Net PSCC-Net IML-ViT \
        --datasets Tampered-IC13 ReceiptForgery

By default, only datasets that are missing from a method's existing results are run.
Use --force to re-run even if a dataset's results already exist.
"""
import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import (
    pixel_f1 as _pixel_f1,
    pixel_iou as _pixel_iou,
    pixel_auc as _pixel_auc,
    optimal_pixel_f1 as _optimal_pixel_f1,
    save_results,
    save_results_csv,
)
from run_exp1 import METHOD_LOADERS, load_dataset, _cleanup_method_imports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/exp1_general_on_docs.yaml"
    )
    parser.add_argument(
        "--methods", nargs="+",
        help="Methods to run. Default: all methods in config.",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="Limit to these dataset names. Default: run only missing datasets.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if dataset results already exist.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Override CUDA device, e.g. 'cuda:1'",
    )
    args = parser.parse_args()

    if args.device is not None:
        if args.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
        elif args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # Load existing results
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"[INFO] Loaded existing results for: {[k for k in all_results if not k.startswith('_')]}")
    else:
        all_results = {}

    methods_filter = set(args.methods) if args.methods else None
    datasets_filter = set(args.datasets) if args.datasets else None

    for method_cfg in config["methods"]:
        method_name = method_cfg["name"]
        if methods_filter and method_name not in methods_filter:
            continue

        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        existing = all_results.get(method_name, {})

        # Determine which datasets to process
        ds_to_run = []
        for ds_cfg in config["datasets"]:
            ds_name = ds_cfg["name"]
            if datasets_filter and ds_name not in datasets_filter:
                continue
            if not args.force and ds_name in existing:
                print(f"  [SKIP] {ds_name} (already in results; use --force to re-run)")
                continue
            ds_to_run.append(ds_cfg)

        if not ds_to_run:
            print(f"  [SKIP] No new datasets to process.")
            continue

        loader = METHOD_LOADERS.get(method_name)
        if loader is None:
            print(f"  [SKIP] No loader registered for {method_name}")
            continue

        # Clean up sys.path/modules to prevent import collisions between methods
        _cleanup_method_imports()

        try:
            result = loader(method_cfg)
            # Subprocess-based loaders return (predict, shutdown); in-process
            # loaders return just predict.
            if isinstance(result, tuple):
                predict, shutdown = result
            else:
                predict, shutdown = result, None
        except Exception as e:
            print(f"  [ERROR] Failed to load {method_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        per_img_dir = output_dir / "per_image"
        per_img_dir.mkdir(exist_ok=True)

        for ds_cfg in ds_to_run:
            ds_name = ds_cfg["name"]
            print(f"\n  Dataset: {ds_name}")

            pairs = load_dataset(ds_cfg)
            if not pairs:
                print(f"    [WARN] No image/mask pairs found for {ds_name}")
                continue

            metric_accum = {"pixel_f1": [], "pixel_iou": [], "pixel_auc": [], "optimal_f1": []}
            per_image_rows = []

            for img_path, mask_path in tqdm(pairs, desc=f"  {method_name}/{ds_name}"):
                try:
                    pred = predict(img_path)
                    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if gt is None:
                        print(f"    [WARN] Cannot read mask {mask_path}")
                        continue
                    gt = gt.astype(np.float32) / 255.0
                    if pred.shape != gt.shape:
                        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

                    f1 = _pixel_f1(pred, gt)
                    iou = _pixel_iou(pred, gt)
                    auc = _pixel_auc(pred, gt)
                    opt_f1, _ = _optimal_pixel_f1(pred, gt)

                    metric_accum["pixel_f1"].append(f1)
                    metric_accum["pixel_iou"].append(iou)
                    if not np.isnan(auc):
                        metric_accum["pixel_auc"].append(auc)
                    metric_accum["optimal_f1"].append(opt_f1)

                    per_image_rows.append({
                        "image": Path(img_path).name,
                        "pixel_f1": f1,
                        "pixel_iou": iou,
                        "pixel_auc": auc,
                        "optimal_f1": opt_f1,
                    })
                    del pred, gt
                except Exception as e:
                    print(f"    [ERROR] {img_path}: {e}")

            if any(metric_accum.values()):
                metrics = {
                    k: float(np.mean(v)) if v else float("nan")
                    for k, v in metric_accum.items()
                }
                existing[ds_name] = metrics
                print(f"  Results for {ds_name}: {json.dumps(metrics, indent=4)}")

                csv_path = per_img_dir / f"{method_name}_{ds_name}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["image", "pixel_f1", "pixel_iou", "pixel_auc", "optimal_f1"]
                    )
                    writer.writeheader()
                    writer.writerows(per_image_rows)
                print(f"  Per-image CSV: {csv_path}")

        # Merge-safe save: re-read current JSON to preserve concurrent writes
        if results_path.exists():
            with open(results_path) as _f:
                _current = json.load(_f)
        else:
            _current = {}
        _current[method_name] = existing
        all_results = _current
        # Save checkpoint after each method
        save_results(all_results, str(results_path))
        save_results_csv(all_results, str(output_dir / "results.csv"), experiment="E1")
        gc.collect()

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
