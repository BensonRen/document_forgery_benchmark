#!/usr/bin/env python3
"""
Compute instance-level (bounding-box AP) metrics for DocForge-Bench methods.

Supports two GT types:
  1. Native box GT   — Tampered-IC13, ReceiptForgery (annotation files parsed directly)
  2. Pixel-mask GT   — DocTamper, T-SROIE, RTM, MixTamper, FSTS-1.5k
                       (GT pixel masks → CCA → bounding boxes per connected component)

Approach:
  1. Load GT bounding boxes (from annotation files or via CCA on GT pixel masks).
  2. For each method: predict soft mask → threshold → CCA → predicted boxes.
  3. Match predictions to GT with box-IoU; compute AP at IoU=0.5 and 0.5:0.05:0.95.

Output:
  results/instance_level/predictions/{Method}_{Dataset}/{stem}.npy  -- soft masks
  results/instance_level/instance_metrics.json                       -- AP results
  results/instance_level/summary.csv                                 -- flat CSV

Usage (from the repository root, venv activated):
  python scripts/compute_instance_metrics.py --method TruFor --gpu 0 --datasets Tampered-IC13
  python scripts/compute_instance_metrics.py --method TruFor --gpu 0 --datasets DocTamper T-SROIE RTM
  python scripts/compute_instance_metrics.py --method DocTamper --gpu 1 --datasets all
"""

import argparse
import ast
import csv
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EXPERIMENTS_DIR / "scripts"))

DATA_ROOT = EXPERIMENTS_DIR / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "instance_level"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INSTANCE_METRICS_JSON = RESULTS_DIR / "instance_metrics.json"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

# ── GT loading ───────────────────────────────────────────────────────

def load_tampered_ic13_gt():
    """Load GT bounding boxes for Tampered-IC13.
    Returns dict: img_stem -> list of (x1, y1, x2, y2) boxes.
    All 233 images are tampered; boxes from test_gt/*.txt files.
    """
    gt_dir = DATA_ROOT / "Tampered-IC13" / "Tampered-IC13" / "test_gt"
    img_dir = DATA_ROOT / "Tampered-IC13" / "Tampered-IC13" / "test_img"

    gt = {}
    for gt_path in sorted(gt_dir.glob("*.txt")):
        stem = gt_path.stem
        boxes = []
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                label = int(parts[4]) if len(parts) >= 5 else 1
                if label == 1:
                    boxes.append((x1, y1, x2, y2))
        if boxes:
            gt[stem] = boxes
    return gt


def load_receipt_forgery_gt():
    """Load GT bounding boxes for ReceiptForgery.
    Returns dict: img_stem -> list of (x1, y1, x2, y2) boxes.
    Only 35/218 images are forged; unforged images have empty box list.
    """
    manifest = DATA_ROOT / "ReceiptForgery" / "findit2" / "test.txt"
    gt = {}
    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["image"]
            stem = Path(img_name).stem
            forged = row["forged"].strip() == "1"
            boxes = []
            if forged:
                annot_str = row.get("forgery annotations", "").strip()
                if annot_str and annot_str != "0":
                    try:
                        annot = ast.literal_eval(annot_str)
                        regions = annot.get("regions", [])
                        for region in regions:
                            sa = region.get("shape_attributes", {})
                            if sa.get("name") == "rect":
                                rx = int(sa.get("x", 0))
                                ry = int(sa.get("y", 0))
                                rw = int(sa.get("width", 0))
                                rh = int(sa.get("height", 0))
                                boxes.append((rx, ry, rx + rw, ry + rh))
                    except Exception as e:
                        print(f"  [WARN] Parse error for {img_name}: {e}")
            gt[stem] = boxes  # empty list for authentic images
    return gt


# ── Pixel-mask → CCA → GT boxes ──────────────────────────────────────

def mask_to_gt_boxes(mask_path: Path, is_color_coded: bool = False,
                     min_area: int = 50) -> list:
    """Load a GT pixel mask and extract GT bounding boxes via CCA.

    Args:
        mask_path: path to the mask PNG
        is_color_coded: True for MixTamper (any non-zero BGR pixel = tampered)
        min_area: minimum component area in pixels to keep

    Returns:
        List of (x1, y1, x2, y2) bounding boxes.  Empty list for authentic images.
    """
    if is_color_coded:
        m = cv2.imread(str(mask_path))
        if m is None:
            return []
        binary = (np.any(m > 0, axis=2)).astype(np.uint8)
    else:
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return []
        binary = (m > 127).astype(np.uint8)

    if binary.max() == 0:
        return []  # authentic image — no tampered regions

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x1 = int(stats[i, cv2.CC_STAT_LEFT])
        y1 = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append((x1, y1, x1 + w, y1 + h))
    return boxes


def load_pixelmask_dataset_gt(img_dir: Path, img_glob: str,
                               mask_dir: Path, mask_ext: str = "png",
                               is_color_coded: bool = False,
                               max_samples: int = None,
                               split_file: Path = None,
                               min_area: int = 50) -> dict:
    """Load GT bounding boxes for a pixel-mask dataset via CCA.

    Reproduces the exact same sampling as run_exp1.load_dataset (seed=42).
    Returns dict: img_stem -> list of (x1, y1, x2, y2) GT boxes.
    """
    import random as _random

    # Load split filter if provided
    split_stems = None
    if split_file is not None and split_file.exists():
        with open(split_file) as f:
            split_stems = set(line.strip() for line in f if line.strip())
        print(f"  [GT] Split file: {split_file} ({len(split_stems)} stems)")

    # Collect image stems matching the dataset
    stems = []
    for img_path in sorted(img_dir.glob(img_glob)):
        if split_stems is not None and img_path.stem not in split_stems:
            continue
        mask_path = mask_dir / f"{img_path.stem}.{mask_ext}"
        if mask_path.exists():
            stems.append(img_path.stem)

    # Reproduce exact sampling from run_exp1.load_dataset
    if max_samples is not None and len(stems) > max_samples:
        _random.seed(42)
        stems = _random.sample(stems, max_samples)
        stems = sorted(stems)

    print(f"  [GT] Loading pixel-mask GT for {len(stems)} images via CCA...")
    gt = {}
    for stem in tqdm(stems, desc="  GT CCA"):
        mask_path = mask_dir / f"{stem}.{mask_ext}"
        gt[stem] = mask_to_gt_boxes(mask_path, is_color_coded=is_color_coded,
                                     min_area=min_area)
    return gt


# ── Instance extraction from soft mask ───────────────────────────────

def extract_instances_from_mask(soft_mask: np.ndarray, threshold: float = 0.5,
                                min_area: int = 50):
    """Run CCA on thresholded soft mask, return list of (box, score) per instance.

    Args:
        soft_mask: float32 H×W in [0,1]
        threshold: binarization threshold
        min_area: minimum pixel area to keep a component
    Returns:
        List of (x1, y1, x2, y2, score) — score = max pixel value in component
    """
    binary = (soft_mask >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    instances = []
    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        x2, y2 = x1 + w, y1 + h
        component_scores = soft_mask[labels == i]
        score = float(component_scores.max())
        instances.append((x1, y1, x2, y2, score))

    return instances


# ── Box IoU ──────────────────────────────────────────────────────────

def box_iou(box_a, box_b):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Average Precision computation ────────────────────────────────────

def compute_ap(all_predictions, all_gt_boxes, iou_threshold: float = 0.5):
    """Compute AP (PASCAL VOC style) across all images.

    Args:
        all_predictions: list of (img_id, score, box) sorted by descending score
        all_gt_boxes: dict {img_id: [(x1,y1,x2,y2), ...]}
        iou_threshold: IoU threshold for TP

    Returns:
        dict with ap, precision, recall, n_gt, n_pred
    """
    # Sort all detections by confidence descending
    all_predictions = sorted(all_predictions, key=lambda x: -x[1])

    n_gt = sum(len(boxes) for boxes in all_gt_boxes.values())
    if n_gt == 0:
        return {"ap": float("nan"), "n_gt": 0, "n_pred": len(all_predictions)}

    # Track which GT boxes have been matched
    gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in all_gt_boxes.items()}

    tp_arr = []
    fp_arr = []

    for img_id, score, pred_box in all_predictions:
        gt_boxes = all_gt_boxes.get(img_id, [])
        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0 and not gt_matched[img_id][best_j]:
            tp_arr.append(1)
            fp_arr.append(0)
            gt_matched[img_id][best_j] = True
        else:
            tp_arr.append(0)
            fp_arr.append(1)

    tp_cum = np.cumsum(tp_arr, dtype=np.float64)
    fp_cum = np.cumsum(fp_arr, dtype=np.float64)
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # Append sentinel values
    recall = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])

    # PASCAL VOC 11-point interpolation or area-under-curve
    # np.trapz removed in NumPy 2.x; use np.trapezoid with fallback
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    ap = float(_trapz(precision, recall))

    return {
        "ap": ap,
        "n_gt": n_gt,
        "n_pred": len(all_predictions),
        "n_tp": int(tp_cum[-1]) if len(tp_cum) else 0,
    }


def compute_coco_ap(all_predictions, all_gt_boxes):
    """Compute COCO mAP: mean AP over IoU thresholds 0.5:0.05:0.95."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for t in iou_thresholds:
        result = compute_ap(all_predictions, all_gt_boxes, iou_threshold=float(t))
        if not np.isnan(result["ap"]):
            aps.append(result["ap"])
    return float(np.mean(aps)) if aps else float("nan")


# ── Per-image precision/recall/F1 at IoU threshold ───────────────────

def per_image_instance_metrics(pred_instances, gt_boxes, iou_threshold: float = 0.5):
    """Compute instance-level P/R/F1 for a single image.

    pred_instances: list of (x1, y1, x2, y2, score)
    gt_boxes: list of (x1, y1, x2, y2)
    """
    if len(gt_boxes) == 0 and len(pred_instances) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_pred": 0, "n_gt": 0, "n_tp": 0}
    if len(gt_boxes) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "n_pred": len(pred_instances),
                "n_gt": 0, "n_tp": 0}
    if len(pred_instances) == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "n_pred": 0,
                "n_gt": len(gt_boxes), "n_tp": 0}

    gt_matched = [False] * len(gt_boxes)
    tp = 0
    for pred in pred_instances:
        pred_box = pred[:4]
        best_iou = 0.0
        best_j = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0 and not gt_matched[best_j]:
            tp += 1
            gt_matched[best_j] = True

    precision = tp / len(pred_instances) if pred_instances else 0.0
    recall = tp / len(gt_boxes)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_pred": len(pred_instances), "n_gt": len(gt_boxes), "n_tp": tp}


# ── Main evaluation loop ─────────────────────────────────────────────

def _pixelmask_gt_loader(img_dir, img_glob, mask_dir, mask_ext="png",
                          is_color_coded=False, max_samples=None,
                          split_file=None, min_area=50):
    """Return a zero-argument callable that loads pixel-mask GT for a dataset."""
    def _load():
        return load_pixelmask_dataset_gt(
            img_dir=img_dir, img_glob=img_glob,
            mask_dir=mask_dir, mask_ext=mask_ext,
            is_color_coded=is_color_coded,
            max_samples=max_samples,
            split_file=split_file,
            min_area=min_area,
        )
    return _load


# All 7 evaluable datasets (FantasyID excluded — image-level only, no pixel GT).
# gt_loader is a zero-argument callable returning {stem: [(x1,y1,x2,y2), ...]}
DATASET_CONFIGS = {
    # ── Native box-GT datasets ──────────────────────────────────────────
    "Tampered-IC13": {
        "img_dir": DATA_ROOT / "Tampered-IC13" / "Tampered-IC13" / "test_img",
        "img_glob": "*.jpg",
        "gt_loader": load_tampered_ic13_gt,
    },
    "ReceiptForgery": {
        "img_dir": DATA_ROOT / "ReceiptForgery" / "findit2" / "test",
        "img_glob": "*.png",
        "gt_loader": load_receipt_forgery_gt,
    },
    # ── Pixel-mask → CCA → GT boxes ─────────────────────────────────────
    "DocTamper": {
        "img_dir": DATA_ROOT / "DocTamper" / "test" / "images",
        "img_glob": "*.png",
        "gt_loader": _pixelmask_gt_loader(
            img_dir=DATA_ROOT / "DocTamper" / "test" / "images",
            img_glob="*.png",
            mask_dir=DATA_ROOT / "DocTamper" / "test" / "masks",
            max_samples=1000,   # same cap as benchmark
        ),
    },
    "T-SROIE": {
        "img_dir": DATA_ROOT / "T-SROIE" / "test",
        "img_glob": "*.jpg",
        "gt_loader": _pixelmask_gt_loader(
            img_dir=DATA_ROOT / "T-SROIE" / "test",
            img_glob="*.jpg",
            mask_dir=DATA_ROOT / "T-SROIE" / "test" / "masks",
        ),
    },
    "RTM": {
        "img_dir": DATA_ROOT / "RealTextManipulation" / "JPEGImages",
        "img_glob": "*.jpg",
        "gt_loader": _pixelmask_gt_loader(
            img_dir=DATA_ROOT / "RealTextManipulation" / "JPEGImages",
            img_glob="*.jpg",
            mask_dir=DATA_ROOT / "RealTextManipulation" / "SegmentationClass",
            max_samples=1000,
            split_file=DATA_ROOT / "RealTextManipulation" / "test_tampered.txt",
        ),
    },
    "MixTamper": {
        "img_dir": DATA_ROOT / "MixTamper" / "img",
        "img_glob": "*.png",
        "gt_loader": _pixelmask_gt_loader(
            img_dir=DATA_ROOT / "MixTamper" / "img",
            img_glob="*.png",
            mask_dir=DATA_ROOT / "MixTamper" / "mask",
            is_color_coded=True,    # BGR (128,0,200) = tampered
            max_samples=1000,
        ),
    },
    "FSTS-1.5k": {
        "img_dir": DATA_ROOT / "FSTS-1.5k" / "images",
        "img_glob": "*.png",
        "gt_loader": _pixelmask_gt_loader(
            img_dir=DATA_ROOT / "FSTS-1.5k" / "images",
            img_glob="*.png",
            mask_dir=DATA_ROOT / "FSTS-1.5k" / "masks",
        ),
    },
}


def evaluate_method_on_dataset(method_name, predict_fn, dataset_name, ds_cfg,
                                gt_boxes_dict, save_masks=True):
    """Run a method on a dataset, extract instances, compute AP.

    Returns:
        dict with AP metrics and per-image results
    """
    img_dir = ds_cfg["img_dir"]
    all_img_paths = sorted(img_dir.glob(ds_cfg["img_glob"]))
    # For sampled datasets (DocTamper, RTM, MixTamper), restrict to the stems
    # that appear in gt_boxes_dict (which already has the correct sampled subset).
    gt_stems = set(gt_boxes_dict.keys())
    if len(gt_stems) < len(all_img_paths):
        img_paths = [p for p in all_img_paths if p.stem in gt_stems]
    else:
        img_paths = all_img_paths

    # Directory for saving soft masks
    mask_save_dir = RESULTS_DIR / "predictions" / f"{method_name}_{dataset_name}"
    if save_masks:
        mask_save_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []   # (img_id, score, box) for AP computation
    per_image_rows = []

    for img_path in tqdm(img_paths, desc=f"  {method_name}/{dataset_name}"):
        stem = img_path.stem

        # Load or compute soft mask
        mask_npy_path = mask_save_dir / f"{stem}.npy"
        if save_masks and mask_npy_path.exists():
            soft_mask = np.load(str(mask_npy_path))
        else:
            try:
                soft_mask = predict_fn(str(img_path))
            except Exception as e:
                print(f"    [ERROR] {img_path.name}: {e}")
                continue
            if save_masks:
                np.save(str(mask_npy_path), soft_mask.astype(np.float32))

        # Extract predicted instances via CCA
        pred_instances = extract_instances_from_mask(soft_mask, threshold=0.5, min_area=50)

        # GT boxes for this image (empty list for authentic images)
        gt_boxes = gt_boxes_dict.get(stem, [])

        # Accumulate for AP
        for (x1, y1, x2, y2, score) in pred_instances:
            all_predictions.append((stem, score, (x1, y1, x2, y2)))

        # Per-image instance metrics at IoU=0.5
        pm = per_image_instance_metrics(pred_instances, gt_boxes, iou_threshold=0.5)
        per_image_rows.append({
            "image": img_path.name,
            "n_gt_boxes": len(gt_boxes),
            "n_pred_instances": pm["n_pred"],
            "n_tp": pm["n_tp"],
            "precision_50": pm["precision"],
            "recall_50": pm["recall"],
            "f1_50": pm["f1"],
        })

    # Dataset-level AP
    ap_50 = compute_ap(all_predictions, gt_boxes_dict, iou_threshold=0.5)
    ap_coco = compute_coco_ap(all_predictions, gt_boxes_dict)

    # Mean per-image P/R/F1
    valid_rows = [r for r in per_image_rows if r["n_gt_boxes"] > 0 or r["n_pred_instances"] > 0]
    mean_p = float(np.mean([r["precision_50"] for r in per_image_rows])) if per_image_rows else float("nan")
    mean_r = float(np.mean([r["recall_50"] for r in per_image_rows])) if per_image_rows else float("nan")
    mean_f1 = float(np.mean([r["f1_50"] for r in per_image_rows])) if per_image_rows else float("nan")

    results = {
        "ap_50": ap_50["ap"],
        "ap_coco": ap_coco,
        "n_gt_total": ap_50["n_gt"],
        "n_pred_total": ap_50["n_pred"],
        "n_tp_total": ap_50.get("n_tp", 0),
        "mean_precision_50": mean_p,
        "mean_recall_50": mean_r,
        "mean_f1_50": mean_f1,
        "n_images": len(img_paths),
    }

    # Save per-image CSV
    csv_path = RESULTS_DIR / f"per_image_{method_name}_{dataset_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "n_gt_boxes", "n_pred_instances", "n_tp",
            "precision_50", "recall_50", "f1_50"
        ])
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"  Per-image CSV saved: {csv_path}")

    return results


def run(methods, gpu=0):
    """Run instance-level evaluation for specified methods."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"Using GPU: {gpu}")

    # Load method loaders (run_exp2 imports run_exp1 loaders too)
    sys.path.insert(0, str(EXPERIMENTS_DIR / "scripts"))
    from run_exp2 import METHOD_LOADERS
    import yaml

    # Load both configs and merge method configs
    with open(EXPERIMENTS_DIR / "configs" / "exp2_doc_specific.yaml") as f:
        config2 = yaml.safe_load(f)
    with open(EXPERIMENTS_DIR / "configs" / "exp1_general_on_docs.yaml") as f:
        config1 = yaml.safe_load(f)

    # Load GT annotations for each dataset (box-GT or pixel-mask CCA)
    print("Loading GT annotations...")
    gt_data = {}
    for ds_name, ds_cfg in DATASET_CONFIGS.items():
        print(f"  {ds_name}...")
        gt_data[ds_name] = ds_cfg["gt_loader"]()
        n_forged = sum(1 for v in gt_data[ds_name].values() if v)
        print(f"    {len(gt_data[ds_name])} images, {n_forged} with GT boxes")

    # Load existing results
    if INSTANCE_METRICS_JSON.exists():
        with open(INSTANCE_METRICS_JSON) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Build method cfg lookup from both configs (exp2 takes precedence)
    method_cfgs = {m["name"]: m for m in config1["methods"]}
    method_cfgs.update({m["name"]: m for m in config2["methods"]})

    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        loader_fn = METHOD_LOADERS.get(method_name)
        if loader_fn is None:
            print(f"  [SKIP] Unknown method: {method_name}")
            continue

        method_cfg = method_cfgs.get(method_name, {})

        try:
            loaded = loader_fn(method_cfg)
        except Exception as e:
            print(f"  [ERROR] Loading {method_name}: {e}")
            continue

        # Subprocess-based loaders return (predict, shutdown); in-process
        # loaders return just the predict callable.
        shutdown_fn = None
        if isinstance(loaded, tuple):
            predict, shutdown_fn = loaded
            print(f"  [INFO] Subprocess-worker loader (predict + shutdown)")
        else:
            predict = loaded

        if method_name not in all_results:
            all_results[method_name] = {}

        for dataset_name, ds_cfg in DATASET_CONFIGS.items():
            if dataset_name in all_results.get(method_name, {}):
                # Check if predictions are already saved
                mask_dir = RESULTS_DIR / "predictions" / f"{method_name}_{dataset_name}"
                existing_masks = len(list(mask_dir.glob("*.npy"))) if mask_dir.exists() else 0
                n_images = len(list(ds_cfg["img_dir"].glob(ds_cfg["img_glob"])))
                if existing_masks >= n_images:
                    print(f"  [SKIP] {method_name}/{dataset_name} already computed")
                    continue

            print(f"\n  Dataset: {dataset_name}")
            gt_boxes = gt_data[dataset_name]
            results = evaluate_method_on_dataset(
                method_name, predict, dataset_name, ds_cfg, gt_boxes, save_masks=True
            )
            all_results[method_name][dataset_name] = results
            print(f"  AP@50: {results['ap_50']:.4f}  AP-COCO: {results['ap_coco']:.4f}  "
                  f"mean-F1@50: {results['mean_f1_50']:.4f}")

            # Save checkpoint after each dataset
            with open(INSTANCE_METRICS_JSON, "w") as f:
                json.dump(all_results, f, indent=2)

        if shutdown_fn is not None:
            try:
                shutdown_fn()
            except Exception:
                pass
        del predict
        gc.collect()

    # Save final results
    with open(INSTANCE_METRICS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {INSTANCE_METRICS_JSON}")

    # Save summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dataset", "ap_50", "ap_coco", "mean_f1_50",
                         "mean_precision_50", "mean_recall_50",
                         "n_gt_total", "n_pred_total", "n_tp_total", "n_images"])
        for method, datasets in sorted(all_results.items()):
            for dataset, metrics in sorted(datasets.items()):
                writer.writerow([
                    method, dataset,
                    metrics.get("ap_50", "nan"),
                    metrics.get("ap_coco", "nan"),
                    metrics.get("mean_f1_50", "nan"),
                    metrics.get("mean_precision_50", "nan"),
                    metrics.get("mean_recall_50", "nan"),
                    metrics.get("n_gt_total", 0),
                    metrics.get("n_pred_total", 0),
                    metrics.get("n_tp_total", 0),
                    metrics.get("n_images", 0),
                ])
    print(f"Summary CSV saved to {SUMMARY_CSV}")


ALL_DATASETS = ["Tampered-IC13", "ReceiptForgery", "DocTamper", "T-SROIE", "RTM",
                "MixTamper", "FSTS-1.5k"]


def main():
    parser = argparse.ArgumentParser(description="Instance-level AP evaluation for DocForge-Bench")
    parser.add_argument("--method", nargs="+", required=True,
                        help="Method name(s) to evaluate")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help="Datasets to evaluate on (or 'all')")
    args = parser.parse_args()

    # 'all' expands to every dataset
    requested = ALL_DATASETS if args.datasets == ["all"] else args.datasets

    # Filter DATASET_CONFIGS to requested datasets
    global DATASET_CONFIGS
    DATASET_CONFIGS = {k: v for k, v in DATASET_CONFIGS.items() if k in requested}
    print(f"Datasets: {list(DATASET_CONFIGS.keys())}")

    run(args.method, gpu=args.gpu)


if __name__ == "__main__":
    main()
