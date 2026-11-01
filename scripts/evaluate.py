#!/usr/bin/env python3
"""
evaluate.py — Unified metrics computation for DocForge-Bench.

Supports:
  - Pixel-level localization: F1, IoU, AUC
  - Image-level detection: AUC, F1, Accuracy, AP
  - Signature verification: EER, AUC
  - Text detection: AUROC, F1, TPR@FPR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ══════════════════════════════════════════════════════════════════════
# Pixel-level metrics
# ══════════════════════════════════════════════════════════════════════

def pixel_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Compute pixel-level F1 score at a given threshold.

    Returns NaN for images with no ground-truth tampered pixels (empty GT),
    matching the NaN convention used by pixel_iou and pixel_auc.
    Callers should filter NaN before aggregating (same as IoU/AUC).
    """
    gt_bin = (gt_mask > 0).ravel()
    if gt_bin.sum() == 0:
        return float("nan")
    pred_bin = (pred_mask >= threshold).ravel()
    # Direct numpy computation avoids sklearn per-call overhead (~20ms on 2MP arrays)
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Compute pixel-level Intersection over Union.

    Returns NaN for images with no ground-truth tampered pixels (empty GT),
    as IoU is undefined in that case. Callers should filter NaN before aggregating.
    """
    pred_bin = (pred_mask >= threshold).ravel()
    gt_bin = (gt_mask > 0).ravel()
    if gt_bin.sum() == 0:
        return float("nan")
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    # union == 0 is impossible here since gt_bin.sum() > 0
    return float(intersection / union)


def pixel_auc(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute pixel-level AUC (ROC)."""
    pred_flat = pred_mask.ravel().astype(np.float64)
    gt_flat = (gt_mask > 0).ravel().astype(np.uint8)
    if gt_flat.sum() == 0 or gt_flat.sum() == len(gt_flat):
        return float("nan")
    return float(roc_auc_score(gt_flat, pred_flat))


# Oracle-F1 threshold grid: 50 linearly spaced thresholds strictly inside (0, 1),
# as defined in the paper (Section 6.1 and Appendix A).
ORACLE_THRESHOLDS = np.linspace(0.0, 1.0, 52)[1:-1]


def _f1_on_grid(pred_flat: np.ndarray, gt_bin: np.ndarray,
                thresholds: np.ndarray) -> np.ndarray:
    """F1 of (pred >= t) vs gt for every t in `thresholds`, from a single sort."""
    order = np.argsort(pred_flat, kind="stable")
    pred_sorted = pred_flat[order]
    # pos_from[i] = number of tampered pixels among the pixels ranked i..n-1 (ascending)
    pos_from = np.concatenate([np.cumsum(gt_bin[order][::-1])[::-1], [0]])
    n_pos = int(gt_bin.sum())
    idx = np.searchsorted(pred_sorted, thresholds, side="left")  # first index with pred >= t
    n_pred = len(pred_flat) - idx
    tp = pos_from[idx]
    denom = n_pred + n_pos
    return np.divide(2.0 * tp, denom, out=np.zeros(len(thresholds)), where=denom > 0)


def compute_all_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray,
                              threshold: float = 0.5) -> dict:
    """Compute the four pixel-level metrics for one image.

    pixel_f1 / pixel_iou : at the fixed threshold (default 0.5)
    pixel_auc            : ROC AUC over all pixels (sklearn roc_auc_score; ties count half)
    optimal_f1           : best F1 over ORACLE_THRESHOLDS (the paper's Oracle-F1)

    Returns NaN for every metric when the mask has no tampered pixels, and NaN for
    pixel_auc / optimal_f1 when every pixel is tampered.
    """
    gt_bin = (gt_mask > 0).ravel()
    n_pos = int(gt_bin.sum())
    n = len(gt_bin)

    if n_pos == 0:
        return {"pixel_f1": float("nan"), "pixel_iou": float("nan"),
                "pixel_auc": float("nan"), "optimal_f1": float("nan")}

    pred_flat = pred_mask.ravel().astype(np.float32)

    # ── Fixed-threshold metrics (pixel_f1, pixel_iou) ──
    pred_bin = pred_flat >= threshold
    tp0 = int(np.logical_and(pred_bin, gt_bin).sum())
    fp0 = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn0 = int(np.logical_and(~pred_bin, gt_bin).sum())
    d0 = 2 * tp0 + fp0 + fn0
    pf1 = float(2 * tp0 / d0) if d0 > 0 else 0.0

    union = tp0 + fp0 + fn0
    piou = float(tp0 / union) if union > 0 else float("nan")

    # ── Threshold-independent metrics ──
    if n_pos == n:
        auc_val = float("nan")
        opt_f1 = float("nan")
    else:
        auc_val = float(roc_auc_score(gt_bin, pred_flat))
        opt_f1 = float(_f1_on_grid(pred_flat, gt_bin, ORACLE_THRESHOLDS).max())

    return {"pixel_f1": pf1, "pixel_iou": piou,
            "pixel_auc": auc_val, "optimal_f1": opt_f1}


def optimal_pixel_f1(pred_mask: np.ndarray, gt_mask: np.ndarray,
                     thresholds: int = 50) -> Tuple[float, float]:
    """Best F1 over `thresholds` linearly spaced thresholds strictly inside (0, 1).

    Returns (best_f1, best_threshold), or (NaN, NaN) when the mask has no tampered
    pixels, matching the NaN convention of pixel_iou and pixel_auc.
    """
    gt_bin = (gt_mask > 0).ravel()
    if gt_bin.sum() == 0:
        return float("nan"), float("nan")
    grid = np.linspace(0.0, 1.0, thresholds + 2)[1:-1]
    f1_arr = _f1_on_grid(pred_mask.ravel().astype(np.float32), gt_bin, grid)
    best = int(f1_arr.argmax())
    return float(f1_arr[best]), float(grid[best])


# ══════════════════════════════════════════════════════════════════════
# Image-level metrics
# ══════════════════════════════════════════════════════════════════════

def image_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level AUC (ROC)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def image_f1(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """Image-level F1 at a given threshold."""
    preds = (scores >= threshold).astype(np.uint8)
    return float(f1_score(labels, preds, zero_division=0))


def image_accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """Image-level accuracy."""
    preds = (scores >= threshold).astype(np.uint8)
    return float(accuracy_score(labels, preds))


def image_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level Average Precision."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


# ══════════════════════════════════════════════════════════════════════
# Signature verification metrics
# ══════════════════════════════════════════════════════════════════════

def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate (EER)."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    # Find the point where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


# ══════════════════════════════════════════════════════════════════════
# Text detection metrics
# ══════════════════════════════════════════════════════════════════════

def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray,
               target_fpr: float = 0.05) -> float:
    """Compute TPR at a specified FPR threshold."""
    fpr, tpr, _ = roc_curve(labels, scores)
    # Interpolate TPR at the target FPR
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(fpr) - 1))
    return float(tpr[idx])


# ══════════════════════════════════════════════════════════════════════
# Batch evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_pixel_localization(pred_masks: list[np.ndarray],
                                gt_masks: list[np.ndarray]) -> dict:
    """Evaluate a batch of pixel-level predictions."""
    results = {"pixel_f1": [], "pixel_iou": [], "pixel_auc": [], "optimal_f1": []}
    for pred, gt in zip(pred_masks, gt_masks):
        f1_val = pixel_f1(pred, gt)
        if not np.isnan(f1_val):
            results["pixel_f1"].append(f1_val)
        iou_val = pixel_iou(pred, gt)
        if not np.isnan(iou_val):
            results["pixel_iou"].append(iou_val)
        auc_val = pixel_auc(pred, gt)
        if not np.isnan(auc_val):
            results["pixel_auc"].append(auc_val)
        opt_f1, _ = optimal_pixel_f1(pred, gt)
        if not np.isnan(opt_f1):
            results["optimal_f1"].append(opt_f1)

    return {k: float(np.mean(v)) if v else float("nan") for k, v in results.items()}


def evaluate_image_detection(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate image-level detection scores."""
    return {
        "image_auc": image_auc(scores, labels),
        "image_f1": image_f1(scores, labels),
        "image_accuracy": image_accuracy(scores, labels),
        "image_ap": image_ap(scores, labels),
    }


def evaluate_signature(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate signature verification scores."""
    return {
        "eer": equal_error_rate(scores, labels),
        "auc": image_auc(scores, labels),
    }


def evaluate_text_detection(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate binary text/score classification (AUROC, F1, TPR@FPR)."""
    return {
        "auroc": image_auc(scores, labels),
        "f1": image_f1(scores, labels),
        "tpr_at_fpr_5": tpr_at_fpr(scores, labels, target_fpr=0.05),
        "tpr_at_fpr_1": tpr_at_fpr(scores, labels, target_fpr=0.01),
    }


# ══════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════

def load_masks(directory: str, suffix: str = ".npy") -> list[np.ndarray]:
    """Load mask arrays from a directory."""
    masks = []
    for f in sorted(Path(directory).glob(f"*{suffix}")):
        masks.append(np.load(f))
    return masks


def save_results(results: dict, output_path: str) -> None:
    """Save evaluation results as JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def save_results_csv(results: dict, csv_path: str, experiment: str = "") -> None:
    """Save aggregate results as a flat CSV: experiment,method,dataset,metric,value.

    Skips special keys starting with '_' (e.g. _summary dicts in E2).
    """
    import csv as _csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["experiment", "method", "dataset", "metric", "value"])
        for method, datasets in results.items():
            if method.startswith("_") or not isinstance(datasets, dict):
                continue
            for dataset, metrics in datasets.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    if metric == "is_document_specific":
                        continue
                    writer.writerow([experiment, method, dataset, metric, value])
    print(f"CSV results saved to {csv_path}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DocForge-Bench unified evaluation")
    parser.add_argument("--mode", required=True,
                        choices=["pixel", "image", "signature", "text"],
                        help="Evaluation mode")
    parser.add_argument("--pred-dir", type=str,
                        help="Directory with prediction .npy files (pixel mode)")
    parser.add_argument("--gt-dir", type=str,
                        help="Directory with ground truth .npy files (pixel mode)")
    parser.add_argument("--scores-file", type=str,
                        help="JSON file with scores and labels (image/sig/text mode)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for results")

    args = parser.parse_args()

    if args.mode == "pixel":
        assert args.pred_dir and args.gt_dir, "Need --pred-dir and --gt-dir for pixel mode"
        pred_masks = load_masks(args.pred_dir)
        gt_masks = load_masks(args.gt_dir)
        assert len(pred_masks) == len(gt_masks), \
            f"Mismatch: {len(pred_masks)} predictions vs {len(gt_masks)} ground truths"
        results = evaluate_pixel_localization(pred_masks, gt_masks)

    elif args.mode in ("image", "signature", "text"):
        assert args.scores_file, "Need --scores-file for image/signature/text mode"
        with open(args.scores_file) as f:
            data = json.load(f)
        scores = np.array(data["scores"], dtype=np.float64)
        labels = np.array(data["labels"], dtype=np.int32)

        if args.mode == "image":
            results = evaluate_image_detection(scores, labels)
        elif args.mode == "signature":
            results = evaluate_signature(scores, labels)
        else:
            results = evaluate_text_detection(scores, labels)

    save_results(results, args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
