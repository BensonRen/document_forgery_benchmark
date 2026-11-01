#!/usr/bin/env python3
"""
Generate a grid of prediction visualizations for all E1 methods on representative
images from each benchmark dataset.

Layout:
  rows    = one sample image per dataset
  columns = Original | Ground Truth | TruFor | ManTraNet | MVSS-Net |
            CAT-Net | PSCC-Net | IML-ViT

Predictions are cached as .npy files in results/visualizations/cache/ so the
script can be re-run without re-running inference.  Delete cache files to force
a fresh inference run.

Usage (run from the repository root with venv active):
    CUDA_VISIBLE_DEVICES=1 python scripts/visualize_predictions.py

SAFIRE is intentionally excluded because it runs on GPU0 simultaneously.
Add it later by passing --include-safire once its FantasyID run finishes.
"""

import argparse
import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

_EXP_ROOT = Path(__file__).resolve().parent.parent  # repository root
sys.path.insert(0, str(_EXP_ROOT / "scripts"))

# ── Sample images ─────────────────────────────────────────────────────────────
# Chosen from per_image CSVs: high TruFor AUC and visible tamper region.
SAMPLES = [
    {
        "label": "DocTamper",
        "image": str(_EXP_ROOT / "data/DocTamper/test/images/000002321.png"),
        "mask":  str(_EXP_ROOT / "data/DocTamper/test/masks/000002321.png"),
    },
    {
        "label": "T-SROIE",
        "image": str(_EXP_ROOT / "data/T-SROIE/test/X51006619328.jpg"),
        "mask":  str(_EXP_ROOT / "data/T-SROIE/test/masks/X51006619328.png"),
    },
    {
        "label": "Tampered-IC13",
        "image": str(_EXP_ROOT / "data/Tampered-IC13/Tampered-IC13/test_img/img_145.jpg"),
        "mask":  str(_EXP_ROOT / "data/Tampered-IC13/Tampered-IC13/test_masks/img_145.png"),
    },
    {
        "label": "RTM",
        "image": str(_EXP_ROOT / "data/RealTextManipulation/JPEGImages/cover_0979.jpg"),
        "mask":  str(_EXP_ROOT / "data/RealTextManipulation/SegmentationClass/cover_0979.png"),
    },
]

# ── Method configs (from exp1_general_on_docs.yaml) ──────────────────────────
_E1_CONFIG_PATH = str(_EXP_ROOT / "configs/exp1_general_on_docs.yaml")

def _load_e1_method_cfg(name):
    with open(_E1_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for m in cfg["methods"]:
        if m["name"] == name:
            return m
    raise KeyError(f"Method {name!r} not found in E1 config")


METHODS_TO_RUN = [
    "TruFor",
    "ManTraNet",
    "MVSS-Net",
    "CAT-Net",
    "PSCC-Net",
    "IML-ViT",
]

# ── Cache and output dirs ─────────────────────────────────────────────────────
CACHE_DIR = _EXP_ROOT / "results/visualizations/cache"
OUT_DIR   = _EXP_ROOT / "results/visualizations"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(method_name, sample_label):
    safe_label = sample_label.replace("/", "_").replace(" ", "_")
    safe_method = method_name.replace("-", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe_method}__{safe_label}.npy"


def _run_method(method_name, force=False):
    """Load the method, run it on all SAMPLES, cache predictions. Return dict."""
    results = {}
    missing = []
    for s in SAMPLES:
        cp = _cache_path(method_name, s["label"])
        if cp.exists() and not force:
            results[s["label"]] = np.load(str(cp))
            print(f"  [cache] {method_name}/{s['label']}")
        else:
            missing.append(s)

    if not missing:
        return results

    print(f"\n── Loading {method_name} ──")
    import run_exp1 as _r1  # noqa — side-effect: defines _cleanup_method_imports etc.

    loader_fn = _r1.METHOD_LOADERS[method_name]
    cfg = _load_e1_method_cfg(method_name)

    loaded = loader_fn(cfg)
    # Some loaders return (predict, shutdown), others return predict directly
    if callable(loaded):
        predict, shutdown = loaded, lambda: None
    else:
        predict, shutdown = loaded

    try:
        for s in missing:
            print(f"  running {method_name}/{s['label']} …", end=" ", flush=True)
            try:
                pred = predict(s["image"])
            except Exception as e:
                print(f"ERROR: {e}")
                pred = np.zeros((64, 64), dtype=np.float32)
            pred = pred.astype(np.float32)
            np.save(str(_cache_path(method_name, s["label"])), pred)
            results[s["label"]] = pred
            print("done")
    finally:
        shutdown()

    # Free GPU memory before loading next method
    _r1._cleanup_method_imports()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    return results


def _load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _load_gt_mask(path, target_hw=None):
    """Load and binarise a ground-truth mask (uint8 0 or 255)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.array(Image.open(path).convert("L"))
    mask = (mask > 127).astype(np.uint8) * 255
    if target_hw is not None:
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask


def _resize_pred(pred, target_hw):
    return cv2.resize(pred, (target_hw[1], target_hw[0]),
                      interpolation=cv2.INTER_LINEAR)


def generate_figure(all_preds, output_path, target_height=300):
    """
    Build and save the prediction grid.

    all_preds: dict[method_name -> dict[sample_label -> np.ndarray float32 H×W]]
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    methods = list(all_preds.keys())
    n_rows = len(SAMPLES)
    n_cols = 2 + len(methods)  # Original + GT + one per method

    col_labels = ["Original", "Ground Truth"] + methods

    # Precompute display images for each cell
    # rows × cols, each entry is an RGB uint8 array or a float heatmap
    cells = []      # list of rows; each row is list of (type, array)
    # type: "rgb" | "gt" | "heatmap"

    for s in SAMPLES:
        img = _load_rgb(s["image"])
        h, w = img.shape[:2]
        # Resize to target height, keep aspect ratio
        scale = target_height / h
        disp_h = target_height
        disp_w = max(1, int(w * scale))
        img_disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        gt = _load_gt_mask(s["mask"], target_hw=(disp_h, disp_w))
        gt_rgb = np.stack([gt, gt, gt], axis=-1)

        row = [("rgb", img_disp), ("gt", gt_rgb)]
        for method in methods:
            pred = all_preds[method].get(s["label"])
            if pred is None:
                pred = np.zeros((disp_h, disp_w), dtype=np.float32)
            pred_resized = _resize_pred(pred, (disp_h, disp_w))
            row.append(("heatmap", pred_resized))
        cells.append(row)

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Use a fixed cell width proportional to target_height
    cell_w = 1.6   # inches per column
    cell_h = 1.4   # inches per row (+ space for column labels)
    header_h = 0.5 # inches for column labels

    fig_w = n_cols * cell_w
    fig_h = n_rows * cell_h + header_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # GridSpec: one extra row for column labels
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        height_ratios=[header_h / cell_h] + [1.0] * n_rows,
        hspace=0.05, wspace=0.03,
        left=0.05, right=0.99, top=0.99, bottom=0.01,
    )

    # Column label row
    for col_idx, label in enumerate(col_labels):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.text(
            0.5, 0.5, label,
            ha="center", va="center",
            fontsize=8 if len(label) <= 10 else 7,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.axis("off")

    # Data rows
    for row_idx, (s, row_cells) in enumerate(zip(SAMPLES, cells)):
        for col_idx, (cell_type, arr) in enumerate(row_cells):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])

            if cell_type == "rgb":
                ax.imshow(arr)
                # Row label on the left of the original column
                ax.set_ylabel(
                    s["label"], fontsize=8, fontweight="bold",
                    rotation=0, labelpad=40, va="center",
                )
                ax.yaxis.set_label_coords(-0.35, 0.5)
            elif cell_type == "gt":
                ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
            else:  # heatmap
                ax.imshow(arr, cmap="hot", vmin=0.0, vmax=1.0)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    out = str(output_path)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if cache exists")
    parser.add_argument("--methods", nargs="*",
                        help="Subset of methods to run (default: all)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference; use existing cache to regenerate plot")
    args = parser.parse_args()

    methods = args.methods or METHODS_TO_RUN

    all_preds = {}
    if not args.plot_only:
        for method in methods:
            try:
                all_preds[method] = _run_method(method, force=args.force)
            except Exception as e:
                print(f"  ERROR loading {method}: {e}")
                all_preds[method] = {s["label"]: np.zeros((64, 64), dtype=np.float32)
                                     for s in SAMPLES}
    else:
        # Load from cache only
        for method in methods:
            all_preds[method] = {}
            for s in SAMPLES:
                cp = _cache_path(method, s["label"])
                if cp.exists():
                    all_preds[method][s["label"]] = np.load(str(cp))
                    print(f"  [cache] {method}/{s['label']}")
                else:
                    print(f"  [missing cache] {method}/{s['label']} — skipping")

    out_path = OUT_DIR / "e1_predictions.png"
    generate_figure(all_preds, out_path)


if __name__ == "__main__":
    main()
