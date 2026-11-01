#!/usr/bin/env python3
"""
Convert bounding-box annotations to pixel-level binary mask PNG files
for Tampered-IC13 and Receipt Forgery datasets.

Usage:
    python scripts/generate_bbox_masks.py --dataset tampered_ic13
    python scripts/generate_bbox_masks.py --dataset receipt_forgery
"""
import argparse
import csv
import json
import os
import ast
from pathlib import Path

import cv2
import numpy as np


def generate_tampered_ic13_masks(data_root: Path):
    """
    Tampered-IC13 test_gt format: x1,y1,x2,y2,label  (label=1 → tampered)
    Images in test_img/, annotations in test_gt/.
    Output: test_masks/ (one PNG per image, 255 = tampered).
    """
    img_dir = data_root / "Tampered-IC13" / "test_img"
    gt_dir = data_root / "Tampered-IC13" / "test_gt"
    mask_dir = data_root / "Tampered-IC13" / "test_masks"
    mask_dir.mkdir(exist_ok=True)

    img_paths = sorted(img_dir.glob("*.jpg"))
    print(f"[Tampered-IC13] {len(img_paths)} images → masks in {mask_dir}")

    for img_path in img_paths:
        gt_path = gt_dir / (img_path.stem + ".txt")
        if not gt_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path}")
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

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
                if label == 1:  # tampered region
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    mask[y1:y2, x1:x2] = 255

        out_path = mask_dir / (img_path.stem + ".png")
        cv2.imwrite(str(out_path), mask)

    print(f"  Done. Generated masks in {mask_dir}")


def generate_receipt_forgery_masks(data_root: Path):
    """
    Receipt Forgery format: CSV manifest with 'forgery annotations' column containing
    VGG-style JSON dicts with 'regions' -> 'shape_attributes' {name: rect, x, y, width, height}.
    Images in findit2/test/, manifest in findit2/test.txt.
    Output: findit2/test_masks/ (one PNG per forged image, 0 = authentic or empty for unforgered).
    """
    base = data_root / "ReceiptForgery" / "findit2"
    manifest = base / "test.txt"
    img_dir = base / "test"
    mask_dir = base / "test_masks"
    mask_dir.mkdir(exist_ok=True)

    print(f"[ReceiptForgery] Reading manifest {manifest}")
    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"  {len(rows)} entries; generating masks in {mask_dir}")
    n_forged = 0

    for row in rows:
        img_name = row["image"]
        forged = row["forged"].strip() == "1"
        img_path = img_dir / img_name

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if forged:
            annot_str = row.get("forgery annotations", "").strip()
            if annot_str:
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
                            x1, y1 = max(0, rx), max(0, ry)
                            x2, y2 = min(w, rx + rw), min(h, ry + rh)
                            mask[y1:y2, x1:x2] = 255
                except Exception as e:
                    print(f"  [WARN] Failed to parse annotations for {img_name}: {e}")
            n_forged += 1

        out_name = Path(img_name).stem + ".png"
        cv2.imwrite(str(mask_dir / out_name), mask)

    print(f"  Done. {n_forged} forged images with masks written to {mask_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["tampered_ic13", "receipt_forgery", "all"],
        default="all",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Path to data/ directory",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.dataset in ("tampered_ic13", "all"):
        generate_tampered_ic13_masks(data_root)

    if args.dataset in ("receipt_forgery", "all"):
        generate_receipt_forgery_masks(data_root)


if __name__ == "__main__":
    main()
