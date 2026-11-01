#!/usr/bin/env python3
"""
Generate binary PNG masks for T-SROIE dataset from COCO-format JSON annotations.

T-SROIE has two annotation categories:
  1 = "text"      (original text regions, authentic)
  2 = "text_temp" (tampered text regions = FORGERY)

Masks are saved as binary PNG:
  255 = tampered (text_temp polygons)
    0 = authentic

Usage:
  python scripts/generate_tsroie_masks.py
  python scripts/generate_tsroie_masks.py --data-root data/T-SROIE --split test
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def generate_masks(data_root: Path, split: str = "test"):
    json_path = data_root / f"sroie_{split}_1011.json"
    img_dir = data_root / split
    mask_dir = data_root / split / "masks"
    mask_dir.mkdir(exist_ok=True)

    with open(json_path) as f:
        coco = json.load(f)

    # Build image_id -> image info mapping
    img_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id, keep only text_temp (category_id == 2)
    tampered_anns = {}
    for ann in coco["annotations"]:
        if ann["category_id"] == 2:
            img_id = ann["image_id"]
            tampered_anns.setdefault(img_id, []).append(ann)

    generated = 0
    skipped = 0

    for img_id, info in img_info.items():
        fname = info["file_name"]
        stem = Path(fname).stem
        mask_path = mask_dir / f"{stem}.png"

        h, w = info["height"], info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        anns = tampered_anns.get(img_id, [])
        for ann in anns:
            segs = ann.get("segmentation", [])
            for seg in segs:
                # seg is a flat list [x1, y1, x2, y2, ...]
                pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], color=255)

        cv2.imwrite(str(mask_path), mask)
        generated += 1

    print(f"Generated {generated} masks in {mask_dir}")
    print(f"Skipped: {skipped}")
    print(f"Images with tampered anns: {len(tampered_anns)}")
    print(f"Images without tampered anns (all-zero masks): {len(img_info) - len(tampered_anns)}")


def main():
    parser = argparse.ArgumentParser(description="Generate T-SROIE binary masks from COCO JSON")
    parser.add_argument("--data-root", default="data/T-SROIE",
                        help="Path to T-SROIE data root")
    parser.add_argument("--split", default="test",
                        help="Dataset split: test or train")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: data root not found: {data_root}")
        return

    generate_masks(data_root, args.split)


if __name__ == "__main__":
    main()
