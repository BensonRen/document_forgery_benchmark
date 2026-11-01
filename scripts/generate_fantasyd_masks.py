#!/usr/bin/env python3
"""Generate pixel-level masks for FantasyID text-only attacks."""
import json, os, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd

DATA = Path(os.environ.get("DOCFORGE_DATA", "data"))
BASE = DATA / "FantasyID" / "FantasyID"
OUT_IMAGES = DATA / "FantasyID" / "text_only" / "images"
OUT_MASKS  = DATA / "FantasyID" / "text_only" / "masks"
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_MASKS.mkdir(parents=True, exist_ok=True)

def get_prefix(rel_path):
    parts = Path(rel_path).parts
    # parts: ('test', 'attack', 'digital_3', 'huawei', 'foo.jpg') or ('test', 'bonafide', 'huawei', 'foo.jpg')
    if parts[1] == 'attack':
        attack_dir = parts[2]
        capture = parts[3]
        if attack_dir == 'digital_3':
            return f"d3_{capture}"
        elif 'textdiffuserft' in attack_dir:
            return f"tdft_{capture}"
        else:
            return f"{attack_dir}_{capture}"
    else:  # bonafide
        capture = parts[2]
        return f"bonafide_{capture}"

def get_image_dims(img_path, json_data):
    """Get image dimensions — use actual image size."""
    with Image.open(img_path) as img:
        return img.size  # (width, height)

def make_mask(img_path, json_path, out_mask_path):
    with open(json_path) as f:
        data = json.load(f)
    w, h = get_image_dims(img_path, data)
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in data.get('regions', []):
        ra = region.get('region_attributes', {})
        if ra.get('region_provenance') == 'altered':
            sa = region['shape_attributes']
            x, y, rw, rh = sa['x'], sa['y'], sa['width'], sa['height']
            x2 = min(x + rw, w)
            y2 = min(y + rh, h)
            mask[y:y2, x:x2] = 255
    Image.fromarray(mask, mode='L').save(out_mask_path)

# Load CSV — try both paths
df = None
for test_csv in [BASE / "test.csv", BASE.parent / "test.csv"]:
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        break
if df is None:
    raise FileNotFoundError("test.csv not found")

# Filter: only text attacks and bonafide (exclude face/facedancer)
df = df[df['attack_type'].isin(['text', 'none'])]
print(f"Processing {len(df)} images (text attacks + bonafide, excluding face)")

ok, err = 0, 0
for _, row in df.iterrows():
    rel_path = row['path']
    prefix = get_prefix(rel_path)
    img_path = BASE / rel_path if not rel_path.startswith('/') else Path(rel_path)
    # Try both
    if not img_path.exists():
        img_path = BASE.parent / rel_path
    if not img_path.exists():
        print(f"  [WARN] Image not found: {rel_path}")
        err += 1
        continue

    json_path = img_path.with_suffix('.json')
    if not json_path.exists():
        print(f"  [WARN] JSON not found: {json_path}")
        err += 1
        continue

    stem = img_path.stem
    ext = img_path.suffix
    out_name = f"{prefix}_{stem}"
    out_img_link = OUT_IMAGES / f"{out_name}{ext}"
    out_mask_path = OUT_MASKS / f"{out_name}.png"

    # Symlink image
    if not out_img_link.exists():
        os.symlink(img_path.resolve(), out_img_link)

    # Generate mask
    if not out_mask_path.exists():
        try:
            make_mask(img_path, json_path, out_mask_path)
            ok += 1
        except Exception as e:
            print(f"  [ERR] {img_path.name}: {e}")
            err += 1
    else:
        ok += 1

print(f"Done: {ok} ok, {err} errors")
print(f"Images: {len(list(OUT_IMAGES.iterdir()))}")
print(f"Masks:  {len(list(OUT_MASKS.iterdir()))}")
