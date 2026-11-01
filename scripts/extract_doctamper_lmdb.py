#!/usr/bin/env python3
"""Extract DocTamper LMDB datasets to PNG image/mask pairs.

Key scheme discovered:
  - image-XXXXXXXXX : JPEG-encoded RGB images (512x512)
  - label-XXXXXXXXX : PNG-encoded grayscale binary masks (512x512, values 0/1)
  - num-samples     : metadata with sample count

Extracts to:
  data/DocTamper/{split}/images/XXXXXXXXX.png
  data/DocTamper/{split}/masks/XXXXXXXXX.png
"""

import argparse
import io
import os
import time
from multiprocessing import Pool, cpu_count

import lmdb
from PIL import Image


def extract_batch(args):
    """Extract a batch of image/label pairs from LMDB."""
    lmdb_path, indices, images_dir, masks_dir = args
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True)
    count = 0
    with env.begin(buffers=True) as txn:
        for idx in indices:
            key_suffix = f"{idx:09d}"
            img_key = f"image-{key_suffix}".encode()
            lbl_key = f"label-{key_suffix}".encode()

            img_data = txn.get(img_key)
            lbl_data = txn.get(lbl_key)

            if img_data is None or lbl_data is None:
                continue

            # Decode and save image as PNG
            img = Image.open(io.BytesIO(bytes(img_data)))
            img.save(os.path.join(images_dir, f"{key_suffix}.png"), "PNG")

            # Label is already PNG; scale mask from {0,1} to {0,255}
            lbl = Image.open(io.BytesIO(bytes(lbl_data)))
            lbl_scaled = lbl.point(lambda x: x * 255)
            lbl_scaled.save(os.path.join(masks_dir, f"{key_suffix}.png"), "PNG")

            count += 1
    env.close()
    return count


def extract_lmdb(lmdb_path, output_base, split_name, num_workers=None):
    """Extract an entire LMDB dataset."""
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    images_dir = os.path.join(output_base, split_name, "images")
    masks_dir = os.path.join(output_base, split_name, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        num_samples = int(txn.get(b"num-samples").decode())
    env.close()

    print(f"Extracting {split_name}: {num_samples} samples from {lmdb_path}")
    print(f"  -> {images_dir}")
    print(f"  -> {masks_dir}")
    print(f"  Using {num_workers} workers")

    indices = list(range(num_samples))
    batch_size = (num_samples + num_workers - 1) // num_workers
    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        batches.append((lmdb_path, batch_indices, images_dir, masks_dir))

    t0 = time.time()
    with Pool(num_workers) as pool:
        results = pool.map(extract_batch, batches)

    total = sum(results)
    elapsed = time.time() - t0
    print(f"  Done: {total} pairs extracted in {elapsed:.1f}s ({total/elapsed:.0f} pairs/s)")
    return total


def main():
    parser = argparse.ArgumentParser(description="Extract DocTamper LMDB to PNG files")
    parser.add_argument("--lmdb-root", default="data/DocTamper_lmdb")
    parser.add_argument("--output-dir", default="data/DocTamper")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    datasets = [
        ("DocTamperV1-TestingSet", "test"),
        ("DocTamperV1-FCD", "fcd"),
        ("DocTamperV1-SCD", "scd"),
    ]

    total_all = 0
    t_start = time.time()
    for lmdb_name, split_name in datasets:
        lmdb_path = os.path.join(args.lmdb_root, lmdb_name)
        if not os.path.isdir(lmdb_path):
            print(f"WARNING: {lmdb_path} not found, skipping")
            continue
        total_all += extract_lmdb(lmdb_path, args.output_dir, split_name, args.workers)

    elapsed_total = time.time() - t_start
    print(f"\nAll done: {total_all} total pairs extracted in {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
