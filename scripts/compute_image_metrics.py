#!/usr/bin/env python3
"""
compute_image_metrics.py — Compute image-level detection metrics from per-image CSVs.

Reads per-image CSVs produced by run_exp1.py / run_exp2.py / run_exp_fantasyd.py
and computes image-level detection metrics (image_AUC, image_F1, image_AP, image_accuracy)
using `max_pred_score` as the image-level detection score and `gt_tamper_ratio > 0`
as the tampered/authentic label.

Requires CSVs to have been generated with the updated scripts that include the
`max_pred_score` and `gt_tamper_ratio` columns. Older CSVs (without these columns)
are skipped with a warning.

Usage:
    cd document_forgery_benchmark   # repository root
    python scripts/compute_image_metrics.py --per-image-dir results/E2/per_image
    python scripts/compute_image_metrics.py --per-image-dir results/E1/per_image --output results/E1/image_metrics.json
    python scripts/compute_image_metrics.py --all   # process E1 and E2 together
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_EXP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import image_auc, image_f1, image_accuracy, image_ap


def load_csv(csv_path: Path) -> dict | None:
    """Load a per-image CSV. Returns None if required columns are missing."""
    import csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "max_pred_score" not in (reader.fieldnames or []):
            return None  # old-format CSV without image-level score columns
        for row in reader:
            rows.append(row)
    return rows


def compute_for_csv(rows: list[dict]) -> dict | None:
    """
    Compute image-level metrics from per-image rows.

    Label derivation: gt_tamper_ratio > 0 → tampered (1), else authentic (0).
    Score: max_pred_score (maximum pixel prediction over the image).

    Returns None if all images share a single label (AUC undefined).
    """
    scores = np.array([float(r["max_pred_score"]) for r in rows], dtype=np.float64)
    labels = np.array([1 if float(r["gt_tamper_ratio"]) > 0 else 0 for r in rows],
                      dtype=np.int32)

    n_tampered = labels.sum()
    n_authentic = (labels == 0).sum()

    if n_tampered == 0 or n_authentic == 0:
        # Single class — AUC and AP are undefined; only F1/accuracy make sense
        return {
            "n_images": len(labels),
            "n_tampered": int(n_tampered),
            "n_authentic": int(n_authentic),
            "image_auc": float("nan"),
            "image_ap": float("nan"),
            "image_f1": image_f1(scores, labels),
            "image_accuracy": image_accuracy(scores, labels),
            "note": "single-class dataset — AUC/AP undefined",
        }

    return {
        "n_images": len(labels),
        "n_tampered": int(n_tampered),
        "n_authentic": int(n_authentic),
        "image_auc": image_auc(scores, labels),
        "image_ap": image_ap(scores, labels),
        "image_f1": image_f1(scores, labels),
        "image_accuracy": image_accuracy(scores, labels),
    }


def process_per_image_dir(per_img_dir: Path) -> dict:
    """
    Process all CSVs in a per_image directory.

    Returns nested dict: {method: {dataset: metrics}}.
    """
    results = {}
    skipped_old_format = []

    for csv_path in sorted(per_img_dir.glob("*.csv")):
        stem = csv_path.stem  # e.g. "TruFor_DocTamper"

        # Parse method and dataset from filename (method name may contain hyphens/underscores)
        rows = load_csv(csv_path)
        if rows is None:
            skipped_old_format.append(csv_path.name)
            continue
        if not rows:
            continue

        # Split stem into method + dataset: last underscore-separated token that
        # matches a known dataset; everything before is the method name.
        known_datasets = [
            "DocTamper", "T-SROIE", "RealTextManipulation", "Tampered-IC13",
            "ReceiptForgery", "FantasyID", "MixTamper",
        ]
        method, dataset = None, None
        for ds in known_datasets:
            suffix = f"_{ds}"
            if stem.endswith(suffix):
                method = stem[: -len(suffix)]
                dataset = ds
                break
        if method is None:
            # Fall back: split on last underscore
            parts = stem.rsplit("_", 1)
            method, dataset = parts[0], parts[1] if len(parts) == 2 else ("unknown", stem)

        metrics = compute_for_csv(rows)
        if metrics is None:
            continue

        results.setdefault(method, {})[dataset] = metrics

    if skipped_old_format:
        print(f"  [WARN] Skipped {len(skipped_old_format)} old-format CSVs "
              f"(no max_pred_score column). Re-run experiments to generate updated CSVs.")

    return results


def print_table(results: dict) -> None:
    """Print a formatted summary table."""
    print(f"\n{'Method':<16} {'Dataset':<25} {'N':>6} {'Tam':>5} {'Auth':>5} "
          f"{'img_AUC':>8} {'img_AP':>7} {'img_F1':>7} {'img_Acc':>8}")
    print("-" * 95)
    for method in sorted(results):
        for dataset in sorted(results[method]):
            m = results[method][dataset]
            auc_s = f"{m['image_auc']:.3f}" if not np.isnan(m['image_auc']) else "  n/a "
            ap_s  = f"{m['image_ap']:.3f}"  if not np.isnan(m['image_ap'])  else "  n/a "
            print(f"{method:<16} {dataset:<25} {m['n_images']:>6} {m['n_tampered']:>5} "
                  f"{m['n_authentic']:>5} {auc_s:>8} {ap_s:>7} "
                  f"{m['image_f1']:>7.3f} {m['image_accuracy']:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compute image-level detection metrics")
    parser.add_argument("--per-image-dir", type=str,
                        help="Directory containing per-image CSVs (e.g. results/E2/per_image)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path. Defaults to <per_image_dir>/../image_metrics.json")
    parser.add_argument("--all", action="store_true",
                        help="Process both results/E1/per_image and results/E2/per_image, "
                             "merging into a single output per experiment.")
    args = parser.parse_args()

    if args.all:
        for exp in ["E1", "E2"]:
            per_img_dir = _EXP_ROOT / "results" / exp / "per_image"
            if not per_img_dir.exists():
                print(f"  Skipping {exp}: {per_img_dir} not found")
                continue
            print(f"\n=== {exp} ===")
            results = process_per_image_dir(per_img_dir)
            out = _EXP_ROOT / "results" / exp / "image_metrics.json"
            out.write_text(json.dumps(results, indent=2))
            print_table(results)
            print(f"\nSaved to {out}")
        return

    if not args.per_image_dir:
        parser.error("Provide --per-image-dir or --all")

    per_img_dir = Path(args.per_image_dir)
    if not per_img_dir.exists():
        print(f"ERROR: {per_img_dir} does not exist")
        sys.exit(1)

    print(f"Processing {per_img_dir} ...")
    results = process_per_image_dir(per_img_dir)

    out = Path(args.output) if args.output else per_img_dir.parent / "image_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
