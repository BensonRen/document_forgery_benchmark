#!/usr/bin/env python3
"""
Calibration Experiment for DocForge-Bench.

Shows that a small domain calibration set can recover much of the
Oracle-F1 gap caused by score distribution shift.

For each (method, dataset) pair, the script:
  1. Runs inference and saves per-image score histograms (2KB/image).
  2. Simulates calibration curves for N ∈ {10, 25, 50, 100, 200}.

Histogram-based score representation
--------------------------------------
For each image we store two 256-bin histograms:
  hist_pos[b] = pixel count with GT=1 and score in bin b
  hist_neg[b] = pixel count with GT=0 and score in bin b
From these, for threshold t (bin b_t = floor(t * 256)):
  TP = sum(hist_pos[b_t:])
  FP = sum(hist_neg[b_t:])
  FN = sum(hist_pos[:b_t])
  F1 = 2*TP / (2*TP + FP + FN)  if TP > 0 else 0.0

Usage:
    python scripts/run_calibration_exp.py \
        --config configs/exp1_general_on_docs.yaml \
        --method TruFor \
        --dataset DocTamper \
        --calibration-sizes 10 25 50 100 200 \
        --output-dir results/calibration
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Insert scripts dir so we can import run_exp1 and run_exp2
sys.path.insert(0, str(Path(__file__).parent))
from run_exp1 import (
    load_dataset,
    METHOD_LOADERS,
    _resolve_repo,
    _resolve_weights,
    _cleanup_method_imports,
    _make_subprocess_worker_loader,
)
# Import run_exp2 to register document-specific method loaders into METHOD_LOADERS
# (run_exp2 extends the shared METHOD_LOADERS dict in-place)
import run_exp2  # noqa: F401


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

NUM_BINS = 256


def _scores_to_hists(pred: np.ndarray, gt: np.ndarray):
    """
    Given prediction map (float32, 0-1, H x W) and binary GT mask (float32,
    0-1, H x W), return a (2, 256) array where row 0 is hist_pos and row 1
    is hist_neg.
    """
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()

    pos_mask = gt_flat >= 0.5
    neg_mask = ~pos_mask

    bins = np.arange(NUM_BINS + 1) / NUM_BINS  # 257 edges → 256 bins

    hist_pos, _ = np.histogram(pred_flat[pos_mask], bins=bins)
    hist_neg, _ = np.histogram(pred_flat[neg_mask], bins=bins)

    return np.stack([hist_pos, hist_neg], axis=0).astype(np.int64)


def _f1_from_hists(hist_pos: np.ndarray, hist_neg: np.ndarray, threshold: float) -> float:
    """
    Compute F1 at a given threshold from aggregated pos/neg histogram arrays.
    Both arrays must have shape (256,).
    threshold is in [0, 1]; converted to bin index b_t = floor(t * 256).
    """
    b_t = int(np.floor(threshold * NUM_BINS))
    b_t = max(0, min(b_t, NUM_BINS))  # clamp

    tp = int(hist_pos[b_t:].sum())
    fp = int(hist_neg[b_t:].sum())
    fn = int(hist_pos[:b_t].sum())

    denom = 2 * tp + fp + fn
    if denom == 0 or tp == 0:
        return 0.0
    return 2.0 * tp / denom


def _find_best_threshold(hist_pos: np.ndarray, hist_neg: np.ndarray) -> tuple:
    """
    Scan 199 thresholds in (0, 1) and return (best_threshold, best_f1).
    """
    thresholds = np.linspace(0, 1, 201)[1:-1]  # 199 values, exclude 0 and 1
    best_t = 0.5
    best_f1 = 0.0
    for t in thresholds:
        f1 = _f1_from_hists(hist_pos, hist_neg, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Inference + histogram collection
# ---------------------------------------------------------------------------

def collect_histograms(predict, pairs, hist_file: Path, method_name: str, dataset_name: str):
    """
    Run inference on all (img, mask) pairs and save per-image histograms to
    an npz file.  Each key is the string index 'i'; value is shape (2, 256).
    Returns the in-memory dict {i: array}.
    """
    print(f"\n  [INFERENCE] {method_name} on {dataset_name}: {len(pairs)} images")
    hists = {}

    for i, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"  {method_name}/{dataset_name}")):
        try:
            pred = predict(img_path)
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                print(f"  [WARN] Could not read mask {mask_path} — skipping")
                continue
            gt = gt.astype(np.float32) / 255.0

            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            h = _scores_to_hists(pred, gt)
            hists[str(i)] = h

        except Exception as e:
            print(f"  [WARN] Inference failed for {img_path}: {e} — skipping")
            continue

        # Release memory immediately
        del pred, gt

    # Save to disk
    hist_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(hist_file), **hists)
    print(f"  [SAVED] Histograms → {hist_file} ({len(hists)} images)")
    return hists


def load_histograms(hist_file: Path):
    """Load histograms from npz file; return dict {str(i): array(2,256)}."""
    data = np.load(str(hist_file))
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Calibration analysis
# ---------------------------------------------------------------------------

def run_calibration_analysis(hists: dict, calibration_sizes: list, pairs: list):
    """
    Given the per-image histogram dict, compute:
      - f1_at_05: F1 at threshold 0.5, averaged over all images
      - oracle_f1: per-image best-F1 averaged over all images
      - calibration_curve: list of {N, cal_threshold, test_f1} for each N

    Images are ordered by their integer key (matching the original pairs list).
    """
    # Sort by integer index to restore original order
    indices = sorted(hists.keys(), key=lambda k: int(k))
    n_total = len(indices)

    if n_total == 0:
        return {
            "f1_at_05": 0.0,
            "oracle_f1": 0.0,
            "calibration_curve": [],
        }

    # Build ordered list of (hist_pos, hist_neg) arrays
    ordered_hists = []
    for k in indices:
        arr = hists[k]
        ordered_hists.append((arr[0], arr[1]))  # (pos, neg)

    # --- F1 @ 0.5 on full set ---
    f1_05_scores = []
    for hp, hn in ordered_hists:
        f1 = _f1_from_hists(hp, hn, 0.5)
        f1_05_scores.append(f1)
    f1_at_05 = float(np.mean(f1_05_scores))

    # --- Oracle F1 on full set ---
    oracle_f1_scores = []
    for hp, hn in ordered_hists:
        _, best_f1 = _find_best_threshold(hp, hn)
        oracle_f1_scores.append(best_f1)
    oracle_f1 = float(np.mean(oracle_f1_scores))

    print(f"  F1@0.5={f1_at_05:.4f}  Oracle-F1={oracle_f1:.4f}  (n={n_total})")

    # --- Calibration curves ---
    calibration_curve = []

    for N in calibration_sizes:
        if N >= n_total:
            print(f"  [SKIP] N={N} >= total images ({n_total}) — skipping")
            continue

        # Calibration set: first N images
        cal_hists = ordered_hists[:N]
        # Test set: remaining images
        test_hists = ordered_hists[N:]

        if len(test_hists) == 0:
            print(f"  [SKIP] N={N} leaves 0 test images — skipping")
            continue

        # Aggregate calibration histograms
        agg_pos = np.zeros(NUM_BINS, dtype=np.int64)
        agg_neg = np.zeros(NUM_BINS, dtype=np.int64)
        for hp, hn in cal_hists:
            agg_pos += hp
            agg_neg += hn

        # Find threshold on calibration set
        cal_threshold, cal_f1 = _find_best_threshold(agg_pos, agg_neg)

        # Apply cal_threshold to test set
        test_f1_scores = []
        for hp, hn in test_hists:
            f1 = _f1_from_hists(hp, hn, cal_threshold)
            test_f1_scores.append(f1)
        test_f1 = float(np.mean(test_f1_scores))

        print(
            f"  N={N:4d}  cal_threshold={cal_threshold:.4f}  "
            f"cal_F1={cal_f1:.4f}  test_F1={test_f1:.4f}  "
            f"(cal={N}, test={len(test_hists)})"
        )

        calibration_curve.append(
            {
                "N": N,
                "cal_threshold": round(cal_threshold, 6),
                "test_f1": round(test_f1, 6),
            }
        )

    return {
        "f1_at_05": round(f1_at_05, 6),
        "oracle_f1": round(oracle_f1, 6),
        "calibration_curve": calibration_curve,
    }


# ---------------------------------------------------------------------------
# Main per-(method, dataset) runner
# ---------------------------------------------------------------------------

def run_pair(
    method_cfg: dict,
    ds_cfg: dict,
    calibration_sizes: list,
    output_dir: Path,
    results: dict,
):
    method_name = method_cfg["name"]
    ds_name = ds_cfg["name"]
    key = f"{method_name}_{ds_name}"

    print(f"\n{'='*60}")
    print(f"  {method_name}  ×  {ds_name}")
    print(f"{'='*60}")

    hist_file = output_dir / f"{method_name}_{ds_name}_hists.npz"

    # Load or compute histograms
    if hist_file.exists():
        print(f"  [RESUME] Loading existing histograms from {hist_file}")
        hists = load_histograms(hist_file)
        predict = None
        shutdown = None
    else:
        # Load method
        loader = METHOD_LOADERS.get(method_name)
        if loader is None:
            print(f"  [SKIP] No loader for {method_name}")
            return

        _cleanup_method_imports()

        try:
            result = loader(method_cfg)
            if isinstance(result, tuple):
                predict, shutdown = result
            else:
                predict, shutdown = result, None
        except Exception as e:
            print(f"  [ERROR] Failed to load {method_name}: {e}")
            return

        pairs = load_dataset(ds_cfg)
        if not pairs:
            print(f"  [SKIP] No data found for {ds_name}")
            if shutdown is not None:
                shutdown()
            return

        hists = collect_histograms(predict, pairs, hist_file, method_name, ds_name)

        # Shutdown subprocess workers
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass

        # Clean up model from memory
        del predict
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # Load dataset for pairing metadata (pairs list length)
    pairs = load_dataset(ds_cfg)
    if not pairs:
        print(f"  [SKIP] No data found for {ds_name}")
        return

    # Run calibration analysis
    analysis = run_calibration_analysis(hists, calibration_sizes, pairs)

    entry = {
        "method": method_name,
        "dataset": ds_name,
        "f1_at_05": analysis["f1_at_05"],
        "oracle_f1": analysis["oracle_f1"],
        "calibration_curve": analysis["calibration_curve"],
    }

    results[key] = entry
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DocForge-Bench calibration experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--method",
        default="all",
        help='Method name to run (e.g. "TruFor"), or "all" to run every method in config',
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help='Dataset name to run (e.g. "DocTamper"), or "all" to run every dataset in config',
    )
    parser.add_argument(
        "--calibration-sizes",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100, 200],
        metavar="N",
        help="List of calibration set sizes to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="results/calibration",
        help="Directory to save histograms and results.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"

    # Load existing results for resumption
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"[INFO] Loaded existing results: {list(results.keys())}")
    else:
        results = {}

    calibration_sizes = sorted(args.calibration_sizes)

    methods_cfg = config.get("methods", [])
    datasets_cfg = config.get("datasets", [])

    # Apply filters
    if args.method != "all":
        methods_cfg = [m for m in methods_cfg if m["name"] == args.method]
        if not methods_cfg:
            print(f"[ERROR] Method '{args.method}' not found in config. "
                  f"Available: {[m['name'] for m in config.get('methods', [])]}")
            sys.exit(1)

    if args.dataset != "all":
        datasets_cfg = [d for d in datasets_cfg if d["name"] == args.dataset]
        if not datasets_cfg:
            print(f"[ERROR] Dataset '{args.dataset}' not found in config. "
                  f"Available: {[d['name'] for d in config.get('datasets', [])]}")
            sys.exit(1)

    print(f"[INFO] Methods:  {[m['name'] for m in methods_cfg]}")
    print(f"[INFO] Datasets: {[d['name'] for d in datasets_cfg]}")
    print(f"[INFO] Calibration sizes: {calibration_sizes}")
    print(f"[INFO] Output dir: {output_dir}")

    for method_cfg in methods_cfg:
        method_name = method_cfg["name"]

        # Check loader exists before loading model
        if method_name not in METHOD_LOADERS:
            print(f"\n[SKIP] No loader registered for method: {method_name}")
            continue

        for ds_cfg in datasets_cfg:
            ds_name = ds_cfg["name"]
            key = f"{method_name}_{ds_name}"

            # Check if calibration analysis already done for this pair
            # (histograms may exist but analysis may need to re-run with different N)
            hist_file = output_dir / f"{method_name}_{ds_name}_hists.npz"
            already_done = (
                key in results
                and set(results[key].get("_calibration_sizes_used", [])) >= set(calibration_sizes)
            )
            if already_done:
                print(f"\n[SKIP] {key} already computed with required N values — skipping")
                continue

            entry = run_pair(
                method_cfg=method_cfg,
                ds_cfg=ds_cfg,
                calibration_sizes=calibration_sizes,
                output_dir=output_dir,
                results=results,
            )

            if entry is not None:
                # Record which N values were used so resumption works correctly
                results[key]["_calibration_sizes_used"] = calibration_sizes

            # Save results after each (method, dataset) pair
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [SAVED] results.json updated at {results_path}")

    # Final save (strip internal metadata key from output)
    final_results = {}
    for k, v in results.items():
        entry = dict(v)
        entry.pop("_calibration_sizes_used", None)
        final_results[k] = entry

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE. Results saved to: {results_path}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Method':<16} {'Dataset':<20} {'F1@0.5':>8} {'Oracle':>8}", end="")
    for N in calibration_sizes:
        print(f"  {'N='+str(N):>8}", end="")
    print()
    print("-" * (16 + 20 + 18 + 10 * len(calibration_sizes)))

    for key, entry in final_results.items():
        curve_by_n = {c["N"]: c["test_f1"] for c in entry.get("calibration_curve", [])}
        print(
            f"{entry['method']:<16} {entry['dataset']:<20} "
            f"{entry['f1_at_05']:>8.4f} {entry['oracle_f1']:>8.4f}",
            end="",
        )
        for N in calibration_sizes:
            val = curve_by_n.get(N, float("nan"))
            if np.isnan(val):
                print(f"  {'—':>8}", end="")
            else:
                print(f"  {val:>8.4f}", end="")
        print()


if __name__ == "__main__":
    main()
