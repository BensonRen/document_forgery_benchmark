#!/usr/bin/env python3
"""
calibration_full.py
====================
Full 112-pair (14 methods x 8 datasets) calibration experiment for the
document forgery detection benchmark paper (paper, Discussion).

For each (method, dataset) pair this script:
  1. Loads per-image predictions (.npy float32, H x W in [0,1]) and binary
     ground-truth masks (.npy uint8/bool, H x W in {0,1}).
  2. Computes three reference F1 values:
       - fixed_f1    : F1 at the default threshold tau = 0.5
       - oracle_f1   : per-image best-F1 over 100 thresholds (unreachable ceiling)
       - global_opt_f1 : best mean-F1 over all images achievable with a single
                         threshold (the true calibration ceiling)
  3. For N in [10, 25, 50, 100, 200]:
       - Runs 20 random train/test splits
       - Calibration set : N images drawn without replacement
       - Test set        : all remaining images
       - For each split  : sweep 100 thresholds, pick tau* maximising mean F1
                           on the N-image calibration set, then apply to test set
       - cal_f1_N        : mean calibrated-test F1 averaged over 20 splits
       - recovery_pct_N  : (cal_f1_N - fixed_f1) / (global_opt_f1 - fixed_f1) * 100
                           (how much of the achievable calibration gap is recovered)

Outputs
-------
  output/calibration_full_results.csv
  output/fig_calibration_heatmap_full.pdf   (14x8 recovery_pct_N=10 heatmap)

Data layout expected on disk
-----------------------------
  {results_dir}/{method_name}/{dataset_name}/predictions/{image_id}.npy  <- float prediction mask
  {results_dir}/{method_name}/{dataset_name}/gt/{image_id}.npy           <- binary GT mask

Usage
-----
  # With real data:
  python calibration_full.py --results_dir results/

  # Smoke-test with synthetic data (no real data needed):
  python calibration_full.py --dry_run
"""

import os
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = [
    "trufor", "mantranet", "mvss_net", "cat_net", "pscc_net",
    "iml_vit", "safire", "doctamper", "dtd", "ffdn",
    "caftb_net", "tifdm", "ascformer", "adcd_net",
]

DATASETS = [
    "doctamper", "tsroie", "realtextmanip", "tampered_ic13",
    "receipt_forgery", "mixtamper", "fantasyid", "fsts_1.5k",
]

# Methods considered "document-specific" (trained on document data).
# The rest are "general" natural-image forensics methods.
DOC_SPECIFIC_METHODS = {
    "doctamper", "dtd", "ffdn", "caftb_net", "tifdm", "ascformer", "adcd_net",
}

CAL_SIZES = [10, 25, 50, 100, 200]   # N images for calibration
N_SPLITS   = 20                       # random train/test splits per N
N_THRESHOLDS_SWEEP = 100              # threshold candidates in [0, 1]
N_THRESHOLDS_GLOBAL = 200             # finer sweep for global_opt_f1


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_f1_at_threshold(pred_flat: np.ndarray, gt_flat: np.ndarray,
                             threshold: float) -> float:
    """Binary F1 for a single flattened prediction/GT pair at a given threshold."""
    pred_bin = (pred_flat >= threshold).astype(np.int32)
    return float(f1_score(gt_flat, pred_bin, zero_division=0))


def is_authentic(gt: np.ndarray) -> bool:
    """Return True if the image has no tampered pixels (authentic image)."""
    return gt.flatten().sum() == 0


def per_image_oracle_f1(pred: np.ndarray, gt: np.ndarray,
                         n_thresholds: int = N_THRESHOLDS_SWEEP) -> float:
    """Best F1 over a threshold grid for a single image. NaN for authentic images."""
    gt_flat = gt.flatten().astype(np.int32)
    if gt_flat.sum() == 0:
        return float("nan")
    pred_flat = pred.flatten().astype(np.float32)
    thresholds = np.linspace(pred_flat.min(), pred_flat.max(), n_thresholds)
    best = 0.0
    for t in thresholds:
        f = compute_f1_at_threshold(pred_flat, gt_flat, t)
        if f > best:
            best = f
    return best


def per_image_fixed_f1(pred: np.ndarray, gt: np.ndarray,
                        threshold: float = 0.5) -> float:
    """F1 at a fixed threshold. NaN for authentic images."""
    gt_flat = gt.flatten().astype(np.int32)
    if gt_flat.sum() == 0:
        return float("nan")
    return compute_f1_at_threshold(pred.flatten().astype(np.float32), gt_flat,
                                   threshold)


def nanmean(values) -> float:
    """Mean ignoring NaN. Returns NaN if all values are NaN."""
    arr = np.array(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Global-Opt-F1
# ---------------------------------------------------------------------------

def global_opt_f1_for_pair(preds_list, gts_list,
                             n_thresholds: int = N_THRESHOLDS_GLOBAL):
    """
    Sweep a single global threshold over all tampered images and return the
    threshold that maximises mean F1.  Authentic images (GT all-zero) are
    excluded from F1 computation.

    Returns (global_opt_f1, tau_star).
    """
    # Filter out authentic images
    tampered_preds = []
    tampered_gts   = []
    for pred, gt in zip(preds_list, gts_list):
        if not is_authentic(gt):
            tampered_preds.append(pred.flatten().astype(np.float32))
            tampered_gts.append(gt.flatten().astype(np.int32))

    if not tampered_preds:
        return float("nan"), float("nan")

    best_f1  = 0.0
    best_tau = 0.5
    for t in np.linspace(0.0, 1.0, n_thresholds):
        f1s = [
            compute_f1_at_threshold(p, g, t)
            for p, g in zip(tampered_preds, tampered_gts)
        ]
        mean_f1 = float(np.mean(f1s))
        if mean_f1 > best_f1:
            best_f1  = mean_f1
            best_tau = float(t)

    return best_f1, best_tau


# ---------------------------------------------------------------------------
# Calibration experiment
# ---------------------------------------------------------------------------

def calibration_experiment(preds_list, gts_list,
                             cal_size: int,
                             n_splits: int = N_SPLITS,
                             n_thresholds: int = N_THRESHOLDS_SWEEP,
                             seed: int = 42) -> float:
    """
    Simulate calibrating a single global threshold on `cal_size` randomly
    chosen tampered images, then evaluating on the held-out tampered images.

    Returns mean calibrated-test F1 averaged over `n_splits` random splits.
    Authentic images are excluded from both calibration and test sets.
    """
    # Only tampered images participate
    tampered_idx = [i for i, g in enumerate(gts_list) if not is_authentic(g)]

    if len(tampered_idx) <= cal_size:
        # Not enough images to hold out a test set; return NaN
        return float("nan")

    rng = random.Random(seed)
    test_f1s = []

    for split_idx in range(n_splits):
        # Draw calibration set
        cal_indices = rng.sample(tampered_idx, cal_size)
        tst_indices = [i for i in tampered_idx if i not in set(cal_indices)]

        # --- Find tau* on calibration set ---
        cal_preds = [preds_list[i].flatten().astype(np.float32) for i in cal_indices]
        cal_gts   = [gts_list[i].flatten().astype(np.int32)   for i in cal_indices]

        best_cal_f1  = -1.0
        best_tau     = 0.5
        for t in np.linspace(0.0, 1.0, n_thresholds):
            f1s = [compute_f1_at_threshold(p, g, t)
                   for p, g in zip(cal_preds, cal_gts)]
            mean_f1 = float(np.mean(f1s))
            if mean_f1 > best_cal_f1:
                best_cal_f1 = mean_f1
                best_tau    = float(t)

        # --- Apply tau* to test set ---
        tst_f1s = [
            compute_f1_at_threshold(
                preds_list[i].flatten().astype(np.float32),
                gts_list[i].flatten().astype(np.int32),
                best_tau,
            )
            for i in tst_indices
        ]
        test_f1s.append(float(np.mean(tst_f1s)))

    return float(np.mean(test_f1s))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions_and_gts(results_dir: str, method: str, dataset: str):
    """
    Load all predictions and ground-truth masks for a (method, dataset) pair.

    Expected layout:
        {results_dir}/{method}/{dataset}/predictions/{image_id}.npy   <- float [0,1]
        {results_dir}/{method}/{dataset}/gt/{image_id}.npy            <- binary {0,1}

    Returns (preds_list, gts_list) where each element is a (H, W) numpy array.
    Raises FileNotFoundError if the directory does not exist.
    """
    pred_dir = Path(results_dir) / method / dataset / "predictions"
    gt_dir   = Path(results_dir) / method / dataset / "gt"

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    pred_files = sorted(pred_dir.glob("*.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No .npy files found in {pred_dir}")

    preds_list, gts_list = [], []
    for pf in pred_files:
        gf = gt_dir / pf.name
        if not gf.exists():
            warnings.warn(f"GT file missing for {pf.name} — skipping")
            continue
        pred = np.load(str(pf)).astype(np.float32)
        gt   = np.load(str(gf)).astype(np.int32)
        # Clip predictions to [0, 1] in case of floating-point drift
        pred = np.clip(pred, 0.0, 1.0)
        preds_list.append(pred)
        gts_list.append(gt)

    return preds_list, gts_list


# ---------------------------------------------------------------------------
# Synthetic data for --dry_run
# ---------------------------------------------------------------------------

def make_synthetic_data(n_images: int = 60,
                         h: int = 64, w: int = 64,
                         authentic_frac: float = 0.2,
                         rng_seed: int = 0):
    """
    Generate synthetic (prediction, gt) pairs for smoke-testing.

    Tampered images: GT has a small random rectangular region set to 1.
    Predictions are noisy versions of GT with a score shift below 0.5 to
    simulate the common score-distribution-shift failure mode discussed in §8.1.
    Authentic images: GT all-zero, predictions drawn from low-score noise.
    """
    rng = np.random.default_rng(rng_seed)
    n_authentic = max(1, int(n_images * authentic_frac))
    n_tampered  = n_images - n_authentic

    preds_list, gts_list = [], []

    # --- Tampered images ---
    for _ in range(n_tampered):
        gt = np.zeros((h, w), dtype=np.int32)
        # Random small forged rectangle (5–20% of image area each dimension)
        r0 = rng.integers(0, h // 2)
        r1 = r0 + rng.integers(max(1, h // 20), max(2, h // 5))
        c0 = rng.integers(0, w // 2)
        c1 = c0 + rng.integers(max(1, w // 20), max(2, w // 5))
        gt[r0:r1, c0:c1] = 1

        # Score map: low background noise + elevated (but shifted) forged signal
        pred = rng.uniform(0.0, 0.15, (h, w)).astype(np.float32)
        # Forged region: signal in [0.10, 0.40] — shifted below 0.5
        pred[r0:r1, c0:c1] = rng.uniform(0.10, 0.40,
                                           (r1 - r0, c1 - c0)).astype(np.float32)
        preds_list.append(pred)
        gts_list.append(gt)

    # --- Authentic images ---
    for _ in range(n_authentic):
        gt   = np.zeros((h, w), dtype=np.int32)
        pred = rng.uniform(0.0, 0.10, (h, w)).astype(np.float32)
        preds_list.append(pred)
        gts_list.append(gt)

    return preds_list, gts_list


# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------

def plot_recovery_heatmap(df: pd.DataFrame, metric_col: str,
                           title: str, output_path: str):
    """
    Plot a 14 (methods) x 8 (datasets) heatmap of `metric_col`.
    Rows = methods, columns = datasets.
    """
    pivot = df.pivot(index="method", columns="dataset", values=metric_col)
    # Ensure consistent ordering
    pivot = pivot.reindex(index=METHODS, columns=DATASETS)

    fig, ax = plt.subplots(figsize=(12, 7))
    data = pivot.values.astype(np.float64)

    # Replace NaN with -1 for display; mark them distinctly
    nan_mask = np.isnan(data)
    data_display = np.where(nan_mask, -1.0, data)

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_under("lightgray")   # NaN cells
    im = ax.imshow(data_display, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Axis labels
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    method_labels = [
        f"{m} {'(doc)' if m in DOC_SPECIFIC_METHODS else '(gen)'}"
        for m in METHODS
    ]
    ax.set_yticklabels(method_labels, fontsize=8)

    # Annotate cells
    for r in range(len(METHODS)):
        for c in range(len(DATASETS)):
            val = pivot.values[r, c]
            if np.isnan(val):
                txt = "N/A"
                color = "gray"
            else:
                txt   = f"{val:.0f}"
                color = "black" if 20 < val < 80 else "white"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=6.5, color=color)

    plt.colorbar(im, ax=ax, label="Recovery % (N=10 calibration images)")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Dataset", fontsize=9)
    ax.set_ylabel("Method", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved heatmap → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Full 112-pair calibration experiment (paper, Discussion)."
    )
    p.add_argument(
        "--results_dir", default="results/",
        help="Root directory containing per-method/per-dataset predictions and GT.",
    )
    p.add_argument(
        "--output_dir", default="paper/scripts/output/",
        help="Directory for CSV and PDF outputs.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Generate synthetic data and run the full pipeline without loading files.",
    )
    p.add_argument(
        "--n_splits", type=int, default=N_SPLITS,
        help=f"Number of random train/test splits per calibration size (default {N_SPLITS}).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path     = os.path.join(args.output_dir, "calibration_full_results.csv")
    fig_path     = os.path.join(args.output_dir, "fig_calibration_heatmap_full.pdf")

    rows = []

    for m_idx, method in enumerate(METHODS):
        for d_idx, dataset in enumerate(DATASETS):
            pair_label = f"{method}/{dataset}"
            print(f"[{m_idx * len(DATASETS) + d_idx + 1:3d}/112] {pair_label}", end=" ... ", flush=True)

            # ---- Load (or synthesise) data ----
            if args.dry_run:
                # Vary synthetic seed per pair so each pair has different characteristics
                pair_seed = hash((method, dataset)) % (2 ** 31)
                preds_list, gts_list = make_synthetic_data(
                    n_images=80, h=64, w=64, rng_seed=pair_seed
                )
            else:
                try:
                    preds_list, gts_list = load_predictions_and_gts(
                        args.results_dir, method, dataset
                    )
                except FileNotFoundError as e:
                    print(f"SKIP ({e})")
                    continue

            if len(preds_list) == 0:
                print("SKIP (no images)")
                continue

            # ---- Reference metrics ----
            # Per-image fixed F1 @ tau = 0.5
            fixed_f1_vals   = [per_image_fixed_f1(p, g, 0.5)
                                for p, g in zip(preds_list, gts_list)]
            fixed_f1_mean   = nanmean(fixed_f1_vals)

            # Per-image Oracle-F1
            oracle_f1_vals  = [per_image_oracle_f1(p, g)
                                for p, g in zip(preds_list, gts_list)]
            oracle_f1_mean  = nanmean(oracle_f1_vals)

            # Global-Opt-F1 (single threshold, correct calibration ceiling)
            global_opt, tau_star = global_opt_f1_for_pair(preds_list, gts_list)

            # ---- Calibration experiment at each N ----
            cal_f1_by_n  = {}
            rec_pct_by_n = {}
            for n in CAL_SIZES:
                cal_f1 = calibration_experiment(
                    preds_list, gts_list,
                    cal_size=n,
                    n_splits=args.n_splits,
                    seed=args.seed,
                )
                cal_f1_by_n[n] = cal_f1
                # Recovery % relative to Global-Opt-F1
                denom = global_opt - fixed_f1_mean
                if np.isnan(denom) or abs(denom) < 1e-9:
                    rec_pct = float("nan")
                else:
                    rec_pct = 100.0 * (cal_f1 - fixed_f1_mean) / denom
                rec_pct_by_n[n] = rec_pct

            row = {
                "method":           method,
                "dataset":          dataset,
                "fixed_f1":         fixed_f1_mean,
                "oracle_f1":        oracle_f1_mean,
                "global_opt_f1":    global_opt,
                "tau_star":         tau_star,
                "cal_f1_n10":       cal_f1_by_n[10],
                "cal_f1_n25":       cal_f1_by_n[25],
                "cal_f1_n50":       cal_f1_by_n[50],
                "cal_f1_n100":      cal_f1_by_n[100],
                "cal_f1_n200":      cal_f1_by_n[200],
                "recovery_pct_n10": rec_pct_by_n[10],
                "recovery_pct_n25": rec_pct_by_n[25],
            }
            rows.append(row)
            print(
                f"fixedF1={fixed_f1_mean:.3f}  oracleF1={oracle_f1_mean:.3f}  "
                f"globalOptF1={global_opt:.3f}  rec@10={rec_pct_by_n[10]:.1f}%"
            )

    if not rows:
        print("No results produced — check --results_dir or use --dry_run.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults CSV saved → {csv_path}")

    # ---- Summary statistics ----
    print("\n" + "=" * 70)
    print("SUMMARY: Recovery % at N=10 calibration images")
    print("=" * 70)
    rec10_all = df["recovery_pct_n10"].dropna()
    print(f"  All 112 pairs  : mean={rec10_all.mean():.1f}%  std={rec10_all.std():.1f}%")

    gen_mask = df["method"].apply(lambda m: m not in DOC_SPECIFIC_METHODS)
    doc_mask = ~gen_mask

    rec10_gen = df.loc[gen_mask, "recovery_pct_n10"].dropna()
    rec10_doc = df.loc[doc_mask, "recovery_pct_n10"].dropna()
    print(f"  General methods: mean={rec10_gen.mean():.1f}%  std={rec10_gen.std():.1f}%")
    print(f"  Doc-specific   : mean={rec10_doc.mean():.1f}%  std={rec10_doc.std():.1f}%")

    print("\n  Top-5 pairs by recovery @ N=10:")
    top5 = df.nlargest(5, "recovery_pct_n10")[
        ["method", "dataset", "fixed_f1", "global_opt_f1", "recovery_pct_n10"]
    ]
    print(top5.to_string(index=False))

    print("\n  Bottom-5 pairs by recovery @ N=10 (negative = calibration hurts):")
    bot5 = df.nsmallest(5, "recovery_pct_n10")[
        ["method", "dataset", "fixed_f1", "global_opt_f1", "recovery_pct_n10"]
    ]
    print(bot5.to_string(index=False))

    # ---- Heatmap ----
    plot_recovery_heatmap(
        df,
        metric_col="recovery_pct_n10",
        title="Calibration Recovery % at N=10 images (relative to Global-Opt-F1)",
        output_path=fig_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
