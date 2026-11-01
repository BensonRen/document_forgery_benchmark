#!/usr/bin/env python3
"""
image_level_auroc.py
=====================
Compute image-level binary detection AUROC for the three mixed-label datasets:
  - tampered_ic13    : 45 authentic  + 188 tampered
  - receipt_forgery  : 183 authentic + 35  forged
  - fantasyid        : 600 bonafide  + 2173 attack (test split)

For each method x dataset combination:
  - Load all predictions, including authentic images.
  - Derive a single image-level score using three aggregation strategies:
      * max_pixel   : max pixel value in the prediction map
      * top1pct_mean: mean of the top 1% of pixel scores
      * top5pct_mean: mean of the top 5% of pixel scores
  - Compute AUROC with labels: authentic=0, tampered/forged=1.

Outputs
-------
  output/image_level_auroc.csv          — tidy CSV
  output/fig_image_level_auroc.pdf      — AUROC vs pixel-AUC scatter figure
  output/tab_image_level_auroc.tex      — LaTeX table for appendix

Data layout expected on disk
-----------------------------
  {results_dir}/{method}/{dataset}/predictions/{image_id}.npy   <- float prediction
  {results_dir}/{method}/{dataset}/gt/{image_id}.npy            <- binary GT mask

Authentic images are identified by GT masks where sum == 0.

Usage
-----
  # With real data:
  python image_level_auroc.py --results_dir results/

  # Smoke-test with synthetic data:
  python image_level_auroc.py --dry_run
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS = [
    "trufor", "mantranet", "mvss_net", "cat_net", "pscc_net",
    "iml_vit", "safire", "doctamper", "dtd", "ffdn",
    "caftb_net", "tifdm", "ascformer", "adcd_net",
]

# The three mixed-label datasets with authentic + tampered images.
MIXED_DATASETS = {
    "tampered_ic13":  {"n_authentic": 45,   "n_tampered": 188},
    "receipt_forgery": {"n_authentic": 183,  "n_tampered": 35},
    "fantasyid":       {"n_authentic": 600,  "n_tampered": 2173},
}

AGGREGATIONS = ["max_pixel", "top1pct_mean", "top5pct_mean"]

# Doc-specific category for colour coding in figures
DOC_SPECIFIC_METHODS = {
    "doctamper", "dtd", "ffdn", "caftb_net", "tifdm", "ascformer", "adcd_net",
}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def image_level_score(pred: np.ndarray, aggregation: str) -> float:
    """Reduce a (H, W) prediction map to a single image-level score."""
    flat = pred.flatten().astype(np.float64)
    if aggregation == "max_pixel":
        return float(flat.max())
    elif aggregation == "top1pct_mean":
        k = max(1, int(len(flat) * 0.01))
        return float(np.partition(flat, -k)[-k:].mean())
    elif aggregation == "top5pct_mean":
        k = max(1, int(len(flat) * 0.05))
        return float(np.partition(flat, -k)[-k:].mean())
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def pixel_auc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel-level AUC. Returns NaN for authentic images (no positive pixels)."""
    gt_flat = gt.flatten().astype(np.int32)
    if gt_flat.sum() == 0:
        return float("nan")
    try:
        return float(roc_auc_score(gt_flat, pred.flatten().astype(np.float64)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions_and_gts(results_dir: str, method: str, dataset: str):
    """
    Load all predictions and GT masks for one (method, dataset) pair.

    Returns (preds_list, gts_list, image_ids).
    Raises FileNotFoundError if directories are missing.
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

    preds_list, gts_list, image_ids = [], [], []
    for pf in pred_files:
        gf = gt_dir / pf.name
        if not gf.exists():
            warnings.warn(f"GT file missing for {pf.name} — skipping")
            continue
        pred = np.clip(np.load(str(pf)).astype(np.float32), 0.0, 1.0)
        gt   = np.load(str(gf)).astype(np.int32)
        preds_list.append(pred)
        gts_list.append(gt)
        image_ids.append(pf.stem)

    return preds_list, gts_list, image_ids


# ---------------------------------------------------------------------------
# Synthetic data for --dry_run
# ---------------------------------------------------------------------------

def make_synthetic_mixed_data(n_authentic: int, n_tampered: int,
                               h: int = 64, w: int = 64,
                               rng_seed: int = 0):
    """
    Generate synthetic data that mimics a mixed-label dataset.

    Tampered images: small forged rectangle; prediction map is signal-shifted.
    Authentic images: all-zero GT; prediction map is pure low-score noise.
    """
    rng = np.random.default_rng(rng_seed)
    preds_list, gts_list = [], []

    # Tampered images
    for i in range(n_tampered):
        gt = np.zeros((h, w), dtype=np.int32)
        r0 = rng.integers(0, h // 2)
        r1 = r0 + rng.integers(max(1, h // 20), max(2, h // 4))
        c0 = rng.integers(0, w // 2)
        c1 = c0 + rng.integers(max(1, w // 20), max(2, w // 4))
        gt[r0:r1, c0:c1] = 1
        # Score map: slightly elevated in forged region
        pred = rng.uniform(0.0, 0.15, (h, w)).astype(np.float32)
        pred[r0:r1, c0:c1] = rng.uniform(0.20, 0.55, (r1 - r0, c1 - c0))
        preds_list.append(pred)
        gts_list.append(gt)

    # Authentic images
    for _ in range(n_authentic):
        gt   = np.zeros((h, w), dtype=np.int32)
        pred = rng.uniform(0.0, 0.12, (h, w)).astype(np.float32)
        preds_list.append(pred)
        gts_list.append(gt)

    return preds_list, gts_list


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def build_latex_table(df: pd.DataFrame) -> str:
    """
    Build a LaTeX table for appendix inclusion.

    Table structure: method | dataset | max_pixel AUROC | top1% AUROC | top5% AUROC
    Grouped by dataset.
    """
    lines = []
    lines.append(r"% Auto-generated by image_level_auroc.py — ready for appendix inclusion")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Image-level detection AUROC on mixed-label datasets.")
    lines.append(r"    Scores derived by three pixel-aggregation strategies.")
    lines.append(r"    Authentic\,=\,0, Tampered/Forged\,=\,1.}")
    lines.append(r"  \label{tab:image_level_auroc}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{llccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Method & Dataset & Max Pixel & Top-1\% Mean & Top-5\% Mean \\")
    lines.append(r"    \midrule")

    prev_dataset = None
    for _, row in df.sort_values(["dataset", "method"]).iterrows():
        ds = row["dataset"]
        if ds != prev_dataset:
            if prev_dataset is not None:
                lines.append(r"    \midrule")
            prev_dataset = ds

        def fmt(val):
            return f"{val:.3f}" if not np.isnan(val) else "---"

        # Highlight doc-specific methods with a marker
        method_str = row["method"]
        if method_str in DOC_SPECIFIC_METHODS:
            method_str = r"\textit{" + method_str + r"}"

        lines.append(
            f"    {method_str} & {ds} & "
            f"{fmt(row['auroc_max_pixel'])} & "
            f"{fmt(row['auroc_top1pct_mean'])} & "
            f"{fmt(row['auroc_top5pct_mean'])} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \smallskip")
    lines.append(r"  {\small \textit{Italics}: document-specific methods.}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Scatter figure: image-level AUROC vs pixel-level AUC
# ---------------------------------------------------------------------------

def plot_auroc_scatter(df_auroc: pd.DataFrame, df_pixel_auc: pd.DataFrame,
                        aggregation: str, output_path: str):
    """
    Scatter plot comparing image-level AUROC (y-axis) vs pixel-level AUC (x-axis)
    for a given aggregation strategy.

    df_pixel_auc expected columns: method, dataset, pixel_auc
    """
    auroc_col = f"auroc_{aggregation}"
    merged = pd.merge(
        df_auroc[["method", "dataset", auroc_col]],
        df_pixel_auc[["method", "dataset", "pixel_auc"]],
        on=["method", "dataset"],
        how="inner",
    ).dropna(subset=[auroc_col, "pixel_auc"])

    if merged.empty:
        print(f"  No data for scatter plot (aggregation={aggregation})")
        return

    dataset_colors = {
        "tampered_ic13":  "#1f77b4",
        "receipt_forgery": "#ff7f0e",
        "fantasyid":       "#2ca02c",
    }
    marker_map = {m: ("o" if m not in DOC_SPECIFIC_METHODS else "s")
                  for m in METHODS}

    fig, ax = plt.subplots(figsize=(7, 6))

    for ds, grp in merged.groupby("dataset"):
        color = dataset_colors.get(ds, "gray")
        for _, row in grp.iterrows():
            marker = marker_map.get(row["method"], "o")
            ax.scatter(
                row["pixel_auc"], row[auroc_col],
                c=color, marker=marker, s=60, alpha=0.8, linewidths=0.5,
                edgecolors="k",
            )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="y=x")

    # Legend for datasets
    for ds, color in dataset_colors.items():
        ax.scatter([], [], c=color, s=60, label=ds)
    # Legend for method category
    ax.scatter([], [], c="gray", marker="o", s=60,
               label="General (circle)", edgecolors="k")
    ax.scatter([], [], c="gray", marker="s", s=60,
               label="Doc-specific (square)", edgecolors="k")

    ax.set_xlabel("Pixel-level AUC", fontsize=11)
    ax.set_ylabel(f"Image-level AUROC ({aggregation.replace('_', ' ')})", fontsize=11)
    ax.set_title(
        f"Image-level AUROC vs Pixel-level AUC\n"
        f"(aggregation: {aggregation.replace('_', ' ')})",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved scatter plot → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Image-level AUROC on mixed-label datasets."
    )
    p.add_argument(
        "--results_dir", default="results/",
        help="Root directory containing per-method/per-dataset predictions and GT.",
    )
    p.add_argument(
        "--output_dir", default="paper/scripts/output/",
        help="Directory for CSV, PDF, and .tex outputs.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Run with synthetic data — no real files needed.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path     = os.path.join(args.output_dir, "image_level_auroc.csv")
    fig_base     = os.path.join(args.output_dir, "fig_image_level_auroc")
    tex_path     = os.path.join(args.output_dir, "tab_image_level_auroc.tex")

    auroc_rows    = []
    pixel_auc_rows = []

    total = len(METHODS) * len(MIXED_DATASETS)
    counter = 0
    for method in METHODS:
        for dataset, ds_info in MIXED_DATASETS.items():
            counter += 1
            pair_label = f"{method}/{dataset}"
            print(f"[{counter:3d}/{total}] {pair_label}", end=" ... ", flush=True)

            # ---- Load (or synthesise) data ----
            if args.dry_run:
                rng_seed = hash((method, dataset)) % (2 ** 31)
                preds_list, gts_list = make_synthetic_mixed_data(
                    n_authentic=ds_info["n_authentic"],
                    n_tampered=ds_info["n_tampered"],
                    rng_seed=rng_seed,
                )
            else:
                try:
                    preds_list, gts_list, _ = load_predictions_and_gts(
                        args.results_dir, method, dataset
                    )
                except FileNotFoundError as e:
                    print(f"SKIP ({e})")
                    continue

            if len(preds_list) == 0:
                print("SKIP (no images)")
                continue

            # Binary image-level label: 0=authentic, 1=tampered/forged
            image_labels = np.array(
                [0 if g.sum() == 0 else 1 for g in gts_list],
                dtype=np.int32,
            )

            # Sanity check: need at least one of each class for AUROC
            if image_labels.sum() == 0 or (1 - image_labels).sum() == 0:
                print("SKIP (only one class present)")
                continue

            # ---- Image-level AUROC for each aggregation ----
            auroc_row = {"method": method, "dataset": dataset}
            for agg in AGGREGATIONS:
                scores = np.array(
                    [image_level_score(p, agg) for p in preds_list],
                    dtype=np.float64,
                )
                try:
                    auroc = float(roc_auc_score(image_labels, scores))
                except Exception:
                    auroc = float("nan")
                auroc_row[f"auroc_{agg}"] = auroc

            auroc_rows.append(auroc_row)

            # ---- Mean pixel-level AUC over tampered images only ----
            pix_aucs = []
            for pred, gt in zip(preds_list, gts_list):
                auc_val = pixel_auc(pred, gt)
                if not np.isnan(auc_val):
                    pix_aucs.append(auc_val)
            mean_pix_auc = float(np.mean(pix_aucs)) if pix_aucs else float("nan")
            pixel_auc_rows.append({
                "method": method, "dataset": dataset, "pixel_auc": mean_pix_auc
            })

            print(
                f"img-AUROC(max)={auroc_row['auroc_max_pixel']:.3f}  "
                f"img-AUROC(top1%)={auroc_row['auroc_top1pct_mean']:.3f}  "
                f"pix-AUC={mean_pix_auc:.3f}"
            )

    if not auroc_rows:
        print("No results produced — check --results_dir or use --dry_run.")
        return

    df_auroc    = pd.DataFrame(auroc_rows)
    df_pix_auc  = pd.DataFrame(pixel_auc_rows)

    # ---- Save CSV ----
    # Melt to tidy format: one row per (method, dataset, aggregation)
    id_cols    = ["method", "dataset"]
    auroc_cols = [f"auroc_{a}" for a in AGGREGATIONS]
    tidy_rows  = []
    for _, row in df_auroc.iterrows():
        for agg in AGGREGATIONS:
            tidy_rows.append({
                "method":      row["method"],
                "dataset":     row["dataset"],
                "aggregation": agg,
                "image_auroc": row[f"auroc_{agg}"],
            })
    df_tidy = pd.DataFrame(tidy_rows)
    df_tidy.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults CSV saved → {csv_path}")

    # ---- LaTeX table ----
    tex_str = build_latex_table(df_auroc)
    with open(tex_path, "w") as fh:
        fh.write(tex_str)
    print(f"LaTeX table saved → {tex_path}")

    # ---- Scatter figures (one per aggregation) ----
    for agg in AGGREGATIONS:
        fig_path = f"{fig_base}_{agg}.pdf"
        plot_auroc_scatter(df_auroc, df_pix_auc, agg, fig_path)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY: Mean image-level AUROC by dataset and aggregation")
    print("=" * 70)
    for ds in MIXED_DATASETS:
        subset = df_auroc[df_auroc["dataset"] == ds]
        if subset.empty:
            continue
        print(f"\n  {ds}:")
        for agg in AGGREGATIONS:
            col = f"auroc_{agg}"
            vals = subset[col].dropna()
            if len(vals):
                print(f"    {agg:20s}: mean={vals.mean():.3f}  "
                      f"best={vals.max():.3f} ({subset.loc[vals.idxmax(), 'method']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
