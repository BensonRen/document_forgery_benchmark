#!/usr/bin/env python3
"""
global_opt_f1.py
================
For each (method, dataset) pair across the full 14 x 8 = 112 benchmark pairs,
compute three F1 variants that serve as increasingly tight bounds on achievable
pixel-level performance:

  fixed_f1_05    : F1 at the standard tau = 0.5 threshold (as deployed by authors)
  oracle_f1      : per-image best F1 over 100 thresholds  (unreachable upper bound)
  global_opt_f1  : best mean F1 over all tampered images achievable with one
                   single threshold applied uniformly — the correct ceiling for
                   calibration experiments (paper, Discussion section, final paragraph)

Additionally records:
  tau_star       : the globally-optimal threshold for each pair
  auc            : mean pixel-level AUC over tampered images

The core insight (from the paper's Discussion): Oracle-F1 is an unachievable per-image
optimum.  Global-Opt-F1 is the tightest achievable single-threshold ceiling.
Recovery percentages in the calibration experiment should be expressed relative
to Global-Opt-F1, not Oracle-F1, to avoid overstating the headroom.

Outputs
-------
  output/global_opt_f1_results.csv            — full 112-pair results
  output/fig_oracle_vs_global_opt_f1.pdf      — scatter comparing the two F1 ceilings
  output/tab_f1_variants_representative.tex   — LaTeX table for representative pairs

Data layout expected on disk
-----------------------------
  {results_dir}/{method}/{dataset}/predictions/{image_id}.npy   <- float [0,1]
  {results_dir}/{method}/{dataset}/gt/{image_id}.npy            <- binary {0,1}

Authentic images (GT sum == 0) are excluded from F1 / AUC computations.

Usage
-----
  # With real data:
  python global_opt_f1.py --results_dir results/

  # Smoke-test with synthetic data:
  python global_opt_f1.py --dry_run
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
from sklearn.metrics import f1_score, roc_auc_score

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

DOC_SPECIFIC_METHODS = {
    "doctamper", "dtd", "ffdn", "caftb_net", "tifdm", "ascformer", "adcd_net",
}

N_THRESHOLDS_GLOBAL  = 200   # threshold sweep for global_opt_f1
N_THRESHOLDS_ORACLE  = 100   # threshold sweep for per-image oracle_f1

# Representative (method, dataset) pairs for the LaTeX table.
# These are chosen to span the performance spectrum (high/mid/low F1),
# both general and doc-specific methods, and include known negative-calibration cases.
REPRESENTATIVE_PAIRS = [
    ("trufor",    "mixtamper"),       # high fixed F1; calibration hurts
    ("trufor",    "receipt_forgery"), # high AUC, low fixed F1 — score shift
    ("pscc_net",  "fsts_1.5k"),       # strong calibration recovery in §8.1
    ("pscc_net",  "doctamper"),       # also good recovery
    ("ffdn",      "tsroie"),          # FFDN out-of-domain, §8.1
    ("caftb_net", "tsroie"),          # degenerate score distribution (negative ctrl)
    ("doctamper", "doctamper"),       # in-domain specialist
    ("ascformer", "tsroie"),          # best cross-dataset doc-specific
    ("cat_net",   "tampered_ic13"),   # JPEG-heavy dataset
    ("iml_vit",   "fantasyid"),       # ID document dataset
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def is_authentic(gt: np.ndarray) -> bool:
    return gt.flatten().sum() == 0


def compute_f1_at_threshold(pred_flat: np.ndarray, gt_flat: np.ndarray,
                             threshold: float) -> float:
    pred_bin = (pred_flat >= threshold).astype(np.int32)
    return float(f1_score(gt_flat, pred_bin, zero_division=0))


def per_image_oracle_f1(pred: np.ndarray, gt: np.ndarray,
                         n_thresholds: int = N_THRESHOLDS_ORACLE) -> float:
    """Best F1 over a threshold grid, for a single tampered image."""
    gt_flat   = gt.flatten().astype(np.int32)
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
    return compute_f1_at_threshold(
        pred.flatten().astype(np.float32),
        gt.flatten().astype(np.int32),
        threshold,
    )


def per_image_pixel_auc(pred: np.ndarray, gt: np.ndarray) -> float:
    gt_flat = gt.flatten().astype(np.int32)
    try:
        return float(roc_auc_score(gt_flat, pred.flatten().astype(np.float64)))
    except Exception:
        return float("nan")


def nanmean(values) -> float:
    arr = np.array(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Global-Opt-F1 computation
# ---------------------------------------------------------------------------

def compute_global_opt_f1(preds_list, gts_list,
                            n_thresholds: int = N_THRESHOLDS_GLOBAL):
    """
    Find the single threshold tau* in [0, 1] that maximises mean F1 over all
    tampered images.  Authentic images (GT all-zero) are excluded.

    Returns (global_opt_f1, tau_star).
    """
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
        f1s = [compute_f1_at_threshold(p, g, t)
               for p, g in zip(tampered_preds, tampered_gts)]
        mean_f1 = float(np.mean(f1s))
        if mean_f1 > best_f1:
            best_f1  = mean_f1
            best_tau = float(t)

    return best_f1, best_tau


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions_and_gts(results_dir: str, method: str, dataset: str):
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
        pred = np.clip(np.load(str(pf)).astype(np.float32), 0.0, 1.0)
        gt   = np.load(str(gf)).astype(np.int32)
        preds_list.append(pred)
        gts_list.append(gt)

    return preds_list, gts_list


# ---------------------------------------------------------------------------
# Synthetic data for --dry_run
# ---------------------------------------------------------------------------

def make_synthetic_data(n_images: int = 60,
                         h: int = 64, w: int = 64,
                         authentic_frac: float = 0.15,
                         rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    n_authentic = max(1, int(n_images * authentic_frac))
    n_tampered  = n_images - n_authentic

    preds_list, gts_list = [], []

    for _ in range(n_tampered):
        gt = np.zeros((h, w), dtype=np.int32)
        r0 = rng.integers(0, h // 2)
        r1 = r0 + rng.integers(max(1, h // 20), max(2, h // 4))
        c0 = rng.integers(0, w // 2)
        c1 = c0 + rng.integers(max(1, w // 20), max(2, w // 4))
        gt[r0:r1, c0:c1] = 1
        pred = rng.uniform(0.0, 0.15, (h, w)).astype(np.float32)
        pred[r0:r1, c0:c1] = rng.uniform(0.15, 0.45, (r1 - r0, c1 - c0))
        preds_list.append(pred)
        gts_list.append(gt)

    for _ in range(n_authentic):
        gt   = np.zeros((h, w), dtype=np.int32)
        pred = rng.uniform(0.0, 0.10, (h, w)).astype(np.float32)
        preds_list.append(pred)
        gts_list.append(gt)

    return preds_list, gts_list


# ---------------------------------------------------------------------------
# Scatter figure: Oracle-F1 vs Global-Opt-F1
# ---------------------------------------------------------------------------

def plot_oracle_vs_global_opt(df: pd.DataFrame, output_path: str):
    """
    Scatter of Oracle-F1 (x) vs Global-Opt-F1 (y) for all 112 (method, dataset)
    pairs.  Points above the diagonal indicate pairs where Oracle-F1 significantly
    overestimates the achievable calibration ceiling.
    """
    subset = df.dropna(subset=["oracle_f1", "global_opt_f1"])
    if subset.empty:
        print("  No data for Oracle vs Global-Opt scatter.")
        return

    dataset_colors = {
        ds: plt.cm.tab10(i / max(1, len(DATASETS) - 1))
        for i, ds in enumerate(DATASETS)
    }
    marker_map = {m: ("o" if m not in DOC_SPECIFIC_METHODS else "s")
                  for m in METHODS}

    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in subset.iterrows():
        color  = dataset_colors.get(row["dataset"], "gray")
        marker = marker_map.get(row["method"], "o")
        ax.scatter(
            row["oracle_f1"], row["global_opt_f1"],
            c=[color], marker=marker, s=55, alpha=0.75,
            linewidths=0.4, edgecolors="k",
        )

    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Oracle = Global-Opt")

    # Shade region where Oracle-F1 overestimates by >0.1
    xs = np.linspace(0, 1, 100)
    ax.fill_between(xs, xs, np.maximum(0, xs - 0.1), alpha=0.06, color="red",
                    label="Oracle overestimates by >0.10")

    # Dataset colour legend
    for ds in DATASETS:
        ax.scatter([], [], c=[dataset_colors[ds]], s=55, label=ds)
    # Method category legend
    ax.scatter([], [], c="gray", marker="o", s=55,
               label="General method", edgecolors="k")
    ax.scatter([], [], c="gray", marker="s", s=55,
               label="Doc-specific method", edgecolors="k")

    ax.set_xlabel("Oracle-F1 (per-image best threshold)", fontsize=11)
    ax.set_ylabel("Global-Opt-F1 (single threshold, all images)", fontsize=11)
    ax.set_title(
        "Oracle-F1 vs Global-Opt-F1 across all 112 (method, dataset) pairs\n"
        "Points above diagonal: Global-Opt exceeds Oracle (numerical artefact)\n"
        "Points below diagonal: Oracle overestimates the achievable ceiling",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved scatter plot → {output_path}")


# ---------------------------------------------------------------------------
# LaTeX table for representative pairs
# ---------------------------------------------------------------------------

def build_representative_latex_table(df: pd.DataFrame,
                                      pairs: list) -> str:
    """
    Build a LaTeX table showing fixed_f1_05, oracle_f1, global_opt_f1, tau_star,
    and auc for a curated selection of representative (method, dataset) pairs.
    """
    lines = []
    lines.append(r"% Auto-generated by global_opt_f1.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Three F1 variants for representative (method, dataset) pairs.")
    lines.append(r"    Fixed-F1: $\tau{=}0.5$.")
    lines.append(r"    Oracle-F1: per-image best threshold (unreachable ceiling).")
    lines.append(r"    Global-Opt-F1: best single threshold over all images")
    lines.append(r"    (the correct calibration ceiling; see \S\ref{sec:discussion}).}")
    lines.append(r"  \label{tab:f1_variants_representative}")
    lines.append(r"  \resizebox{\linewidth}{!}{%")
    lines.append(r"  \begin{tabular}{llccccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Method & Dataset & Fixed-F1 & Oracle-F1 & Global-Opt-F1"
        r" & $\tau^*$ & Pixel-AUC \\"
    )
    lines.append(r"    \midrule")

    def fmt(val, decimals=3):
        if isinstance(val, float) and np.isnan(val):
            return "---"
        return f"{val:.{decimals}f}"

    for method, dataset in pairs:
        row = df[(df["method"] == method) & (df["dataset"] == dataset)]
        if row.empty:
            continue
        row = row.iloc[0]

        m_str = (r"\textit{" + method + r"}"
                 if method in DOC_SPECIFIC_METHODS else method)

        # Bold the highest F1 among the three variants for easy reading
        f_vals = {
            "fixed":  row["fixed_f1_05"],
            "oracle": row["oracle_f1"],
            "global": row["global_opt_f1"],
        }
        valid_vals = {k: v for k, v in f_vals.items() if not np.isnan(v)}
        if valid_vals:
            best_key = max(valid_vals, key=valid_vals.get)
        else:
            best_key = None

        def fmt_f1(key, val):
            s = fmt(val)
            if key == best_key and s != "---":
                return r"\textbf{" + s + r"}"
            return s

        lines.append(
            f"    {m_str} & {dataset} & "
            f"{fmt_f1('fixed',  row['fixed_f1_05'])} & "
            f"{fmt_f1('oracle', row['oracle_f1'])} & "
            f"{fmt_f1('global', row['global_opt_f1'])} & "
            f"{fmt(row['tau_star'])} & "
            f"{fmt(row['auc'])} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \smallskip")
    lines.append(r"  {\small \textit{Italics}: document-specific methods; "
                 r"\textbf{bold}: highest F1 variant for that pair.}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute Global-Opt-F1 vs Oracle-F1 for all 112 benchmark pairs."
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
        help="Run with synthetic data — no real prediction files needed.",
    )
    p.add_argument(
        "--n_thresholds", type=int, default=N_THRESHOLDS_GLOBAL,
        help=f"Number of threshold candidates in [0,1] sweep (default {N_THRESHOLDS_GLOBAL}).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path     = os.path.join(args.output_dir, "global_opt_f1_results.csv")
    fig_path     = os.path.join(args.output_dir, "fig_oracle_vs_global_opt_f1.pdf")
    tex_path     = os.path.join(args.output_dir, "tab_f1_variants_representative.tex")

    rows = []
    total = len(METHODS) * len(DATASETS)

    for m_idx, method in enumerate(METHODS):
        for d_idx, dataset in enumerate(DATASETS):
            idx   = m_idx * len(DATASETS) + d_idx + 1
            label = f"{method}/{dataset}"
            print(f"[{idx:3d}/{total}] {label}", end=" ... ", flush=True)

            # ---- Load (or synthesise) data ----
            if args.dry_run:
                rng_seed = hash((method, dataset)) % (2 ** 31)
                preds_list, gts_list = make_synthetic_data(rng_seed=rng_seed)
            else:
                try:
                    preds_list, gts_list = load_predictions_and_gts(
                        args.results_dir, method, dataset
                    )
                except FileNotFoundError as e:
                    print(f"SKIP ({e})")
                    continue

            if not preds_list:
                print("SKIP (no images)")
                continue

            # Only tampered images contribute to F1 and AUC statistics
            tampered_preds = [(p, g) for p, g in zip(preds_list, gts_list)
                               if not is_authentic(g)]

            if not tampered_preds:
                print("SKIP (no tampered images)")
                continue

            # ---- Fixed F1 @ tau = 0.5 ----
            fixed_f1_vals = [
                per_image_fixed_f1(p, g, 0.5)
                for p, g in tampered_preds
            ]
            fixed_f1_mean = nanmean(fixed_f1_vals)

            # ---- Oracle F1 ----
            oracle_f1_vals = [
                per_image_oracle_f1(p, g)
                for p, g in tampered_preds
            ]
            oracle_f1_mean = nanmean(oracle_f1_vals)

            # ---- Global-Opt-F1 ----
            global_opt, tau_star = compute_global_opt_f1(
                preds_list, gts_list, n_thresholds=args.n_thresholds
            )

            # ---- Mean pixel-level AUC ----
            auc_vals = [per_image_pixel_auc(p, g) for p, g in tampered_preds]
            mean_auc = nanmean(auc_vals)

            row = {
                "method":          method,
                "dataset":         dataset,
                "fixed_f1_05":     fixed_f1_mean,
                "oracle_f1":       oracle_f1_mean,
                "global_opt_f1":   global_opt,
                "tau_star":        tau_star,
                "auc":             mean_auc,
                "n_tampered":      len(tampered_preds),
                "n_total":         len(preds_list),
            }
            rows.append(row)

            # How much does Oracle-F1 overestimate Global-Opt-F1?
            overestimate = (oracle_f1_mean - global_opt
                            if not (np.isnan(oracle_f1_mean) or np.isnan(global_opt))
                            else float("nan"))

            print(
                f"fixedF1={fixed_f1_mean:.3f}  "
                f"oracle={oracle_f1_mean:.3f}  "
                f"globalOpt={global_opt:.3f}  "
                f"tau*={tau_star:.3f}  "
                f"AUC={mean_auc:.3f}  "
                f"overest={overestimate:+.3f}"
            )

    if not rows:
        print("No results produced — check --results_dir or use --dry_run.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nResults CSV saved → {csv_path}")

    # ---- Summary statistics ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    overest = (df["oracle_f1"] - df["global_opt_f1"]).dropna()
    print(f"\n  Oracle-F1 overestimation vs Global-Opt-F1:")
    print(f"    Mean  : {overest.mean():.3f}")
    print(f"    Median: {overest.median():.3f}")
    print(f"    Max   : {overest.max():.3f}")
    print(f"    Fraction where overest > 0.10: "
          f"{(overest > 0.10).mean() * 100:.1f}%")

    print(f"\n  Global-Opt tau* distribution:")
    tau_vals = df["tau_star"].dropna()
    print(f"    Mean  : {tau_vals.mean():.3f}")
    print(f"    Median: {tau_vals.median():.3f}")
    print(f"    % with tau* < 0.3 (score-shifted): "
          f"{(tau_vals < 0.3).mean() * 100:.1f}%")
    print(f"    % with tau* in [0.4, 0.6] (near-standard): "
          f"{((tau_vals >= 0.4) & (tau_vals <= 0.6)).mean() * 100:.1f}%")

    print(f"\n  Mean Global-Opt-F1 by method category:")
    gen_mask = df["method"].apply(lambda m: m not in DOC_SPECIFIC_METHODS)
    doc_mask = ~gen_mask
    print(f"    General     : {df.loc[gen_mask, 'global_opt_f1'].mean():.3f}")
    print(f"    Doc-specific: {df.loc[doc_mask, 'global_opt_f1'].mean():.3f}")

    # ---- Scatter figure ----
    plot_oracle_vs_global_opt(df, fig_path)

    # ---- LaTeX table ----
    # Filter representative pairs to those actually present in results
    available_pairs = [
        (m, d) for m, d in REPRESENTATIVE_PAIRS
        if not df[(df["method"] == m) & (df["dataset"] == d)].empty
    ]
    if available_pairs:
        tex_str = build_representative_latex_table(df, available_pairs)
        with open(tex_path, "w") as fh:
            fh.write(tex_str)
        print(f"LaTeX table saved → {tex_path}")
    else:
        print("No representative pairs found in results — LaTeX table not written.")

    print("\nDone.")


if __name__ == "__main__":
    main()
