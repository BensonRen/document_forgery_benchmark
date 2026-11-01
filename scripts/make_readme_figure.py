#!/usr/bin/env python3
"""Regenerate the README teaser figure (the "calibration gap").

Reads results/paper/main_results.csv and writes
assets/calibration_gap.png (2x scale) and assets/calibration_gap.svg.

Each method is averaged over the datasets that report all three metrics
(Pixel-F1@0.5, Oracle-F1, Pixel-AUC), i.e. MixTamper is excluded so the
three means are over the same 7 datasets.

Usage:  python scripts/make_readme_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "paper" / "main_results.csv"
OUT = ROOT / "assets"

METRICS = [  # (column, legend label, color, marker) -- ordered low -> high
    ("pixel_f1", "Pixel-F1 @ 0.5", "#eb6834", "o"),
    ("oracle_f1", "Oracle-F1 (best per-image threshold)", "#1baf7a", "D"),
    ("pixel_auc", "Pixel-AUC", "#2a78d6", "s"),
]
GROUPS = [("document", "Document-specific"), ("general", "General-purpose")]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, CONNECT, BG = "#e1e0d9", "#c3c2b7", "#ffffff"


def load_means() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    cols = [m[0] for m in METRICS]
    full = df.dropna(subset=cols)
    n_ds = full["dataset"].nunique()
    means = (
        full.groupby(["method", "method_type"], as_index=False)[cols]
        .mean()
        .assign(n_datasets=full.groupby("method")["dataset"].nunique().values)
    )
    assert (means["n_datasets"] == n_ds).all(), "methods cover different datasets"
    return means, n_ds


def main() -> None:
    means, n_ds = load_means()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # y positions: group header, then rows sorted by Pixel-AUC (highest on top)
    rows, headers, y = [], [], 0.0
    for key, label in GROUPS:
        headers.append((y, label))
        y += 1.0
        g = means[means.method_type == key].sort_values("pixel_auc", ascending=False)
        for _, r in g.iterrows():
            rows.append((y, r))
            y += 1.0
        y += 0.35  # breathing room between groups
    ymax = y - 0.35

    for yy, r in rows:
        lo, hi = r["pixel_f1"], r["pixel_auc"]
        ax.plot([lo, hi], [yy, yy], color=CONNECT, lw=2, solid_capstyle="round", zorder=1)
        # F1@0.5 drawn last so it stays visible when it coincides with Oracle-F1
        for z, (col, _, color, marker) in zip((5, 3, 4), METRICS):
            ax.scatter(r[col], yy, s=58 if marker != "D" else 46, marker=marker,
                       color=color, edgecolor=BG, linewidth=1.6, zorder=z)
        ax.text(-0.015, yy, r["method"], ha="right", va="center", color=INK, fontsize=10.5)

    for yy, label in headers:
        ax.text(-0.015, yy + 0.05, label.upper(), ha="right", va="center",
                color=INK2, fontsize=9, fontweight="bold")

    # Direct labels on the first data row only (identity also carried by legend + shape)
    y0, r0 = rows[0]
    for col, _, _, _ in METRICS:
        ax.text(r0[col], y0 - 0.55, f"{r0[col]:.2f}", ha="center", va="bottom",
                color=INK2, fontsize=9)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(ymax - 0.4, -0.6)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.tick_params(axis="x", colors=MUTED, length=0, labelsize=10, pad=6)
    ax.set_yticks([])
    for x in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.axvline(x, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=CONNECT, lw=1, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(f"Mean score over {n_ds} datasets (higher is better)", color=MUTED,
                  fontsize=10, labelpad=8)

    # Title, subtitle, legend (top-left aligned to figure margin)
    fig.text(0.02, 0.965, "High AUC, low F1: zero-shot detectors are miscalibrated on documents",
             ha="left", va="top", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, f"Mean over {n_ds} datasets · DocForge-Bench (arXiv:2603.01433)",
             ha="left", va="top", fontsize=10.5, color=INK2)
    handles = [
        plt.Line2D([], [], ls="none", marker=m, ms=8 if m != "D" else 7, color=c,
                   markeredgecolor=BG, label=lab)
        for _, lab, c, m in METRICS
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.865), ncol=3,
               frameon=False, fontsize=10, handletextpad=0.3, columnspacing=1.6,
               labelcolor=INK2)

    fig.subplots_adjust(left=0.2, right=0.975, top=0.77, bottom=0.11)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "calibration_gap.png", dpi=200, facecolor=BG)
    fig.savefig(OUT / "calibration_gap.svg", facecolor=BG)

    show = means.sort_values(["method_type", "pixel_auc"], ascending=[True, False])
    print(show[["method", "method_type", "pixel_f1", "oracle_f1", "pixel_auc"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
