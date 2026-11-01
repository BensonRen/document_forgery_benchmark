#!/usr/bin/env python3
"""compare_validation_results.py

Compare validation run results against paper-reported performance targets.
Run after:
    python scripts/run_exp1.py --config configs/exp_validation.yaml

Usage:
    cd experiments
    python scripts/compare_validation_results.py [--results results/E_validation/results.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ─── Paper reference targets ─────────────────────────────────────────────────
# Source references noted inline.
# None = "just log our value, no exact target to compare against"
TARGETS = {
    "TruFor": {
        # CVPR 2023 supp Table S1 (pixel-level optimal F1 under MVSS-protocol)
        "Columbia": {"optimal_f1": 0.807},
        "COVERAGE": {"optimal_f1": 0.525},
    },
    "MVSS-Net": {
        # ICCV 2021 Table 2 (image-level AUC, but we use pixel_auc as proxy)
        "Columbia": {"pixel_auc": 0.980},
        "COVERAGE": {"pixel_auc": 0.731},
    },
    "IML-ViT": {
        # arXiv 2307.14863 Table 3
        "Columbia": {"optimal_f1": 0.810},
        "COVERAGE": {"optimal_f1": None},  # log only
    },
    "HiFi-Net": {
        # CVPR 2023 — verifying non-zero localization on training domain
        "Columbia": {"optimal_f1": None},  # no exact target; must be > 0
        "COVERAGE": {"optimal_f1": None},
    },
    "PSCC-Net": {
        # TCSVT 2022 — no exact published target for COVERAGE/Columbia
        "Columbia": {"optimal_f1": None},
        "COVERAGE": {"optimal_f1": None},
    },
}

THRESHOLD_PASS = 0.05   # ±5% relative tolerance
HIFINET_MIN_F1 = 0.05   # HiFi-Net must exceed this on Columbia (sanity check)


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(results: dict) -> int:
    """Print comparison table. Returns number of FAILED checks."""
    n_fail = 0
    n_pass = 0
    n_log  = 0

    print()
    print("=" * 72)
    print("  Validation: Paper vs Benchmark Comparison")
    print("=" * 72)
    print(f"  {'Method':<12} {'Dataset':<12} {'Metric':<14} {'Paper':>8} "
          f"{'Ours':>8}  {'RelErr':>7}  Status")
    print("  " + "-" * 68)

    for method, datasets in TARGETS.items():
        method_results = results.get(method, {})
        for dataset, metrics in datasets.items():
            ds_results = method_results.get(dataset, {})
            for metric, target in metrics.items():
                got = ds_results.get(metric, float("nan"))

                if math.isnan(got):
                    status = "⚠  NO DATA (dataset not run)"
                    n_log += 1
                    target_str = f"{target:.3f}" if target is not None else "   —   "
                    print(f"  {method:<12} {dataset:<12} {metric:<14} {target_str:>8} "
                          f"{'N/A':>8}  {'N/A':>7}  {status}")
                    continue

                if target is None:
                    # Just log — no pass/fail
                    status = "   (noting)"
                    n_log += 1
                    print(f"  {method:<12} {dataset:<12} {metric:<14} {'   —   ':>8} "
                          f"{got:>8.3f}  {'N/A':>7}  {status}")
                else:
                    rel_err = abs(got - target) / target
                    rel_str = f"{rel_err:+.1%}"
                    if rel_err <= THRESHOLD_PASS:
                        status = "✅ PASS"
                        n_pass += 1
                    else:
                        status = "❌ INVESTIGATE"
                        n_fail += 1
                    print(f"  {method:<12} {dataset:<12} {metric:<14} {target:>8.3f} "
                          f"{got:>8.3f}  {rel_str:>7}  {status}")

    # HiFi-Net special check: must produce non-trivial F1 on Columbia
    hifi_col = results.get("HiFi-Net", {}).get("Columbia", {})
    hifi_f1  = hifi_col.get("optimal_f1", float("nan"))
    print()
    print("  HiFi-Net sanity check (must be non-trivial on training domain):")
    if math.isnan(hifi_f1):
        print("    ⚠  HiFi-Net/Columbia not in results — dataset not run")
    elif hifi_f1 < HIFINET_MIN_F1:
        print(f"    ❌ HiFi-Net optimal_f1={hifi_f1:.4f} on Columbia < {HIFINET_MIN_F1} → CODE BUG")
        n_fail += 1
    else:
        print(f"    ✅ HiFi-Net optimal_f1={hifi_f1:.4f} on Columbia — domain shift only")
        n_pass += 1

    print()
    print(f"  Summary: {n_pass} PASS, {n_fail} FAIL, {n_log} log-only")
    print("=" * 72)
    print()

    if n_fail > 0:
        print("  NEXT STEPS for failing methods:")
        for method, datasets in TARGETS.items():
            for dataset, metrics in datasets.items():
                ds_results = results.get(method, {}).get(dataset, {})
                for metric, target in metrics.items():
                    if target is None:
                        continue
                    got = ds_results.get(metric, float("nan"))
                    if not math.isnan(got):
                        rel_err = abs(got - target) / target
                        if rel_err > THRESHOLD_PASS:
                            print(f"    {method}/{dataset}/{metric}: "
                                  f"ours={got:.3f} vs paper={target:.3f} "
                                  f"({rel_err:+.1%} relative)")
        print()
        print("  Common fixes to try:")
        print("    - Check RGB vs BGR channel order")
        print("    - Verify normalization (mean/std must match training)")
        print("    - Check image resize / padding policy")
        print("    - Verify checkpoint key mapping (module. prefix stripping)")
        print("    - For image-level AUC: use max(pred_mask) as image score")
        print()

    return n_fail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="results/E_validation/results.json",
        help="Path to validation results.json",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        print("Run: python scripts/run_exp1.py --config configs/exp_validation.yaml")
        sys.exit(1)

    results = load_results(results_path)
    n_fail = compare(results)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
