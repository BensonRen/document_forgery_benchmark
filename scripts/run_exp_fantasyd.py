#!/usr/bin/env python3
"""
Run all forensics methods on FantasyID text-only dataset.

Saves results into:
  - results/E1/results.json  (general methods: TruFor, MVSS-Net, PSCC-Net, IML-ViT)
  - results/E2/results.json  (all methods including DocTamper, ASCFormer, ADCDNet)

Merge-safe: re-reads JSON before each write to preserve any concurrent changes.
Skips already-computed method+dataset combinations.

Usage:
    python scripts/run_exp_fantasyd.py [--gpu N]
"""

import argparse
import csv
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Resolve paths ────────────────────────────────────────────────────
_EXP_ROOT = Path(__file__).resolve().parent.parent  # repository root
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import (
    compute_all_pixel_metrics as _compute_all_pixel_metrics,
    save_results,
    save_results_csv,
)
from run_exp1 import (
    METHOD_LOADERS,
    load_dataset,
    _cleanup_method_imports,
    load_trufor,
    load_mvssnet,
    load_psccnet,
    load_imlvit,
)
from run_exp2 import (
    load_doctamper_model,
    load_ascformer_model,
    load_adcdnet_model,
    load_ffdn_model,
    load_dtd_fh_model,
    load_caftb_model,
    load_tifdm_model,
    _make_subprocess_worker_loader,
)

# ── Dataset config ───────────────────────────────────────────────────
FANTASYD_CFG = {
    "name": "FantasyID",
    "root": "data/FantasyID/text_only",
    "image_dir": "images",
    "mask_dir": "masks",
    "format": "jpg",        # images are jpg (symlinked)
    "mask_format": "png",   # masks are png
}

# ── Method configs (verbatim from exp1/exp2 YAML) ───────────────────
# General methods (E1)
_GENERAL_METHOD_CFGS = [
    {
        "name": "TruFor",
        "repo": "methods/TruFor",
        "framework": "pytorch",
        "task": "localization",
        "weights": "TruFor_train_test/pretrained_models/weights/trufor.pth.tar",
        "input_size": [512, 512],
        "is_document_specific": False,
    },
    {
        "name": "MVSS-Net",
        "repo": "methods/MVSS-Net",
        "framework": "pytorch",
        "task": "localization",
        "weights": "ckpt/mvssnet_model/mvssnet_casia.pt",
        "input_size": [512, 512],
        "is_document_specific": False,
    },
    {
        "name": "PSCC-Net",
        "repo": "methods/PSCC-Net",
        "framework": "pytorch",
        "task": "localization",
        "weights": "checkpoints/HRNet.pth",
        "input_size": [512, 512],
        "is_document_specific": False,
    },
    {
        "name": "IML-ViT",
        "repo": "methods/IML-ViT",
        "framework": "pytorch",
        "task": "localization",
        "weights": "checkpoints/iml-vit_checkpoint.pth",
        "input_size": [1024, 1024],
        "is_document_specific": False,
    },
    {
        "name": "ManTraNet",
        "repo": "methods/ManTraNet",
        "framework": "keras",
        "task": "localization",
        "weights": "pretrained_weights/ManTraNet_Ptrain4.h5",
        "pretrain_index": 4,
        "is_document_specific": False,
    },
    {
        "name": "CAT-Net",
        "repo": "methods/CAT-Net",
        "framework": "pytorch",
        "task": "localization",
        "weights": "output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar",
        "input_size": [512, 512],
        "is_document_specific": False,
    },
    {
        "name": "SAFIRE",
        "repo": "methods/SAFIRE",
        "framework": "pytorch",
        "task": "localization",
        "weights": "safire.pth",
        "is_document_specific": False,
    },
]

# Doc-specific methods (E2 only)
_DOC_SPECIFIC_METHOD_CFGS = [
    {
        "name": "DocTamper",
        "repo": "methods/DocTamper",
        "framework": "pytorch",
        "task": "localization",
        "weights": "dtd_doctamper.pth",
        "input_size": [512, 512],
        "is_document_specific": True,
    },
    {
        "name": "ASCFormer",
        "repo": "methods/RTM",
        "framework": "pytorch",
        "task": "localization",
        "input_size": [512, 512],
        "is_document_specific": True,
        "conda_env": "ascformer",
    },
    {
        "name": "ADCDNet",
        "repo": "methods/ADCD-Net",
        "framework": "pytorch",
        "task": "localization",
        "input_size": [512, 512],
        "is_document_specific": True,
        "conda_env": "adcdnet",
    },
    {
        "name": "FFDN",
        "repo": "methods/ForensicHub",
        "framework": "pytorch",
        "task": "localization",
        "input_size": [512, 512],
        "is_document_specific": True,
        "conda_env": "dtd_ffdn",
    },
    {
        "name": "DTD_FH",
        "repo": "methods/ForensicHub",
        "framework": "pytorch",
        "task": "localization",
        "input_size": [512, 512],
        "is_document_specific": True,
        "conda_env": "dtd_ffdn",
    },
    {
        "name": "CAFTB",
        "repo": "methods/ForensicHub",
        "framework": "pytorch",
        "task": "localization",
        "input_size": [512, 512],
        "is_document_specific": True,
        "conda_env": "dtd_ffdn",
    },
    {
        "name": "TIFDM",
        "repo": "methods/ForensicHub",
        "framework": "pytorch",
        "task": "localization",
        "input_size": [512, 512],
        "is_document_specific": True,
        "conda_env": "dtd_ffdn",
    },
]

# All methods in run order (general first, then doc-specific)
ALL_METHOD_CFGS = _GENERAL_METHOD_CFGS + _DOC_SPECIFIC_METHOD_CFGS

# Register all loaders (run_exp1.METHOD_LOADERS already has general ones)
METHOD_LOADERS["DocTamper"] = load_doctamper_model
METHOD_LOADERS["ASCFormer"] = load_ascformer_model
METHOD_LOADERS["ADCDNet"] = load_adcdnet_model
METHOD_LOADERS["FFDN"] = load_ffdn_model
METHOD_LOADERS["DTD_FH"] = load_dtd_fh_model
METHOD_LOADERS["CAFTB"] = load_caftb_model
METHOD_LOADERS["TIFDM"] = load_tifdm_model

# Set of general method names (saved to E1 as well as E2)
_GENERAL_METHODS = {cfg["name"] for cfg in _GENERAL_METHOD_CFGS}

# Cap inference input to this pixel count on the long edge to prevent OOM.
# FantasyID images are up to 3230x2022; most methods were trained on <=1024px.
MAX_INFERENCE_DIM = 1500


def _merge_save_results(results_path: Path, method_name: str, ds_results: dict,
                        experiment: str):
    """Re-read JSON, merge new results for method_name, write back."""
    if results_path.exists():
        with open(results_path) as f:
            current = json.load(f)
    else:
        current = {}

    # Merge: preserve existing datasets for this method, add/update new ones
    existing = current.get(method_name, {})
    existing.update(ds_results)
    current[method_name] = existing

    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(current, str(results_path))
    save_results_csv(current, str(results_path.parent / "results.csv"), experiment=experiment)
    return current


def _evaluate_method_on_dataset(predict, method_name: str, ds_cfg: dict,
                                 per_img_dir: Path):
    """Run predict over all image/mask pairs, return metrics dict or None."""
    # Resolve dataset root relative to EXP_ROOT
    ds_cfg_abs = dict(ds_cfg)
    root = Path(ds_cfg["root"])
    if not root.is_absolute():
        ds_cfg_abs["root"] = str(_EXP_ROOT / root)

    pairs = load_dataset(ds_cfg_abs)
    if not pairs:
        print(f"    [WARN] No image/mask pairs found for {ds_cfg['name']}")
        return None

    ds_name = ds_cfg["name"]
    metric_accum = {"pixel_f1": [], "pixel_iou": [], "pixel_auc": [], "optimal_f1": []}
    per_image_rows = []

    for _img_idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"  {method_name}/{ds_name}")):
        _tmp_infer_path = None
        try:
            # Pre-resize large images before sending to model workers to prevent OOM.
            _chk = cv2.imread(img_path)
            if _chk is not None:
                _ih, _iw = _chk.shape[:2]
                if max(_ih, _iw) > MAX_INFERENCE_DIM:
                    _scale = MAX_INFERENCE_DIM / max(_ih, _iw)
                    _nh, _nw = max(1, int(_ih * _scale)), max(1, int(_iw * _scale))
                    _chk = cv2.resize(_chk, (_nw, _nh), interpolation=cv2.INTER_AREA)
                    _fd, _tmp_infer_path = tempfile.mkstemp(suffix=Path(img_path).suffix)
                    os.close(_fd)
                    cv2.imwrite(_tmp_infer_path, _chk)
                    _infer_path = _tmp_infer_path
                else:
                    _infer_path = img_path
                del _chk
            else:
                _infer_path = img_path
            pred = predict(_infer_path)
            if _tmp_infer_path is not None:
                try:
                    os.unlink(_tmp_infer_path)
                except OSError:
                    pass
                _tmp_infer_path = None
            gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                print(f"    [WARN] Cannot read mask {mask_path}")
                continue
            gt = gt.astype(np.float32) / 255.0
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

            # Cap evaluation resolution to 2MP to keep metrics tractable on large images
            _h, _w = gt.shape[:2]
            if _h * _w > 2_000_000:
                _scale = (2_000_000 / (_h * _w)) ** 0.5
                _nh, _nw = max(1, int(_h * _scale)), max(1, int(_w * _scale))
                gt = cv2.resize(gt, (_nw, _nh), interpolation=cv2.INTER_AREA)
                pred = cv2.resize(pred, (_nw, _nh), interpolation=cv2.INTER_AREA)

            # Compute all four pixel metrics for this image (see evaluate.py).
            m = _compute_all_pixel_metrics(pred, gt)
            f1, iou, auc, opt_f1 = m["pixel_f1"], m["pixel_iou"], m["pixel_auc"], m["optimal_f1"]
            max_score = float(pred.max())
            gt_tamper_ratio = float(gt.mean())

            if not np.isnan(f1):
                metric_accum["pixel_f1"].append(f1)
            if not np.isnan(iou):
                metric_accum["pixel_iou"].append(iou)
            if not np.isnan(auc):
                metric_accum["pixel_auc"].append(auc)
            if not np.isnan(opt_f1):
                metric_accum["optimal_f1"].append(opt_f1)

            per_image_rows.append({
                "image": Path(img_path).name,
                "pixel_f1": f1,
                "pixel_iou": iou,
                "pixel_auc": auc,
                "optimal_f1": opt_f1,
                "max_pred_score": max_score,
                "gt_tamper_ratio": gt_tamper_ratio,
            })
            del pred, gt
            if _img_idx % 100 == 99:
                gc.collect()
        except Exception as e:
            print(f"    [ERROR] {img_path}: {e}")
        finally:
            if _tmp_infer_path is not None:
                try:
                    os.unlink(_tmp_infer_path)
                except OSError:
                    pass
                _tmp_infer_path = None

    if not any(metric_accum.values()):
        return None

    metrics = {
        k: float(np.mean(v)) if v else float("nan")
        for k, v in metric_accum.items()
    }

    # Save per-image CSV
    csv_path = per_img_dir / f"{method_name}_{ds_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "pixel_f1", "pixel_iou", "pixel_auc", "optimal_f1",
                           "max_pred_score", "gt_tamper_ratio"]
        )
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"    Per-image CSV: {csv_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run forensics methods on FantasyID text-only dataset"
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (sets CUDA_VISIBLE_DEVICES). "
                             "If omitted, respects existing CUDA_VISIBLE_DEVICES env var; "
                             "falls back to 0 only if neither is set.")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Limit to these method names. Default: all.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if method+dataset already in results.")
    args = parser.parse_args()

    # Set GPU before any torch import.
    # Priority: --gpu flag > existing CUDA_VISIBLE_DEVICES env var > default 0
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[INFO] Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # Change to repository root so relative paths (data/, results/) resolve correctly
    os.chdir(str(_EXP_ROOT))
    print(f"[INFO] Working directory: {_EXP_ROOT}")

    e1_results_path = _EXP_ROOT / "results" / "E1" / "results.json"
    e2_results_path = _EXP_ROOT / "results" / "E2" / "results.json"
    e1_per_img_dir = _EXP_ROOT / "results" / "E1" / "per_image"
    e2_per_img_dir = _EXP_ROOT / "results" / "E2" / "per_image"
    e1_per_img_dir.mkdir(parents=True, exist_ok=True)
    e2_per_img_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results for skip-check
    def _load_json(path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    methods_filter = set(args.methods) if args.methods else None
    ds_name = FANTASYD_CFG["name"]

    for method_cfg in ALL_METHOD_CFGS:
        method_name = method_cfg["name"]
        if methods_filter and method_name not in methods_filter:
            continue

        is_general = method_name in _GENERAL_METHODS
        tag = " [GENERAL]" if is_general else " [DOC-SPECIFIC]"
        print(f"\n{'='*60}")
        print(f"Method: {method_name}{tag}")
        print(f"{'='*60}")

        # Check skip: general methods save to E1, doc-specific save to E2
        if not args.force:
            e1_existing = _load_json(e1_results_path)
            e2_existing = _load_json(e2_results_path)
            _canonical = e1_existing if is_general else e2_existing
            if ds_name in _canonical.get(method_name, {}):
                # Also verify CSV has max_pred_score (old CSVs need regeneration)
                _csv_dir = e1_per_img_dir if is_general else e2_per_img_dir
                _csv_check = _csv_dir / f"{method_name}_{ds_name}.csv"
                _csv_current = False
                if _csv_check.exists():
                    with open(_csv_check, newline="") as _cf:
                        _csv_current = "max_pred_score" in (_cf.readline())
                if _csv_current:
                    print(f"  [SKIP] {method_name}/{ds_name} already computed — skipping")
                    continue
                print(f"  [RE-RUN] {method_name}/{ds_name} — CSV missing max_pred_score, regenerating")

        loader = METHOD_LOADERS.get(method_name)
        if loader is None:
            print(f"  [SKIP] No loader registered for {method_name}")
            continue

        _cleanup_method_imports()

        try:
            result = loader(method_cfg)
            # Subprocess-based loaders return (predict, shutdown); in-process loaders
            # return just predict. Handle both.
            if isinstance(result, tuple):
                predict, shutdown = result
            else:
                predict, shutdown = result, None
        except Exception as e:
            print(f"  [ERROR] Failed to load {method_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Evaluate on FantasyID
        per_img_dir = e1_per_img_dir if is_general else e2_per_img_dir
        metrics = _evaluate_method_on_dataset(predict, method_name, FANTASYD_CFG, per_img_dir)

        if metrics is None:
            print(f"  [WARN] No metrics computed for {method_name}")
            gc.collect()
            continue

        metrics["is_document_specific"] = method_cfg.get("is_document_specific", False)
        print(f"  Results for {ds_name}: {json.dumps(metrics, indent=4)}")

        # Merge-safe save to E2 (all methods)
        _merge_save_results(e2_results_path, method_name, {ds_name: metrics}, experiment="E2")
        print(f"  [SAVED] E2: {e2_results_path}")

        # Also save to E1 for general methods
        if is_general:
            # E1 metrics don't include is_document_specific flag
            e1_metrics = {k: v for k, v in metrics.items() if k != "is_document_specific"}
            _merge_save_results(e1_results_path, method_name, {ds_name: e1_metrics},
                                experiment="E1")
            print(f"  [SAVED] E1: {e1_results_path}")

        if callable(shutdown):
            print(f"  [INFO] Shutting down {method_name} worker to free GPU memory.")
            shutdown()

        gc.collect()

    print(f"\n[DONE] FantasyID evaluation complete.")
    print(f"  E1 results: {e1_results_path}")
    print(f"  E2 results: {e2_results_path}")


if __name__ == "__main__":
    main()
