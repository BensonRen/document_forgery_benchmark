#!/usr/bin/env python3
"""
Document-specific and general image forensic methods on document datasets (zero-shot).
"""

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import (
    compute_all_pixel_metrics as _compute_all_pixel_metrics,
    save_results,
    save_results_csv,
)

# Reuse loaders from E1 + add DocTamper-specific loader
from run_exp1 import METHOD_LOADERS, load_dataset, _resolve_repo

# Per-method conda environments (see methods/README.md). Override with DOCFORGE_CONDA_ENVS.
CONDA_ENVS_DIR = os.environ.get("DOCFORGE_CONDA_ENVS", os.path.expanduser("~/.conda/envs"))


def load_doctamper_model(cfg):
    """Load the DocTamper DTD model (Qu et al., CVPR 2023).

    DTD uses a dual-stream architecture: VPH (visual) + Swin (frequency/DCT).
    Weights must be at methods/DocTamper/pths/:
      - dtd_doctamper.pth  (main checkpoint, saved from DataParallel)
      - swin_imagenet.pt   (Swin backbone pretrained)
      - vph_imagenet.pt    (VPH backbone pretrained)

    The DTD.__init__ loads swin_imagenet.pt and vph_imagenet.pt via
    torch.load() with bare filenames (relative to cwd), so we must
    os.chdir to the pths/ directory before instantiation.

    Input: any image (PNG/JPEG). Internally saved as JPEG to extract DCT.
    Output: sigmoid probability map (float32, 0-1, H×W).
    """
    import torch
    import torch.nn.functional as F
    import tempfile
    import pickle

    repo = Path(_resolve_repo(cfg))
    models_dir = repo / "models"
    pths_dir = repo / "pths"

    sys.path.insert(0, str(models_dir))
    sys.path.insert(0, str(repo))

    saved_cwd = os.getcwd()
    os.chdir(str(pths_dir))  # DTD.__init__ does torch.load('vph_imagenet.pt')

    from dtd import seg_dtd

    # Patch __main__ so torch.load can deserialize old-style pickled checkpoints
    # (swin_imagenet.pt and vph_imagenet.pt were saved with __main__.BasicLayer etc.)
    import __main__ as _main_mod
    import importlib as _importlib
    for _mod_name in ['swins', 'dtd', 'fph']:
        try:
            _mod = _importlib.import_module(_mod_name)
            for _attr in dir(_mod):
                if not _attr.startswith('_'):
                    setattr(_main_mod, _attr, getattr(_mod, _attr))
        except ImportError:
            pass

    # Patch timm.models.layers.drop → timm.models.layers (timm 1.0.x moved DropPath here)
    import timm.models.layers as _timm_layers
    import sys as _sys
    _timm_drop = type(_sys)('timm.models.layers.drop')
    _timm_drop.DropPath = _timm_layers.DropPath
    _sys.modules['timm.models.layers.drop'] = _timm_drop

    model = seg_dtd("", 2)
    checkpoint_name = cfg.get("weights", "dtd_doctamper.pth")
    # weights path is relative to pths_dir
    ckpt_path = pths_dir / Path(checkpoint_name).name
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Checkpoint was saved from nn.DataParallel wrapping seg_dtd.
    # Keys are 'module.model.*'; strip 'module.model.' to get bare keys for model.model,
    # then fall back to stripping just 'module.' for any other DataParallel-saved variants.
    state_dict = ckpt.get("state_dict", ckpt)
    new_sd = {}
    for k, v in state_dict.items():
        k2 = k.replace("module.model.", "").replace("module.", k) if k.startswith("module.model.") or k.startswith("module.") else k
        if k2.startswith("module."):
            k2 = k2[7:]
        new_sd[k2] = v
    missing, unexpected = model.model.load_state_dict(new_sd, strict=False)
    print(f"  [DocTamper] Loaded weights: {len(new_sd) - len(missing)} / {len(new_sd)} keys matched"
          f" (missing={len(missing)}, unexpected={len(unexpected)})")

    # Fix GELU compatibility: old checkpoints were pickled without 'approximate' attribute
    # (PyTorch < 1.12 GELU had no 'approximate'; pickle restores __dict__ without re-running __init__)
    for m in model.modules():
        if isinstance(m, torch.nn.GELU) and not hasattr(m, 'approximate'):
            m.approximate = 'none'
        # Fix DropPath: ensure scale_by_keep exists on unpickled instances
        if m.__class__.__name__ == 'DropPath' and not hasattr(m, 'scale_by_keep'):
            m.scale_by_keep = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    os.chdir(saved_cwd)

    # Load quantization table lookup used during training
    qt_table_path = repo / "qt_table.pk"  # file is at repo root, not models/ subdir
    if qt_table_path.exists():
        with open(qt_table_path, "rb") as f:
            pks_raw = pickle.load(f)
        pks = {k: torch.LongTensor(v) for k, v in pks_raw.items()}
    else:
        pks = None

    import torchvision.transforms as T
    toctsr = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    try:
        import jpegio
        has_jpegio = True
    except ImportError:
        has_jpegio = False
        print("  [WARN] jpegio not available; DocTamper DCT stream will use zeros")

    DEFAULT_Q = 90  # quality used for re-saving non-JPEG inputs

    def predict(img_path):
        from PIL import Image as PILImage
        img = PILImage.open(img_path).convert("RGB")
        ori_w, ori_h = img.size

        # DTD forward has H = W = int(L**0.5) which requires square feature maps.
        # Resize all images to 512×512 (model's expected resolution) before inference,
        # then resize the output mask back to original size.
        INFER_SIZE = 512
        img = img.resize((INFER_SIZE, INFER_SIZE), PILImage.LANCZOS)
        # After resize to 512×512, padding is not needed (512 % 64 == 0)

        # Save as JPEG to extract DCT coefficients (required by model)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img_gray = img.convert("L")
            img_gray.save(tmp_path, "JPEG", quality=DEFAULT_Q)

        try:
            if has_jpegio:
                jpg = jpegio.read(tmp_path)
                dct = torch.tensor(
                    np.clip(np.abs(jpg.coef_arrays[0].copy()), 0, 20),
                    dtype=torch.long,
                ).unsqueeze(0)
                q = DEFAULT_Q
                use_qtb = pks[q] if pks and q in pks else torch.zeros(64, dtype=torch.long)
            else:
                dct = torch.zeros(1, INFER_SIZE // 8, INFER_SIZE // 8, dtype=torch.long)
                use_qtb = torch.zeros(64, dtype=torch.long)
        finally:
            os.unlink(tmp_path)

        img_tensor = toctsr(img.convert("RGB")).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        dct = dct.to(device)
        qs = use_qtb.reshape(8, 8).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,8,8) → model expects (B,1,8,8)

        with torch.no_grad():
            pred = model(img_tensor, dct, qs)

        # pred shape: (1, 2, H, W) — class-1 probability is forgery score
        prob = F.softmax(pred, dim=1)[:, 1, :, :]  # (1, H, W)
        mask = prob.squeeze().cpu().numpy().astype(np.float32)
        if mask.shape != (ori_h, ori_w):
            mask = cv2.resize(mask, (ori_w, ori_h))
        return mask

    return predict


def _read_method_env(method_dir):
    """Read the conda env name from METHOD_ENV file in the method directory."""
    env_file = Path(method_dir) / "METHOD_ENV"
    if env_file.exists():
        return env_file.read_text().strip()
    return None


def _make_subprocess_worker_loader(worker_script, worker_args, env_name, cwd=None):
    """
    Generic loader that starts a persistent worker subprocess in a conda env.

    The worker reads '<img_path>\\t<out_npy_path>\\n' from stdin and writes
    'DONE\\n' or 'ERROR\\n' to stdout after processing each image.

    Returns a predict(img_path) -> np.ndarray function.
    """
    import subprocess
    import tempfile
    import threading

    # Resolve the conda env's Python directly to avoid inheriting the active venv.
    # When a venv is activated, VIRTUAL_ENV and the venv bin/ prepended to PATH
    # cause 'conda run' to pick up the venv's Python instead of the conda env's.
    import shutil
    conda_python = f"{CONDA_ENVS_DIR}/{env_name}/bin/python"
    if not os.path.exists(conda_python):
        # Fallback: try to find via conda info
        conda_python = shutil.which("python") or "python"

    cmd = [conda_python, worker_script] + [str(a) for a in worker_args]

    # Build a clean environment: inherit os.environ but strip venv activation vars
    # so the subprocess uses the conda env's site-packages.
    import copy
    clean_env = copy.copy(os.environ)
    clean_env.pop("VIRTUAL_ENV", None)
    clean_env.pop("PYTHONPATH", None)
    # Replace PATH: remove the venv bin/ prefix, keep the conda env bin/ first
    conda_bin = f"{CONDA_ENVS_DIR}/{env_name}/bin"
    path_parts = clean_env.get("PATH", "").split(os.pathsep)
    venv_bin = os.path.join(clean_env.get("VIRTUAL_ENV", ""), "bin")
    path_parts = [p for p in path_parts if p != venv_bin]
    if conda_bin not in path_parts:
        path_parts.insert(0, conda_bin)
    clean_env["PATH"] = os.pathsep.join(path_parts)
    # Limit threads aggressively to avoid RLIMIT_NPROC exhaustion
    for _tvar in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        clean_env[_tvar] = "2"
    clean_env["TOKENIZERS_PARALLELISM"] = "false"

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,   # let stderr pass through to terminal
        cwd=str(cwd) if cwd else None,
        env=clean_env,
    )

    # Wait for READY signal — drain all startup lines until we see "READY"
    # Some workers print diagnostic lines to stdout before signalling READY.
    import select
    ready_line = None
    while True:
        line = proc.stdout.readline()
        if not line:
            break  # EOF — process died
        decoded = line.decode().strip()
        if decoded == "READY":
            ready_line = "READY"
            break
        # Startup chatter — print to stderr so it's visible but doesn't break detection
        import sys as _sys
        _sys.stderr.write("[WORKER STARTUP] " + decoded + "\n")
    if ready_line != "READY":
        raise RuntimeError(f"Worker did not signal READY (got {ready_line!r})")

    lock = threading.Lock()

    def predict(img_path):
        with lock:
            # Resolve to absolute path so the worker subprocess (different cwd) can find it
            img_path = os.path.abspath(img_path)
            out_fd, out_path = tempfile.mkstemp(suffix=".npy")
            os.close(out_fd)
            try:
                proc.stdin.write(f"{img_path}\t{out_path}\n".encode())
                proc.stdin.flush()
                status = proc.stdout.readline().decode().strip()
                if status != "DONE":
                    raise RuntimeError(f"Worker returned {status!r} for {img_path}")
                mask = np.load(out_path)
            finally:
                if os.path.exists(out_path):
                    os.unlink(out_path)
        return mask

    def shutdown():
        """Terminate the worker subprocess and free its GPU memory immediately.

        Call this once all datasets for a method are done.  Without an explicit
        shutdown the worker's proc reference is only held by the atexit closure,
        so the subprocess — and its GPU VRAM — stays alive until the entire
        run_exp2.py process exits, creating ghost GPU processes.
        """
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

    # Fallback safety net: also register at atexit in case shutdown() is never
    # called (e.g. early exit due to exception).
    import atexit
    atexit.register(shutdown)

    return predict, shutdown


def load_ascformer_model(cfg):
    """Load ASCFormer via persistent subprocess in the 'ascformer' conda env."""
    repo = Path(_resolve_repo(cfg)) / "ASCFormer"
    env_name = _read_method_env(repo.parent) or "ascformer"
    config_path = repo / "configs" / "ascformer" / "ascformer_rtm.py"
    weights_path = repo / "work_dirs" / "ascformer_rtm" / "ascformer_model.pth"
    worker_script = str(repo / "worker.py")
    device = cfg.get("device", "cuda:0")

    return _make_subprocess_worker_loader(
        worker_script=worker_script,
        worker_args=[str(config_path), str(weights_path), "--device", device],
        env_name=env_name,
        cwd=repo,
    )


def load_adcdnet_model(cfg):
    """Load ADCD-Net via persistent subprocess in the 'adcdnet' conda env."""
    repo = Path(_resolve_repo(cfg))
    env_name = _read_method_env(repo) or "adcdnet"
    weights_dir = repo / "weights" / "ADCD-Net_exp_data"
    worker_script = str(repo / "worker.py")
    device = cfg.get("device", "cuda:0")

    return _make_subprocess_worker_loader(
        worker_script=worker_script,
        worker_args=[str(weights_dir), "--device", device],
        env_name=env_name,
        cwd=repo,
    )


def load_forensichub_model(cfg, model_name):
    """Load a ForensicHub document model (DTD, FFDN, CAFTB-Net, TIFDM) via persistent subprocess.

    model_name: 'dtd' | 'ffdn' | 'caftb' | 'tifdm'
    Checkpoints in methods/ForensicHub/checkpoints_doc/Doc_Bench/:
      - dtd-9.pth, ffdn_ckpt-8.pth, caftb-9.pth, tifdm-9.pth
    Env: dtd_ffdn (Python 3.10, numpy 1.24.4, torch 2.7.1+cu128, jpegio)
    """
    worker_dir = Path(__file__).parent.parent / "methods" / "ForensicHub"
    ckpt_dir = worker_dir / "checkpoints_doc" / "Doc_Bench"

    ckpt_map = {
        "dtd": "dtd-9.pth",
        "ffdn": "ffdn_ckpt-8.pth",
        "caftb": "caftb-9.pth",
        "tifdm": "tifdm-9.pth",
    }
    ckpt_path = ckpt_dir / ckpt_map[model_name]

    worker_script = str(worker_dir / "worker.py")
    device = cfg.get("device", "cuda:0")

    return _make_subprocess_worker_loader(
        worker_script=worker_script,
        worker_args=[model_name, str(ckpt_path), "--device", device],
        env_name="dtd_ffdn",
        cwd=worker_dir,
    )


def load_ffdn_model(cfg):
    """Load ForensicHub FFDN (Frequency Feature Decomposition Network, ECCV 2024)."""
    return load_forensichub_model(cfg, "ffdn")


def load_dtd_fh_model(cfg):
    """Load ForensicHub DTD (Dual-stream Text Detector, CVPR 2023)."""
    return load_forensichub_model(cfg, "dtd")


def load_caftb_model(cfg):
    """Load ForensicHub CAFTB-Net (Cross-Attention Two-Branch, TOMM 2024)."""
    return load_forensichub_model(cfg, "caftb")


def load_tifdm_model(cfg):
    """Load ForensicHub TIFDM (Robust Text Image Tampering Localization, TCE 2024)."""
    return load_forensichub_model(cfg, "tifdm")


# Register DocTamper loader
METHOD_LOADERS["DocTamper"] = load_doctamper_model
METHOD_LOADERS["ASCFormer"] = load_ascformer_model
METHOD_LOADERS["ADCDNet"] = load_adcdnet_model
METHOD_LOADERS["FFDN"] = load_ffdn_model
METHOD_LOADERS["DTD_FH"] = load_dtd_fh_model
METHOD_LOADERS["CAFTB"] = load_caftb_model
METHOD_LOADERS["TIFDM"] = load_tifdm_model










def run_experiment(config_path: str, method_filter: str = None, dataset_filter: list = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from existing results
    results_path = output_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"[INFO] Resuming from existing results: {[k for k in all_results if not k.startswith('_')]}")
    else:
        all_results = {}

    for method_cfg in config["methods"]:
        method_name = method_cfg["name"]
        if method_filter and method_name != method_filter:
            continue
        is_doc_specific = method_cfg.get("is_document_specific", False)
        tag = " [DOC-SPECIFIC]" if is_doc_specific else " [GENERAL]"
        print(f"\n{'='*60}")
        print(f"Method: {method_name}{tag}")
        print(f"{'='*60}")

        loader = METHOD_LOADERS.get(method_name)
        if loader is None:
            print(f"  [SKIP] No loader for {method_name}")
            continue

        try:
            result = loader(method_cfg)
            # Subprocess-based loaders return (predict, shutdown); in-process
            # loaders (E1 methods, DocTamper) return just predict.
            if isinstance(result, tuple):
                predict, shutdown = result
            else:
                predict, shutdown = result, None
        except Exception as e:
            print(f"  [ERROR] Failed to load {method_name}: {e}")
            continue

        # Pre-populate with already-computed results to enable per-dataset skip
        method_results = dict(all_results.get(method_name, {}))
        per_img_dir = output_dir / "per_image"
        per_img_dir.mkdir(exist_ok=True)

        for ds_cfg in config["datasets"]:
            ds_name = ds_cfg["name"]
            if dataset_filter and ds_name not in dataset_filter:
                continue
            print(f"\n  Dataset: {ds_name}")

            # Skip already-computed (method, dataset) pairs — but only if the
            # per-image CSV is also present and has the current column format
            # (includes max_pred_score). If the CSV is old-format or missing,
            # re-run to regenerate it with updated columns.
            _csv_check = per_img_dir / f"{method_name}_{ds_name}.csv"
            _csv_current = False
            if _csv_check.exists():
                with open(_csv_check, newline="") as _cf:
                    _csv_current = "max_pred_score" in (_cf.readline())
            if ds_name in method_results and _csv_current:
                print(f"  [SKIP] {method_name}/{ds_name} already computed — skipping")
                continue

            pairs = load_dataset(ds_cfg)
            if not pairs:
                continue

            metric_accum = {"pixel_f1": [], "pixel_iou": [], "pixel_auc": [], "optimal_f1": []}
            per_image_rows = []

            for _img_idx, (img_path, mask_path) in enumerate(tqdm(pairs, desc=f"  {method_name}/{ds_name}")):
                try:
                    pred = predict(img_path)
                    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    if pred.shape != gt.shape:
                        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

                    # Cap evaluation resolution to 2MP to keep metrics tractable on large images
                    _h, _w = gt.shape[:2]
                    if _h * _w > 2_000_000:
                        _scale = (2_000_000 / (_h * _w)) ** 0.5
                        _nh, _nw = max(1, int(_h * _scale)), max(1, int(_w * _scale))
                        gt = cv2.resize(gt, (_nw, _nh), interpolation=cv2.INTER_AREA)
                        pred = cv2.resize(pred, (_nw, _nh), interpolation=cv2.INTER_AREA)

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
                    # Periodic GC to prevent memory fragmentation buildup
                    if _img_idx % 100 == 99:
                        gc.collect()
                except Exception as e:
                    print(f"    [ERROR] {img_path}: {e}")

            if any(metric_accum.values()):
                metrics = {k: float(np.mean(v)) if v else float("nan")
                           for k, v in metric_accum.items()}
                metrics["is_document_specific"] = is_doc_specific
                method_results[ds_name] = metrics
                print(f"  Results: {json.dumps(metrics, indent=4)}")

                csv_path = per_img_dir / f"{method_name}_{ds_name}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["image", "pixel_f1", "pixel_iou", "pixel_auc", "optimal_f1",
                                       "max_pred_score", "gt_tamper_ratio"]
                    )
                    writer.writeheader()
                    writer.writerows(per_image_rows)
                print(f"  Per-image CSV: {csv_path}")

                # Per-dataset checkpoint save (crash-safe)
                if results_path.exists():
                    with open(results_path) as _ckpt_f:
                        _ckpt = json.load(_ckpt_f)
                else:
                    _ckpt = {}
                _ckpt[method_name] = method_results
                save_results(_ckpt, str(results_path))

            gc.collect()

        # Merge-safe save: re-read current JSON to avoid overwriting concurrent method runs
        if results_path.exists():
            with open(results_path) as _f:
                _current = json.load(_f)
        else:
            _current = {}
        _current[method_name] = method_results
        all_results = _current
        # Save checkpoint after each method
        save_results(all_results, str(results_path))

        # Explicitly shut down the worker subprocess to free GPU VRAM immediately.
        # Without this, the proc reference is only held by the atexit closure and
        # the worker (with its full model weights) stays alive as a ghost process
        # for the entire duration of run_exp2.py, wasting several GB of VRAM per method.
        if callable(shutdown):
            print(f"  [INFO] Shutting down {method_name} worker to free GPU memory.")
            shutdown()
        save_results_csv(all_results, str(output_dir / "results.csv"), experiment="E2")

    # Compute domain gap summary
    summary = {"domain_gap": {}}
    doc_method = "DocTamper"
    if doc_method in all_results:
        for ds_name in all_results[doc_method]:
            doc_f1 = all_results[doc_method][ds_name].get("pixel_f1", 0)
            general_f1s = []
            for m, res in all_results.items():
                if m != doc_method and ds_name in res:
                    general_f1s.append(res[ds_name].get("pixel_f1", 0))
            if general_f1s:
                avg_general = np.mean(general_f1s)
                best_general = max(general_f1s)
                summary["domain_gap"][ds_name] = {
                    "doc_specific_f1": doc_f1,
                    "avg_general_f1": float(avg_general),
                    "best_general_f1": float(best_general),
                    "gap_vs_avg": float(doc_f1 - avg_general),
                    "gap_vs_best": float(doc_f1 - best_general),
                }

    all_results["_summary"] = summary
    results_path = output_dir / "results.json"
    save_results(all_results, str(results_path))
    save_results_csv(all_results, str(output_dir / "results.csv"), experiment="E2")
    print(f"\nAll E2 results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2: Doc-specific vs general methods")
    parser.add_argument("--config", default="configs/exp2_doc_specific.yaml")
    parser.add_argument("--device", default=None,
                        help="CUDA device (e.g. 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument("--method", default=None,
                        help="Run only this method. Useful for parallel per-GPU execution.")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated list of dataset names to run (e.g. 'MixTamper,FSTS-1.5k'). "
                             "Useful for splitting datasets across GPUs.")
    args = parser.parse_args()

    if args.device is not None:
        if args.device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
        elif args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ds_filter = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    run_experiment(args.config, method_filter=args.method, dataset_filter=ds_filter)
