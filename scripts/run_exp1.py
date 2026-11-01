#!/usr/bin/env python3
"""
General image forensic methods on document datasets (zero-shot).

Evaluates TruFor, ManTraNet, MVSS-Net, CAT-Net, PSCC-Net, IML-ViT, SAFIRE
on all 8 benchmark datasets. Measures pixel-F1, IoU, pixel-AUC, optimal-F1.
"""

import argparse
import csv
import fcntl
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

# Per-method conda environments (see methods/README.md). Override with DOCFORGE_CONDA_ENVS.
CONDA_ENVS_DIR = os.environ.get("DOCFORGE_CONDA_ENVS", os.path.expanduser("~/.conda/envs"))

# ── Resolve repo path relative to experiment root ────────────────────
_EXP_ROOT = Path(__file__).resolve().parent.parent  # repository root

# Keep track of the original sys.path so we can clean up between method loads
_ORIGINAL_SYS_PATH = list(sys.path)
_ORIGINAL_MODULES = set(sys.modules.keys())


def _cleanup_method_imports():
    """Remove method-specific entries from sys.path and sys.modules
    to prevent import collisions between methods (e.g. all have 'models/')."""
    methods_dir = str(_EXP_ROOT / "methods")
    sys.path[:] = [p for p in sys.path if not p.startswith(methods_dir)]
    # Restore original path entries
    for p in _ORIGINAL_SYS_PATH:
        if p not in sys.path:
            sys.path.append(p)
    # Remove method-specific modules
    for mod_name in list(sys.modules.keys()):
        if mod_name not in _ORIGINAL_MODULES:
            mod = sys.modules[mod_name]
            mod_file = getattr(mod, '__file__', '') or ''
            if methods_dir in mod_file:
                del sys.modules[mod_name]


def _resolve_repo(cfg):
    """Return absolute repo path from config entry (may be relative)."""
    repo = Path(cfg["repo"])
    if not repo.is_absolute():
        repo = _EXP_ROOT / repo
    return str(repo.resolve())


def _resolve_weights(cfg):
    """Return absolute weights path: try repo-relative, then raw."""
    w = Path(cfg["weights"])
    if w.is_absolute():
        return str(w)
    repo = Path(_resolve_repo(cfg))
    candidate = repo / w
    if candidate.exists():
        return str(candidate)
    return str(w)


# ── Method wrappers ──────────────────────────────────────────────────
# Each wrapper loads a model and returns a predict function:
#   predict(img_path: str) -> np.ndarray  (float32, 0-1, shape H x W)


def load_trufor(cfg):
    """Load TruFor (GRIP-UNINA, 2023).

    Uses the test_docker inference path which builds a confcmx model
    from the bundled config YAML and loads checkpoint['state_dict'].
    The localization map is obtained via F.softmax(pred, dim=0)[1].
    """
    import torch
    from torch.nn import functional as F
    from PIL import Image

    repo = _resolve_repo(cfg)
    src_dir = os.path.join(repo, "test_docker", "src")
    sys.path.insert(0, src_dir)

    # Build config from the bundled YAML
    saved_cwd = os.getcwd()
    os.chdir(src_dir)  # config.update_config reads trufor.yaml relative to cwd
    try:
        from config import _C as trufor_config
        trufor_config.defrost()
        trufor_config.merge_from_file("trufor.yaml")
        trufor_config.freeze()
    finally:
        os.chdir(saved_cwd)

    # Build model
    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    model = confcmx(cfg=trufor_config)

    # Load weights
    weights_path = _resolve_weights(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    MAX_DIM = 4096  # cap resolution to avoid OOM on very large images

    def predict(img_path):
        img_RGB = np.array(Image.open(img_path).convert("RGB"))
        # Resize if image exceeds MAX_DIM (prevents OOM on high-res captures)
        h, w = img_RGB.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img_RGB = cv2.resize(
                img_RGB, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        # TruFor data_core normalises as float / 256.0, channels-first
        tensor = torch.tensor(
            img_RGB.transpose(2, 0, 1), dtype=torch.float
        ).unsqueeze(0) / 256.0
        tensor = tensor.to(device)

        with torch.no_grad():
            pred, conf, det, npp = model(tensor)

        pred = torch.squeeze(pred, 0)
        pred = F.softmax(pred, dim=0)[1]  # class-1 probability
        return pred.cpu().numpy().astype(np.float32)

    return predict


def load_mantranet(cfg):
    """Load ManTraNet (Wu et al., 2019) via persistent subprocess in the 'mantranet' conda env.

    ManTraNet requires Keras 2.x (TF1-style custom layers) which is incompatible with
    the main venv's Keras 3.x.  We run inference in a dedicated 'mantranet' conda
    env (Python 3.8, TF 2.9 + bundled Keras 2.9) via a persistent worker subprocess.

    The worker (methods/ManTraNet/worker.py) applies the following compatibility shims:
      - keras.legacy.interfaces  -> no-op module
      - keras.engine.InputSpec   -> keras.layers.InputSpec
      - keras.layers.convolutional._Conv -> base_conv.Conv
      - CombinedConv2D._build_SRM_kernel patched to use tf.constant instead of
        K.variable, preventing Keras 2.9 from tracking the SRM kernel as a
        layer weight (which would cause a weight-count mismatch when loading .h5).
    """
    repo = Path(_resolve_repo(cfg))
    worker_script = str(repo / "worker.py")
    weights_dir = str(repo / "pretrained_weights")
    pretrain_index = cfg.get("pretrain_index", 4)

    return _make_subprocess_worker_loader(
        worker_script=worker_script,
        worker_args=["--weights-dir", weights_dir, "--pretrain-index", str(pretrain_index)],
        env_name="mantranet",
        cwd=str(repo),
    )


def load_mvssnet(cfg):
    """Load MVSS-Net (Dong et al., 2022).

    Uses get_mvss() to build the network, loads checkpoint state_dict,
    and runs inference_single from common/tools.py.
    """
    import torch

    repo = _resolve_repo(cfg)
    sys.path.insert(0, repo)

    from models.mvssnet import get_mvss
    from common.tools import inference_single

    model_path = _resolve_weights(cfg)
    model = get_mvss(
        backbone="resnet50",
        pretrained_base=True,
        nclass=1,
        sobel=True,
        constrain=True,
        n_input=3,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    resize = cfg.get("input_size", [512, 512])
    new_size = resize[0] if isinstance(resize, list) else resize

    def predict(img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        ori_h, ori_w = img.shape[:2]
        img_resized = cv2.resize(img, (new_size, new_size))
        # inference_single returns (seg_map_uint8, max_score)
        seg, _ = inference_single(img=img_resized, model=model, th=0)
        # seg is uint8 0-255 from ToPILImage, resize back to original
        seg = cv2.resize(seg, (ori_w, ori_h))
        return seg.astype(np.float32) / 255.0

    return predict


def _make_subprocess_worker_loader(worker_script, worker_args, env_name, cwd=None):
    """
    Generic loader that starts a persistent worker subprocess in a conda env.

    The worker reads '<img_path>\t<out_npy_path>\n' from stdin and writes
    'DONE\n' or 'ERROR\n' to stdout after processing each image.

    Returns a (predict, shutdown) tuple.
    """
    import subprocess
    import tempfile
    import threading
    import shutil
    import copy

    conda_python = f"{CONDA_ENVS_DIR}/{env_name}/bin/python"
    if not os.path.exists(conda_python):
        conda_python = shutil.which("python") or "python"

    cmd = [conda_python, worker_script] + [str(a) for a in worker_args]

    clean_env = copy.copy(os.environ)
    clean_env.pop("VIRTUAL_ENV", None)
    clean_env.pop("PYTHONPATH", None)
    conda_bin = f"{CONDA_ENVS_DIR}/{env_name}/bin"
    path_parts = clean_env.get("PATH", "").split(os.pathsep)
    venv_bin = os.path.join(clean_env.get("VIRTUAL_ENV", ""), "bin")
    path_parts = [p for p in path_parts if p != venv_bin]
    if conda_bin not in path_parts:
        path_parts.insert(0, conda_bin)
    clean_env["PATH"] = os.pathsep.join(path_parts)
    # Prepend conda env lib dir to LD_LIBRARY_PATH so TF/Keras can find
    # bundled CUDA 11 libraries (cudatoolkit=11.2, cudnn=8.1 from conda-forge).
    conda_lib = f"{CONDA_ENVS_DIR}/{env_name}/lib"
    ld_parts = clean_env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if conda_lib not in ld_parts:
        ld_parts.insert(0, conda_lib)
    clean_env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in ld_parts if p)
    for _tvar in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                  "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        clean_env[_tvar] = "2"
    clean_env["TOKENIZERS_PARALLELISM"] = "false"

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        cwd=str(cwd) if cwd else None,
        env=clean_env,
    )

    ready_line = None
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        decoded = line.decode().strip()
        if decoded == "READY":
            ready_line = "READY"
            break
        import sys as _sys
        _sys.stderr.write("[WORKER STARTUP] " + decoded + "\n")
    if ready_line != "READY":
        raise RuntimeError(f"Worker did not signal READY (got {ready_line!r})")

    lock = threading.Lock()

    def predict(img_path):
        with lock:
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

    import atexit
    atexit.register(shutdown)

    return predict, shutdown



def load_catnet(cfg):
    """Load CAT-Net v2 (Kwon et al., IJCV 2022) via persistent subprocess.

    Runs in the isolated 'cat' conda env (Python 3.8, PyTorch 2.x) to avoid
    import collisions with other methods.  The worker handles JPEG DCT
    extraction via jpegio + torch_dct, then runs network_CAT inference.

    Weights layout expected under the repo root:
        output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar
        pretrained_models/DCT_djpeg.pth.tar
        pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
    """
    repo = _resolve_repo(cfg)
    worker_script = os.path.join(repo, "worker.py")
    # weights_dir is the repo root — worker resolves sub-paths internally
    device = cfg.get("device", "cuda:0")

    return _make_subprocess_worker_loader(
        worker_script=worker_script,
        worker_args=[repo, "--device", device],
        env_name="cat",
        cwd=repo,
    )


def load_psccnet(cfg):
    """Load PSCC-Net (Liu et al., 2022).

    PSCC-Net has three sub-networks: FENet (HRNet backbone), SegNet
    (NLCDetection localization head), and ClsNet (DetectionHead).
    For localization we only need FENet + SegNet outputs.
    """
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    repo = _resolve_repo(cfg)
    sys.path.insert(0, repo)

    from models.seg_hrnet import get_seg_model
    from models.seg_hrnet_config import get_hrnet_cfg
    from models.NLCDetection import NLCDetection
    from utils.config import get_pscc_args

    args = get_pscc_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    # Build backbone
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg).to(device)
    SegNet = NLCDetection(args).to(device)

    if device_ids:
        FENet = nn.DataParallel(FENet, device_ids=device_ids)
        SegNet = nn.DataParallel(SegNet, device_ids=device_ids)

    # Load weights from checkpoint directory
    checkpoint_dir = os.path.dirname(_resolve_weights(cfg))
    if not checkpoint_dir:
        checkpoint_dir = os.path.join(repo, "checkpoint")

    def _load_weight(net, ckpt_dir, name):
        weight_path = os.path.join(ckpt_dir, f"{name}.pth")
        if not os.path.exists(weight_path):
            # Try the checkpoint/<Name>_checkpoint/ convention
            alt_dir = os.path.join(repo, "checkpoint", f"{name}_checkpoint")
            weight_path = os.path.join(alt_dir, f"{name}.pth")
        state_dict = torch.load(weight_path, map_location=device, weights_only=False)
        net.load_state_dict(state_dict)
        print(f"  [PSCC-Net] Loaded {name} from {weight_path}")

    _load_weight(FENet, checkpoint_dir, "HRNet")
    _load_weight(SegNet, checkpoint_dir, "NLCDetection")

    FENet.eval()
    SegNet.eval()

    def predict(img_path):
        import imageio
        image = imageio.imread(img_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        ori_h, ori_w = image.shape[:2]
        tensor = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = FENet(tensor)
            pred_mask = SegNet(feat)[0]
            pred_mask = F.interpolate(
                pred_mask,
                size=(tensor.size(2), tensor.size(3)),
                mode="bilinear",
                align_corners=True,
            )

        # NLCDetection already applies sigmoid internally; do not apply again
        mask = pred_mask.squeeze().cpu().numpy()
        if mask.ndim > 2:
            mask = mask[0]
        return mask.astype(np.float32)

    return predict


def load_imlvit(cfg):
    """Load IML-ViT (Ma et al., 2023).

    Builds the iml_vit_model, loads checkpoint, and runs inference.
    Images are padded/cropped to input_size with ImageNet normalisation.
    The model's forward() requires (x, masks, edge_masks) but during
    inference we pass dummy masks and extract the sigmoid prediction.
    """
    import torch
    from torch.nn import functional as F
    import albumentations as albu
    from albumentations.pytorch import ToTensorV2

    repo = _resolve_repo(cfg)
    sys.path.insert(0, repo)

    from iml_vit_model import iml_vit_model

    input_size = cfg.get("input_size", [1024, 1024])
    if isinstance(input_size, list):
        input_size = input_size[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = _resolve_weights(cfg)

    model = iml_vit_model(input_size=input_size)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Handle both raw state_dict and wrapped checkpoint
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Padding + normalisation transform (same as repo's 'pad' transform)
    pad_transform = albu.Compose([
        albu.PadIfNeeded(
            min_height=input_size,
            min_width=input_size,
            border_mode=0,
            value=0,
            position="top_left",
            mask_value=0,
        ),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albu.Crop(0, 0, input_size, input_size),
        ToTensorV2(transpose_mask=True),
    ])

    def predict(img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img.shape[:2]

        transformed = pad_transform(image=img, mask=np.zeros((ori_h, ori_w), dtype=np.uint8))
        tensor = transformed["image"].unsqueeze(0).float().to(device)

        # Dummy mask and edge_mask for forward pass
        dummy_mask = torch.zeros(1, 1, input_size, input_size).to(device)
        dummy_edge = torch.zeros(1, 1, input_size, input_size).to(device)

        with torch.no_grad():
            _, mask_pred, _ = model(tensor, dummy_mask, dummy_edge)

        # mask_pred is already sigmoid-ed in the model's forward
        pred = mask_pred.squeeze().cpu().numpy()
        # Crop back to original size (padding was top-left)
        pred = pred[:ori_h, :ori_w]
        return pred.astype(np.float32)

    return predict




def load_safire_model(cfg):
    """Load SAFIRE (Segment Any Forged Image Region, AAAI 2025) via persistent
    subprocess in the 'safire' conda env.

    SAFIRE extends SAM (ViT-B adaptor) and performs binary forgery localization
    by clustering SAM segment features with k-means (k=2).

    Weights required in the repo root:
      - safire.pth               (SAFIRE fine-tuned weights, Google Drive)
      - sam_vit_b_01ec64.pth    (SAM ViT-B base, Meta public download)
    """
    repo = Path(_resolve_repo(cfg))
    weights = Path(_resolve_weights(cfg))   # safire.pth
    sam_ckpt = repo / "sam_vit_b_01ec64.pth"
    worker_script = str(repo / "worker.py")
    env_name = "safire"

    predict, shutdown = _make_subprocess_worker_loader(
        worker_script,
        [
            "--weights", str(weights),
            "--sam_checkpoint", str(sam_ckpt),
            "--device", "cuda:0",
            "--points_per_side", "16",
            "--points_per_batch", "256",
        ],
        env_name=env_name,
        cwd=str(repo),
    )
    return predict, shutdown


METHOD_LOADERS = {
    "TruFor": load_trufor,
    "ManTraNet": load_mantranet,
    "MVSS-Net": load_mvssnet,
    "CAT-Net": load_catnet,
    "PSCC-Net": load_psccnet,
    "IML-ViT": load_imlvit,
    "SAFIRE": load_safire_model,
}


# ── Dataset loader ───────────────────────────────────────────────────

def load_dataset(ds_cfg):
    """Return list of (image_path, mask_path) pairs.

    Supports optional keys:
      format: image extension (default: png)
      mask_format: mask extension if different from format (default: same as format)
      max_samples: if set, randomly sample this many pairs (for large datasets)
      split_file: path to a text file listing image stems (one per line, no extension)
                  to restrict evaluation to an official train/test split. When set,
                  only images whose stem appears in the file are included. This is
                  critical for datasets like RealTextManipulation that provide
                  official split files — loading all images would include training
                  images and inflate F1 for methods trained on that data.
    """
    import random as _random
    root = Path(ds_cfg["root"])
    img_dir = root / ds_cfg.get("image_dir", "images")
    mask_dir = root / ds_cfg.get("mask_dir", "masks")
    fmt = ds_cfg.get("format", "png")
    mask_fmt = ds_cfg.get("mask_format", fmt)  # masks may use a different extension
    max_samples = ds_cfg.get("max_samples", None)
    split_file = ds_cfg.get("split_file", None)

    # Load official split filter if provided
    split_stems = None
    if split_file is not None:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = root / split_path
        if split_path.exists():
            with open(split_path) as _sf:
                split_stems = set(line.strip() for line in _sf if line.strip())
            print(f"  [INFO] Using split file {split_path} ({len(split_stems)} stems)")
        else:
            print(f"  [WARN] split_file not found: {split_path} — loading all images")

    pairs = []
    for img_path in sorted(img_dir.glob(f"*.{fmt}")):
        if split_stems is not None and img_path.stem not in split_stems:
            continue
        mask_path = mask_dir / f"{img_path.stem}.{mask_fmt}"
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))

    if not pairs:
        print(f"  [WARN] No image-mask pairs found in {root}")
        return pairs

    if max_samples is not None and len(pairs) > max_samples:
        _random.seed(42)
        pairs = _random.sample(pairs, max_samples)
        pairs = sorted(pairs)  # sort for reproducibility
        print(f"  [INFO] Sampled {max_samples} from {img_dir} (total: {len(pairs)} after sampling)")

    return pairs


# ── Main ─────────────────────────────────────────────────────────────

def run_experiment(config_path: str, method_filter: str = None, dataset_filter: list = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_cfg = config["experiment"]
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing results to allow resuming
    results_path = output_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"[INFO] Resuming from existing results: {list(all_results.keys())}")
    else:
        all_results = {}

    for method_cfg in config["methods"]:
        method_name = method_cfg["name"]
        if method_filter and method_name != method_filter:
            continue
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        # Load method
        loader = METHOD_LOADERS.get(method_name)
        if loader is None:
            print(f"  [SKIP] No loader for {method_name}")
            continue

        # Clean up sys.path/modules to prevent import collisions
        _cleanup_method_imports()

        try:
            result = loader(method_cfg)
            # Subprocess-based loaders return (predict, shutdown); in-process
            # loaders return just predict.
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

            # Streaming evaluation: compute metrics per image immediately
            # to avoid accumulating large mask arrays in RAM.
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
                    # Free mask memory immediately
                    del pred, gt
                    # Periodic GC to prevent memory fragmentation buildup
                    if _img_idx % 100 == 99:
                        gc.collect()
                except Exception as e:
                    print(f"    [ERROR] {img_path}: {e}")
                    continue

            if any(metric_accum.values()):
                metrics = {k: float(np.mean(v)) if v else float("nan")
                           for k, v in metric_accum.items()}
                method_results[ds_name] = metrics
                print(f"  Results: {json.dumps(metrics, indent=4)}")

                # Save per-image CSV for later analysis
                csv_path = per_img_dir / f"{method_name}_{ds_name}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["image", "pixel_f1", "pixel_iou", "pixel_auc", "optimal_f1",
                                       "max_pred_score", "gt_tamper_ratio"]
                    )
                    writer.writeheader()
                    writer.writerows(per_image_rows)
                print(f"  Per-image CSV: {csv_path}")

                # Per-dataset checkpoint save (crash-safe, multi-process-safe)
                _lock_path = str(results_path) + ".lock"
                with open(_lock_path, "w") as _lf:
                    fcntl.flock(_lf, fcntl.LOCK_EX)
                    try:
                        if results_path.exists():
                            with open(results_path) as _ckpt_f:
                                _ckpt = json.load(_ckpt_f)
                        else:
                            _ckpt = {}
                        if method_name not in _ckpt:
                            _ckpt[method_name] = {}
                        _ckpt[method_name].update(method_results)
                        save_results(_ckpt, str(results_path))
                    finally:
                        fcntl.flock(_lf, fcntl.LOCK_UN)

            gc.collect()

        # Shut down subprocess worker (if any) to free GPU memory
        if callable(shutdown):
            shutdown()

        all_results[method_name] = method_results
        # Save after each method (allows partial resume)
        save_results(all_results, str(results_path))
        save_results_csv(all_results, str(output_dir / "results.csv"), experiment="E1")
        print(f"  [INFO] Checkpoint saved to {results_path}")

    # Final save
    save_results(all_results, str(results_path))
    save_results_csv(all_results, str(output_dir / "results.csv"), experiment="E1")
    print(f"\nAll E1 results saved to {results_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1: General forensics on document data")
    parser.add_argument("--config", default="configs/exp1_general_on_docs.yaml")
    parser.add_argument("--device", default=None,
                        help="CUDA device to use (e.g. 'cuda:0', 'cuda:1', 'cpu'). "
                             "Sets CUDA_VISIBLE_DEVICES for single-GPU isolation.")
    parser.add_argument("--method", default=None,
                        help="Run only this method (e.g. 'TruFor'). "
                             "Useful for parallel per-GPU execution.")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated list of dataset names to run (e.g. 'MixTamper,FantasyID'). "
                             "Useful for splitting datasets across GPUs.")
    args = parser.parse_args()

    if args.device is not None:
        # Map --device cuda:N to CUDA_VISIBLE_DEVICES=N so all loaders see
        # a single GPU as "cuda:0" without code changes.
        if args.device.startswith("cuda:"):
            gpu_id = args.device.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        elif args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ds_filter = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    run_experiment(args.config, method_filter=args.method, dataset_filter=ds_filter)
