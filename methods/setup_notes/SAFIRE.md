# SAFIRE Setup Notes

## Overview

SAFIRE ("Segment Any Forged Image Region", AAAI 2025) extends SAM (Segment
Anything Model) with adaptor layers for image forgery localization. It uses
k-means clustering (k=2) of SAM segment features to distinguish forged from
authentic regions.

- **Paper**: https://arxiv.org/abs/2412.08197
- **GitHub**: https://github.com/mjkwon2021/SAFIRE
- **Authors**: Myung-Joon Kwon, Wonjun Lee, Seung-Hun Nam, Minji Son, Changick Kim

## Architecture

- Base: SAM ViT-B with adaptor layers injected into the image encoder
- Inference: Grid-based point prompts (16x16 = 256 points) → segment masks
- Binary localization: k-means (k=2) on per-segment image encoder features;
  smaller cluster = forged region
- Output: Float32 probability map [0, 1] at original image resolution

## Python / PyTorch Version

- Python 3.10
- PyTorch 2.5.1 with CUDA 11.8 (conda install from pytorch channel)
- See `environment.yaml` for full pinned dependencies
- See `manual_env_setup.txt` for the recommended minimal install commands

## Conda Environment Setup

```bash
conda create -n safire python=3.10 -y
conda install -n safire pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
~/.conda/envs/safire/bin/pip install \
    connected-components-3d SimpleITK monai matplotlib scikit-image nibabel \
    tqdm scipy ipympl opencv-python jupyterlab ipywidgets pycocotools \
    onnx onnxruntime albumentations gdown kmeans-pytorch
```

Status: conda env `safire` created and all dependencies installed (2026-02-20).

## Weights

### SAM ViT-B Checkpoint (PUBLIC — downloaded)

File: `sam_vit_b_01ec64.pth` (358 MB)

Downloaded from Meta's official public URL:
```bash
wget -O methods/SAFIRE/sam_vit_b_01ec64.pth \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
```

Status: Downloaded and verified (2026-02-20).

### SAFIRE Weights (OWNER-RESTRICTED — NOT downloaded)

File: `safire.pth`
File: `safire_encoder_pretrained.pth` (encoder pretraining only, optional)

Google Drive folder: https://drive.google.com/drive/folders/1NRxep2G42OnVwCR9sGdf1iPqhCUrGmv2

Attempted download with `gdown` — failed with:
> "Only the owner and editors can download this file."

This is the same restriction as CAT-Net. The Google Drive sharing settings do
not allow public programmatic download.

**To obtain the weights:**
- Contact the author (mjkwon2021) via the GitHub issues page
- Request sharing the Google Drive folder with your account
- Or wait for the authors to make the weights publicly accessible

**Once obtained**, place `safire.pth` in `methods/SAFIRE/safire.pth`.

## Worker Protocol

`worker.py` implements the standard DocForge-Bench subprocess protocol:

1. Loads SAM ViT-B adaptor + SAFIRE fine-tuned weights at startup
2. Prints `READY\n` to stdout when ready
3. Reads `<img_path>\t<out_npy_path>\n` from stdin for each image
4. Runs binary forgery localization (`SafirePredictor.safire_predict`)
5. Saves float32 numpy array (H, W) in [0, 1] to `out_npy_path`
6. Prints `DONE\n` or `ERROR\n` to stdout

Usage:
```bash
~/.conda/envs/safire/bin/python worker.py \
    --weights methods/SAFIRE/safire.pth \
    --sam_checkpoint methods/SAFIRE/sam_vit_b_01ec64.pth \
    --device cuda:0 \
    --points_per_side 16 \
    --points_per_batch 256
```

Memory note: `--points_per_batch 256` should fit in ~8 GB VRAM with ViT-B.
Reduce to 128 or 64 if OOM.

## Integration

- **Loader**: `load_safire_model(cfg)` in `scripts/run_exp1.py`
- **Config**: `configs/exp1_general_on_docs.yaml` under `methods`
- **Env**: `safire` conda environment
- **Worker**: `methods/SAFIRE/worker.py`

The loader uses `_make_subprocess_worker_loader` (already present in
run_exp1.py at line 223) to spawn the worker in the `safire` conda env.

## Inference Notes

- SAFIRE processes images at 1024x1024 internally (SAM's native resolution)
- `infer_binary.py` uses `points_per_side=16` (256 grid points) — we match this
- The final prediction is resized back to the original image resolution with
  `cv2.INTER_LINEAR` interpolation before saving

## Known Issues / Limitations

1. **safire.pth not yet downloaded** — weights are Google Drive owner-restricted.
   SAFIRE will be skipped in E1 results until weights are obtained.
2. SAFIRE was designed for natural-image forensics (trained on
   FantasticReality, CASIA, IMD2020, TampCOCO) — document domain transfer
   performance is the main research question for E1.
3. Inference is slower than CNN-based methods due to SAM's grid-based
   point prompting (256 prompts per image at 1024x1024).
