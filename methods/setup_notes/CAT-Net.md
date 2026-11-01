# CAT-Net v2 — Setup Notes

## Overview

CAT-Net v2 (Kwon et al., IJCV 2022) detects image manipulation by analyzing
JPEG compression artifacts. It uses a dual-stream architecture with an RGB
stream (HRNet-W48 backbone) and a DCT coefficient stream.

This integration uses a **persistent subprocess worker** in the isolated `cat`
conda environment (Python 3.8, PyTorch 2.4) to avoid import collisions with
other methods in the DocForge-Bench framework.

## Weights Download

Weights are hosted on Google Drive and Baidu Pan:

- **Google Drive**: https://drive.google.com/drive/folders/1hBEfnFtGG6q_srBHVEmbF3fTq0IhP8jq?usp=sharing
- **Baidu Pan**: https://pan.baidu.com/s/1hecZC0IZXdgh5WRbRoAytQ (code: `ycft`)

**Note**: As of 2026-02-20, all files in this Google Drive folder are
owner-restricted and cannot be downloaded programmatically via gdown. Files
must be downloaded manually through the browser.

### Required Files

Place the downloaded files as follows (paths relative to `methods/CAT-Net/`):

```
methods/CAT-Net/
├── pretrained_models/
│   ├── DCT_djpeg.pth.tar           # DCT stream pretrained weights
│   └── hrnetv2_w48_imagenet_pretrained.pth  # RGB stream pretrained weights
└── output/
    └── splicing_dataset/
        └── CAT_full/
            └── CAT_full_v2.pth.tar  # Full model v2 (IJCV 2022)
```

## Environment Setup

The `cat` conda environment is created and configured automatically by the
setup described here. To recreate it manually:

```bash
conda create -n cat python=3.8 -y
conda activate cat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pyyaml yacs scipy tqdm torch_dct
pip install "jpegio @ git+https://github.com/dwgoon/jpegio.git"
```

Installed packages (2026-02-20):
- Python 3.8
- torch 2.4.1+cu118, torchvision 0.19.1+cu118
- jpegio 0.2.4 (compiled from source)
- torch_dct 0.1.6
- opencv-python 4.13.0.92
- yacs 0.1.8, pyyaml 6.0.3, scipy 1.10.1, tqdm 4.67.1

**Note**: The original CAT-Net requirements.txt lists `torch==1.1.0` and
`python=3.6`, but the model architecture is fully compatible with modern
PyTorch. We use PyTorch 2.4 for CUDA 11.8 compatibility with the rest of
the benchmark infrastructure.

## Worker Protocol

`methods/CAT-Net/worker.py` implements the standard DocForge-Bench persistent
worker protocol:

- Accepts `<img_path>\t<out_npy_path>\n` on stdin
- Outputs `DONE\n` or `ERROR\n` on stdout per image
- Sends `READY\n` on startup once the model is loaded
- Non-JPEG images are converted to JPEG on the fly (lossless quality=100)
  so the DCT pipeline always has valid quantization tables

## Integration

- **Loader**: `load_catnet()` in `scripts/run_exp1.py`
- **Config**: `configs/exp1_general_on_docs.yaml` (CAT-Net entry)
- **Model**: CAT_full_v2.pth.tar (IJCV 2022 general forgery model)

## Known Issues

- The original requirements.txt specifies `torch==1.1.0` (from 2021). We use
  modern torch instead — this works because the model only uses standard
  PyTorch operations.
- `jpegio` requires building a C extension (libjpeg). This is handled
  automatically by pip install.
- ManTraNet in the same experiment is disabled due to Keras compatibility
  issues. CAT-Net has no such issue with the modern torch setup.

## Paper Reference

```bibtex
@article{kwon2022learning,
  title={Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization},
  author={Kwon, Myung-Joon and Nam, Seung-Hun and Yu, In-Jae and Lee, Heung-Kyu and Kim, Changick},
  journal={International Journal of Computer Vision},
  volume={130},
  number={8},
  pages={1875--1895},
  year={2022},
  publisher={Springer},
}
```
