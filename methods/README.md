# Methods

DocForge-Bench evaluates 14 published detectors with their authors' pretrained weights.
We do not redistribute any third-party code or weights. This folder holds only what is
ours:

| Path | What it is |
|---|---|
| `pins.txt` | upstream repository and the exact commit we evaluated (`name\|url\|commit`) |
| `patches/` | small compatibility patches for newer NumPy / albumentations / timm (no change to model behavior) |
| `wrappers/` | our inference workers for methods that run in their own environment |
| `setup_notes/` | how the ManTraNet, CAT-Net and SAFIRE environments were built |

`bash scripts/setup_methods.sh` clones every repository into `methods/<name>/` at the
pinned commit, applies the patch and copies the wrapper in. The clones are git-ignored.
You then add each method's weights at the paths below.

## The 14 methods

Paths in the **Weights** column are relative to `methods/<repo>/`. **Env** is the
environment the method runs in: `main` is the top-level environment from
`requirements.txt`; anything else is a separate conda environment that the runner starts
as a worker process (see `CONDA_ENVS_DIR` / `DOCFORGE_CONDA_ENVS` in
`scripts/run_exp1.py`).

| # | Method | Repo (`pins.txt`) | Weights | Env | Our code |
|---|---|---|---|---|---|
| 1 | TruFor (CVPR 2023) | TruFor | `TruFor_train_test/pretrained_models/weights/trufor.pth.tar` | main | loader in `run_exp1.py` |
| 2 | ManTraNet (CVPR 2019) | ManTraNet | `pretrained_weights/ManTraNet_Ptrain4.h5` (ships with the repo) | `mantranet` (Py 3.8, TF 2.9.3) | `wrappers/ManTraNet/worker.py` |
| 3 | MVSS-Net (ICCV 2021) | MVSS-Net | `ckpt/mvssnet_model/mvssnet_casia.pt` | main | patch |
| 4 | CAT-Net v2 (IJCV 2022) | CAT-Net | `output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar`, `pretrained_models/DCT_djpeg.pth.tar`, `pretrained_models/hrnetv2_w48_imagenet_pretrained.pth` | `cat` (Py 3.8, PyTorch 2.4) | `wrappers/CAT-Net/worker.py`, patch |
| 5 | PSCC-Net (TCSVT 2022) | PSCC-Net | `checkpoints/HRNet.pth`, `checkpoints/NLCDetection.pth` | main | patch |
| 6 | IML-ViT (arXiv 2023) | IML-ViT | `checkpoints/iml-vit_checkpoint.pth` | main | loader in `run_exp1.py` |
| 7 | SAFIRE (AAAI 2025) | SAFIRE | `safire.pth`, `sam_vit_b_01ec64.pth` (SAM ViT-B) | `safire` (Py 3.10) | `wrappers/SAFIRE/worker.py` |
| 8 | DocTamper model / DTD (CVPR 2023) | DocTamper | `pths/dtd_doctamper.pth`, `pths/swin_imagenet.pt`, `pths/vph_imagenet.pt` | main | loader in `run_exp2.py`, patch |
| 9 | DTD, ForensicHub weights | ForensicHub | `checkpoints_doc/Doc_Bench/dtd-9.pth` | `dtd_ffdn` | `wrappers/ForensicHub/worker.py` |
| 10 | FFDN (ECCV 2024) | ForensicHub | `checkpoints_doc/Doc_Bench/ffdn_ckpt-8.pth` | `dtd_ffdn` | same worker |
| 11 | CAFTB-Net (TOMM 2024) | ForensicHub | `checkpoints_doc/Doc_Bench/caftb-9.pth` | `dtd_ffdn` | same worker |
| 12 | TIFDM (TCE 2024) | ForensicHub | `checkpoints_doc/Doc_Bench/tifdm-9.pth` | `dtd_ffdn` | same worker |
| 13 | ASCFormer (PR 2024) | RTM | `ASCFormer/work_dirs/ascformer_rtm/ascformer_model.pth` (config `ASCFormer/configs/ascformer/ascformer_rtm.py`) | `ascformer` (MMSegmentation 1.x) | `wrappers/RTM/ASCFormer/worker.py` |
| 14 | ADCD-Net (ICCV 2025) | ADCD-Net | `weights/ADCD-Net_exp_data/` containing `ADCDNet.pth`, `docres.pkl` and the extracted `DocTamperOCR` | `adcdnet` | `wrappers/ADCD-Net/worker.py` |

Config names: the configs and result files use `DTD_FH`, `CAFTB` and `ADCDNet` for rows 9,
11 and 14, and `DocTamper` for row 8.

## Environments

- **main.** `pip install -r requirements.txt` (we used Python 3.13 and PyTorch 2.10). It
  runs TruFor, MVSS-Net, PSCC-Net, IML-ViT and the DocTamper model in-process.
- **mantranet, cat, safire.** Step-by-step instructions, with the package versions that
  worked, are in `setup_notes/`.
- **dtd_ffdn.** Python 3.10, NumPy 1.24.4, PyTorch 2.7.1 (CUDA 12.8), `jpegio`,
  `segmentation-models-pytorch` and `timm`. The worker imports the ForensicHub model
  files directly and does not need the ForensicHub package installed.
- **ascformer, adcdnet.** Follow the upstream repositories' installation instructions at
  the pinned commit. ASCFormer is built on MMSegmentation 1.x (`mmengine`, `mmcv` 2.x).

The runner looks for each environment's Python at
`$DOCFORGE_CONDA_ENVS/<env>/bin/python`, defaulting to `~/.conda/envs`. ASCFormer and
ADCD-Net read their environment name from the `METHOD_ENV` file in their wrapper folder.

## How a worker works

A worker is a small script that loads one model and then loops: read
`<image_path>\t<output.npy>` from stdin, run the model, save the score map (float32 in
[0, 1], image size) to the `.npy` path, print `DONE` (or `ERROR`). The main process starts
it once per (method, dataset), sends every image, and scores the saved maps exactly like
an in-process method. This keeps incompatible dependency stacks (TensorFlow 2.9,
MMSegmentation, older PyTorch) out of the main environment without changing how scores
are computed.

## Weight sources

Every checkpoint comes from the method's authors: the links in the upstream README at the
pinned commit, or the authors directly where noted in the paper (CAT-Net). ForensicHub's
document checkpoints come from the ForensicHub release. Please follow each project's
license and terms.
