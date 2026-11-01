# DocForge-Bench

Evaluation code for **DocForge-Bench: A Comprehensive Benchmark for Document Forgery
Detection and Analysis** ([arXiv:2603.01433](https://arxiv.org/abs/2603.01433)).

DocForge-Bench runs 14 published forgery detectors **zero-shot** (each with its authors'
pretrained weights, no fine-tuning) on 8 document datasets covering text tampering,
receipt forgery and identity documents. It reports threshold-independent metrics next to
the usual fixed-threshold F1. The main finding is a calibration failure: many methods rank
tampered pixels well (Pixel-AUC ≥ 0.76) yet score near zero Pixel-F1 at τ = 0.5. Tampered
regions cover only 0.27–4.17% of a document image, so a fixed 0.5 threshold is badly
placed. Re-choosing one threshold on as few as 10 images from the target domain
recovers 39–55% of the gap in the high-AUC cases we tested.

This repository contains the full pipeline used for the paper: method loaders and
inference wrappers, the metric code, experiment configs, data-preparation scripts, the
calibration and image-level analyses, and our result files.

## What's here

```
configs/            experiment configs (methods, datasets, sampling, output paths)
scripts/            everything that runs: inference, metrics, data prep, analyses
methods/            pinned upstream commits, our compatibility patches and inference wrappers
docs/PROTOCOL.md    exactly how a method is scored, step by step, with pointers into the code
docs/DATASETS.md    where to get each dataset and how to lay it out under data/
results/            the paper's numbers as CSV, full per-pair metrics, calibration and validation outputs
```

Main scripts:

| Script | Role |
|---|---|
| `scripts/evaluate.py` | Pixel-level metrics (Pixel-F1, Pixel-IoU, Pixel-AUC, Oracle-F1) and image-level metrics |
| `scripts/run_exp1.py` | Loaders for the 7 general forensic methods; dataset loading and sampling; runs a config |
| `scripts/run_exp2.py` | Adds the 7 document-specific methods; runs the main benchmark config |
| `scripts/run_exp_fantasyd.py` | Runs all methods on FantasyID |
| `scripts/run_calibration_exp.py` | The calibration experiment (per-image score histograms, N-image threshold calibration) |
| `scripts/image_level_auroc.py`, `scripts/compute_image_metrics.py` | Image-level detection AUROC on the three datasets that include authentic images |
| `scripts/global_opt_f1.py`, `scripts/calibration_full.py` | Single-global-threshold ceiling and calibration across all 112 pairs |
| `scripts/compare_validation_results.py` | Validation of each method against its published numbers |
| `scripts/setup_methods.sh` | Clones the 11 upstream repositories at our pinned commits and installs patches and wrappers |

## Quick start

```bash
git clone https://github.com/scamai/docforge-bench.git
cd docforge-bench

# 1. Main environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Methods: clone at the evaluated commits, then add the authors' weights
bash scripts/setup_methods.sh
#    weight files and per-method environments: methods/README.md

# 3. Datasets: download from their authors, then prepare (docs/DATASETS.md)
python scripts/extract_doctamper_lmdb.py --lmdb-root data/DocTamper_lmdb --output-dir data/DocTamper
python scripts/generate_tsroie_masks.py --data-root data/T-SROIE --split test
python scripts/generate_bbox_masks.py --dataset receipt_forgery
python scripts/generate_bbox_masks.py --dataset tampered_ic13 --data_root data/Tampered-IC13
python scripts/generate_fantasyd_masks.py

# 4. Run
python scripts/run_exp2.py --config configs/exp2_doc_specific.yaml          # 14 methods × 7 datasets
python scripts/run_exp_fantasyd.py                                           # FantasyID column
python scripts/run_exp2.py --config configs/exp2_doc_specific.yaml \
    --method PSCC-Net --datasets FSTS-1.5k                                   # one method / dataset
```

Run everything from the repository root. Use `CUDA_VISIBLE_DEVICES` to pick a GPU; we
used one NVIDIA GPU with at least 24 GB of memory. Each run writes
`results/<E>/results.json`, `results/<E>/results.csv` and per-image CSVs under
`results/<E>/per_image/`.

Calibration and image-level analyses:

```bash
python scripts/run_calibration_exp.py --config configs/exp1_general_on_docs.yaml \
    --method PSCC-Net --dataset FSTS-1.5k
python scripts/compute_image_metrics.py --all
python scripts/image_level_auroc.py --help
```

## Protocol in brief

- **Zero-shot.** Official pretrained weights, no fine-tuning, no threshold tuning.
- **Sampling.** DocTamper, RealTextManipulation and MixTamper are capped at 1,000 test
  images, drawn with `random.seed(42)` and then sorted. Other datasets use their full test
  split.
- **Scoring.** Each predicted map is resized to the mask (bilinear). Maps and masks larger
  than 2 MP are jointly downsampled to 2 MP (area interpolation). Metrics are computed per
  image and averaged.
- **Metrics.** Pixel-F1 and Pixel-IoU at τ = 0.5; Pixel-AUC (threshold-free); Oracle-F1,
  the best per-image F1 over thresholds (a diagnostic ceiling, not a reachable operating
  point).
- **Authentic images.** Images with empty masks give NaN for pixel metrics and are left
  out of the dataset mean. They are still used for image-level AUROC.

The details and their exact location in the code are in
[docs/PROTOCOL.md](docs/PROTOCOL.md).

## Methods

| General image forensics | Document-specific |
|---|---|
| TruFor, ManTraNet, MVSS-Net, CAT-Net, PSCC-Net, IML-ViT, SAFIRE | DocTamper model (DTD), DTD (ForensicHub weights), FFDN, CAFTB-Net, TIFDM, ASCFormer, ADCD-Net |

Upstream repositories, pinned commits, weight locations and environments:
[methods/README.md](methods/README.md).

## Datasets

DocTamper, T-SROIE, RealTextManipulation, Tampered-IC13, ReceiptForgery, MixTamper,
FSTS-1.5k and FantasyID. None is redistributed here; see
[docs/DATASETS.md](docs/DATASETS.md).

## Results

`results/paper/main_results.csv` has the paper's Tables 4 and 5 in machine-readable form.
`results/full_metrics.csv` adds Pixel-IoU and full-precision values from the pipeline's
output files. See [results/README.md](results/README.md).

## Citation

```bibtex
@article{zhao2026docforgebench,
  title   = {{DocForge-Bench}: A Comprehensive Benchmark for Document Forgery Detection and Analysis},
  author  = {Zhao, Zengqi and Xia, Weidi and Wei, En and Zhang, Yan and Mo, Jane and
             Zhang, Tiannan and Dai, Yuanqin and Chen, Zexi and Tao, Yiran and Ren, Simiao},
  journal = {arXiv preprint arXiv:2603.01433},
  year    = {2026}
}
```

## License

The code in this repository is released under the MIT License (see `LICENSE`). Methods,
weights and datasets belong to their authors and keep their own licenses.

## Contact

Simiao Ren, benren@scam.ai. Issues and pull requests are welcome.
