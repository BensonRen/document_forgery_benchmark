# Datasets

DocForge-Bench uses eight public datasets. We do not redistribute any of them: each comes
from its authors under its own license, and several require a request form. This page
lists where each one comes from, the folder layout the configs expect under `data/`, and
the preparation step, if any.

All paths are relative to the repository root. `scripts/download_datasets.sh [data]`
creates the folders and prints the download instructions it can.

| Dataset | Source | Expected layout | Preparation |
|---|---|---|---|
| **DocTamper** (Qu et al., CVPR 2023) | [github.com/qcf-568/DocTamper](https://github.com/qcf-568/DocTamper) (LMDB release) | `data/DocTamper/test/{images,masks}/*.png` | `python scripts/extract_doctamper_lmdb.py --lmdb-root data/DocTamper_lmdb --output-dir data/DocTamper` converts the test LMDB to PNG files |
| **T-SROIE** (Zhang et al., 2022; arXiv:2202.01999) | from the authors (Tampered-SROIE) | `data/T-SROIE/test/*.jpg`, `data/T-SROIE/sroie_test_1011.json` | `python scripts/generate_tsroie_masks.py --data-root data/T-SROIE --split test` rasterizes the COCO polygons to `test/masks/*.png` |
| **RealTextManipulation** (Luo et al.) | [github.com/DrLuo/RTM](https://github.com/DrLuo/RTM) | `data/RealTextManipulation/{JPEGImages,SegmentationClass}/`, `test_tampered.txt` | `test_tampered.txt` lists the stems of the 1,203 tampered images in the official test split (no `good_*` images) |
| **Tampered-IC13** (Wang et al., 2022) | from the authors | `data/Tampered-IC13/Tampered-IC13/{test_img,test_masks}/` | `python scripts/generate_bbox_masks.py --dataset tampered_ic13 --data_root data/Tampered-IC13` fills each box from `test_gt/` (the release unpacks to a nested `Tampered-IC13/` folder) |
| **ReceiptForgery** (Mesquita et al., ICDAR 2023, "Find it again") | [github.com/l3i-la-rochelle/receipt-forgery-detection](https://github.com/l3i-la-rochelle/receipt-forgery-detection) | `data/ReceiptForgery/findit2/{test,test_masks}/` | `python scripts/generate_bbox_masks.py --dataset receipt_forgery` fills each box from the `test.txt` manifest; authentic images get all-zero masks |
| **MixTamper** (PeerJ, 2024) | from the authors (FigShare) | `data/MixTamper/{img,mask}/*.png` | none; RGB masks are binarized as "any channel non-zero" |
| **FSTS-1.5k** (Yu et al., NeurIPS 2025 D&B; arXiv:2511.12658) | from the authors | `data/FSTS-1.5k/{images,masks}/*.png` | none |
| **FantasyID** (Korshunov et al., 2025) | Zenodo, CC-BY 4.0 | `data/FantasyID/FantasyID/` (as released, including `test.csv` and the per-image JSON) | `python scripts/generate_fantasyd_masks.py` writes `data/FantasyID/text_only/{images,masks}/`, filling every region marked `altered` |

## Notes

- **Bounding-box ground truth.** Tampered-IC13, ReceiptForgery, T-SROIE and FantasyID give
  boxes or polygons, not pixel masks. Filled rectangles include some background around the
  text, which affects Pixel-F1 and IoU on these datasets (paper §4).
- **Authentic images.** Tampered-IC13 (45), ReceiptForgery (183) and FantasyID (600) include
  untampered images with empty masks. They are run through every method so that
  image-level AUROC can be computed, but they are left out of pixel-level averages
  ([PROTOCOL.md](PROTOCOL.md) §4).
- **Sampling.** DocTamper, RealTextManipulation and MixTamper are capped at 1,000 images
  with `random.seed(42)`. Keep `max_samples: 1000` in the configs to score the same images
  we did.
- **ReceiptForgery's small positive set.** Only 35 test images are forged, so per-method
  results on it have high variance.

## Validation datasets (Appendix B)

CASIAv1, Columbia, COVERAGE and IMD2020 are only used to check that each general method
loads and runs correctly. `scripts/download_validation_datasets.sh` prints where to get
them, and `scripts/prepare_validation_datasets.py` arranges them into the layout in
`configs/exp_validation.yaml`.

## Catalogued but not evaluated

The paper also lists OSTF (access-restricted), SIDTD, MIDV-LAIT and MIDV-2020, which have
limited forgery coverage, as well as the standard natural-image corpora. They are not
needed to reproduce the results.
