# Evaluation protocol

This page describes exactly what the code does when it scores a method on a dataset, and
points to the function that does each step. Section numbers refer to the paper
([arXiv:2603.01433](https://arxiv.org/abs/2603.01433)).

## 1. Zero-shot, frozen weights

Every method runs with the pretrained weights its authors published. Nothing is fine-tuned
or adapted to documents, and no threshold is tuned. Where a method was trained on one of the
benchmark datasets, we evaluate on that dataset's held-out test split, and the result is
in-domain (see the paper, §5, "Training data and in-domain evaluation").

## 2. Image selection

`load_dataset()` in `scripts/run_exp1.py`:

1. Lists every image in the dataset's image directory that has a mask with the same stem.
2. If the config gives a `split_file`, keeps only the stems listed in it. RealTextManipulation
   uses `test_tampered.txt`, the 1,203 tampered images of the official test split.
3. If the config gives `max_samples` (1,000 for DocTamper, RealTextManipulation and
   MixTamper), draws that many pairs with `random.seed(42); random.sample(...)`, then sorts
   them by path.

| Dataset | Images scored | Selection |
|---|---|---|
| DocTamper | 1,000 | sampled (seed 42) from the official test split |
| T-SROIE | 360 | full test split |
| RealTextManipulation | 1,000 | sampled (seed 42) from the 1,203 tampered test images |
| Tampered-IC13 | 233 (188 tampered + 45 authentic) | full test split |
| ReceiptForgery | 218 (35 forged + 183 authentic) | full test split |
| FantasyID | 2,773 (2,173 attack + 600 bona fide) | full test split |
| MixTamper | 1,000 | sampled (seed 42) from the 6,817-image test split |
| FSTS-1.5k | 1,488 | full set |

Authentic images stay in the run (they are needed for image-level AUROC, §7 below) but
are excluded from the pixel-level averages by the NaN rule in §4.

## 3. Inference and resizing

- Each method runs at its own native input handling (`run_exp1.py`, `run_exp2.py`,
  `run_exp_fantasyd.py`). For example, TruFor accepts up to 4096×4096 px. MVSS-Net and
  PSCC-Net resize to 512×512. IML-ViT pads or crops to 1024×1024, and on larger images
  only the top-left 1024×1024 region is scored.
- The predicted score map is resized to the ground-truth mask size with bilinear
  interpolation (`cv2.resize`, default `INTER_LINEAR`).
- If the mask has more than 2,000,000 pixels, the prediction and the mask are both
  downsampled with area interpolation (`cv2.INTER_AREA`) to at most 2 MP before scoring.
  Smaller images are not resized.
- Masks are binarized as "any non-zero pixel is tampered". For MixTamper's RGB masks this
  merges all five tampering types into one class.
- Bounding-box datasets (Tampered-IC13, ReceiptForgery, T-SROIE, FantasyID) are rasterized
  to filled rectangles once, during data preparation (see [DATASETS.md](DATASETS.md)).

Methods that need a different software stack run in their own conda environment through a
small persistent worker process: `_make_subprocess_worker_loader()` in `run_exp1.py` and
`run_exp2.py`, with workers in `methods/wrappers/`. The worker reads image paths on stdin,
writes each score map to a `.npy` file, and the main process scores it like any other
method.

## 4. Pixel-level metrics

All four metrics are computed per image by `compute_all_pixel_metrics()` in
`scripts/evaluate.py`. Each dataset score is the mean over images, ignoring NaN values
(`evaluate_pixel_localization()`).

| Metric | Per-image definition | NaN when |
|---|---|---|
| **Pixel-F1** | F1 of `score >= 0.5` against the mask | the mask has no tampered pixels |
| **Pixel-IoU** | intersection over union of `score >= 0.5` and the mask | the mask has no tampered pixels |
| **Pixel-AUC** | ROC AUC over all pixels (`sklearn.metrics.roc_auc_score`; tied scores count half) | the mask is all tampered |
| **Oracle-F1** | the best F1 over 50 linearly spaced thresholds strictly inside (0, 1), `np.linspace(0, 1, 52)[1:-1]` | the mask is all tampered |

The NaN rule is what keeps authentic images out of the pixel-level averages. For example,
ReceiptForgery's pixel scores are means over its 35 forged test images, not all 218.

Oracle-F1 is a per-image ceiling. It picks the best threshold for each image separately,
so no single deployed threshold can reach it. `scripts/global_opt_f1.py` computes the
tighter "one global threshold" ceiling, and uses its own 100-point grid for the per-image
reference value it reports alongside.

## 5. Experiments

| Paper | Script | Config | Output |
|---|---|---|---|
| Main benchmark, 7 general methods (Tables 4–5) | `run_exp1.py` | `configs/exp1_general_on_docs.yaml` | `results/E1/` |
| Main benchmark, 7 document-specific methods plus the general ones (Tables 4–5) | `run_exp2.py` | `configs/exp2_doc_specific.yaml` | `results/E2/` |
| FantasyID column, all 14 methods | `run_exp_fantasyd.py` | built in | merged into `results/E1/` and `results/E2/` |
| Method validation on CASIAv1 / IMD2020 / Columbia / COVERAGE (Tables 6–7) | `run_exp1.py` | `configs/exp_validation.yaml` | `results/E_validation/` |
| Calibration experiment, 8 pairs (§8) | `run_calibration_exp.py` | the E1/E2 configs | `results/calibration/` |
| Image-level AUROC (§6.1, "future extension") | `image_level_auroc.py`, `compute_image_metrics.py` | — | per-method CSVs |
| Single global threshold ceiling, all pairs (Discussion) | `global_opt_f1.py`, `calibration_full.py` | — | CSV + figures |

`run_exp1_add_datasets.py` and `run_exp2_add_datasets.py` re-run only the named datasets
and merge the results into an existing results file. They are useful when adding a dataset
without re-running everything.

Each run writes `results/<E>/results.json` (method → dataset → metrics),
`results/<E>/results.csv` (one row per metric), and per-image CSVs under
`results/<E>/per_image/`. The per-image CSVs include `max_pred_score` and
`gt_tamper_ratio`, which the image-level scripts use.

## 6. Calibration experiment (§8)

`scripts/run_calibration_exp.py` runs a method over a dataset and stores, per image,
two 256-bin histograms of the predicted scores: one for tampered pixels and one for
authentic pixels (about 2 KB per image). From these it simulates choosing one global
threshold on N ∈ {10, 25, 50, 100, 200} randomly drawn images, without replacement, and
scoring the remaining images at that threshold. Each N is repeated over 20 random draws.
For bin `b = floor(t · 256)`: TP = Σ pos[b:], FP = Σ neg[b:], FN = Σ pos[:b].

The paper reports 8 (method, dataset) pairs: 6 chosen from the top quartile of the AUC–F1
gap, plus two negative controls (TruFor/MixTamper and CAFTB-Net/T-SROIE). Their output is
in `results/calibration/results.json`. `calibration_full.py` runs the same analysis over
all 112 pairs from stored per-image predictions. That is the extension the paper leaves
to future work.

## 7. Image-level detection (§6.1)

Tampered-IC13, ReceiptForgery and FantasyID contain authentic images, so an image-level
detection score can be evaluated. `scripts/image_level_auroc.py` turns each predicted
map into one score per image in three ways: the maximum pixel, the mean of the top 1% of
pixels, and the mean of the top 5%. It then computes AUROC (authentic = 0, tampered = 1).
`scripts/compute_image_metrics.py` computes image AUC, F1, AP and accuracy from the
`max_pred_score` column of the per-image CSVs.

## 8. Method validation (Appendix B)

Before scoring documents, the general methods were run on their usual benchmarks to check
that the weights load and inference runs correctly. `scripts/prepare_validation_datasets.py`
and `scripts/download_validation_datasets.sh` prepare CASIAv1, Columbia, COVERAGE and
IMD2020. `scripts/compare_validation_results.py` puts our numbers next to the published
ones. Our outputs are in `results/validation/results.csv`.
