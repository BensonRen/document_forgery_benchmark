# Results

| File | Contents |
|---|---|
| `paper/main_results.csv` | The numbers published in the paper (arXiv v3, Tables 4 and 5): Pixel-F1 for all 14 methods × 8 datasets, and Pixel-AUC and Oracle-F1 for the 7 datasets in Table 5 (Table 5 has no MixTamper column). |
| `full_metrics.csv` | All four pixel metrics (Pixel-F1, Pixel-IoU, Pixel-AUC, Oracle-F1) at full precision, from the pipeline's output files, for every (method, dataset) pair whose stored output reproduces the published values to three decimals. |
| `calibration/results.json` | The 8-pair calibration experiment (§8): Pixel-F1 at τ = 0.5, Oracle-F1, and the calibrated threshold and test F1 for N ∈ {10, 25, 50, 100, 200}, averaged over 20 draws. |
| `validation/results.csv` | Method validation on CASIAv1, Columbia, COVERAGE and IMD2020 (Appendix B). |

`full_metrics.csv` leaves one pair blank, PSCC-Net on MixTamper, marked "pending
verification": its stored output does not reproduce the published Pixel-F1. We will
update this file once that pair has been re-run.

New runs write to `results/E1/`, `results/E2/` and so on (git-ignored). See
[docs/PROTOCOL.md](../docs/PROTOCOL.md) §5.
