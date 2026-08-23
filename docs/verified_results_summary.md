# Verified Results Summary

| Evaluation | Mineral mIoU | Mineral Dice | Interpretation |
|---|---:|---:|---|
| Original fixed source-held test (10 sources) | 0.2440 | 0.3732 | Final validation-calibrated B7=0.6/B8=0.4 ensemble result for one fixed split. |
| Repeated grouped CV (2 repeats × 5 folds) | 0.3294 ± 0.0459 | 0.4738 ± 0.0545 | Primary estimate; ten source-level hold-out evaluations. |

The test and grouped-CV figures should be reported separately. The grouped-CV value is not an improvement obtained by retuning on the original test set.

## Supplemental map-derived diagnostics

The 10-source held-out quantitative analysis produced mean mineral area MAE 0.0412 and mean mineral-composition L1 0.3294. These are **not mIoU** values despite their numerically matching rounded means. Alkali feldspar was over-represented by +0.0701 mean area fraction and quartz under-represented by −0.0546.

Muscovite–Schist–3 and Pyroxene–Peridotite–1 were analysed as source-level error cases. Their eight angular views were aggregated before reporting; they are not independent facies replicates.
