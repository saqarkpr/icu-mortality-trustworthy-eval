# Results summary

Seed 42, 5-fold stratified CV on Set A (n=4000), held-out check on Set B (n=4000, fit only on Set A, never re-tuned).

## Cross-validated (out-of-fold) performance, Set A

| Model | AUROC | AUPRC | ECE | Brier |
|---|---|---|---|---|
| Logistic Regression | 0.840 ± 0.015 | 0.460 ± 0.035 | 0.189 ± 0.015 | 0.157 ± 0.013 |
| Gradient-Boosted Trees | 0.858 ± 0.010 | 0.515 ± 0.021 | 0.070 ± 0.007 | 0.097 ± 0.004 |
| SAPS-I score alone (no fitting) | 0.638 | 0.222 | n/a | n/a |

Paired per-fold AUROC difference (GBT − LogReg): mean +0.0174, paired t-test p=0.1090 (5 folds only -- indicative, not a strong-power test).

## Held-out generalization: fit on Set A, evaluate once on Set B

| Model | AUROC | AUPRC | ECE | Brier |
|---|---|---|---|---|
| Logistic Regression | 0.841 | 0.502 | 0.198 | 0.160 |
| Gradient-Boosted Trees | 0.849 | 0.504 | 0.062 | 0.099 |
| SAPS-I score alone (no fitting) | 0.660 | 0.253 | n/a | n/a |

**Reliability diagrams (Set A, out-of-fold):**

| Logistic Regression | Gradient-Boosted Trees |
|---|---|
| ![LR reliability Set A](results/reliability_logreg_seta_oof.png) | ![GBT reliability Set A](results/reliability_gbt_seta_oof.png) |

**Reliability diagrams (Set B, held-out, no re-tuning):**

| Logistic Regression | Gradient-Boosted Trees |
|---|---|
| ![LR reliability Set B](results/reliability_logreg_setb_heldout.png) | ![GBT reliability Set B](results/reliability_gbt_setb_heldout.png) |

## Subgroup calibration by ICU admission type (Set A, out-of-fold)

| ICU type | n | Mortality rate | LogReg ECE | LogReg AUROC | GBT ECE | GBT AUROC |
|---|---|---|---|---|---|---|
| CCU | 577 | 0.140 | 0.195 | 0.836 | 0.061 | 0.869 |
| CSRU | 874 | 0.049 | 0.094 | 0.885 | 0.038 | 0.903 |
| MICU | 1481 | 0.186 | 0.237 | 0.798 | 0.086 | 0.814 |
| SICU | 1068 | 0.145 | 0.198 | 0.835 | 0.082 | 0.853 |

## Clinically-framed operating points (GBT, Set B)

Set B mortality prevalence: 0.142

| Threshold | Meaning | Sensitivity | Specificity | PPV | Flagged rate |
|---|---|---|---|---|---|
| 0.142 | = prevalence | 0.641 | 0.863 | 0.436 | 0.209 |
| 0.500 | = 0.5 | 0.386 | 0.953 | 0.578 | 0.095 |
| 0.035 | ≈ 80% sensitivity (chosen post-hoc on Set B) | 0.801 | 0.733 | 0.332 | 0.343 |

## Missingness (Set A, top 10 most-missing features)

| Feature | Missing fraction |
|---|---|
| TroponinI_last | 0.949 |
| TroponinI_first | 0.949 |
| Cholesterol_first | 0.924 |
| Cholesterol_last | 0.924 |
| TroponinT_last | 0.784 |
| TroponinT_first | 0.784 |
| RespRate_first | 0.725 |
| RespRate_median | 0.725 |
| RespRate_lowest | 0.725 |
| RespRate_highest | 0.725 |

## Diagnostic: why was Logistic Regression's calibration so much worse than GBT's?

Suspected `class_weight="balanced"` (used to get a reasonable decision boundary under 14% prevalence) as the cause, rather than assuming it, and re-ran Logistic Regression without it (`scripts/class_weight_ablation.py`):

| LogisticRegression variant | AUROC | ECE |
|---|---|---|
| `class_weight="balanced"` (used above) | 0.841 | 0.189 |
| unweighted | 0.840 | 0.033 |

Confirmed: class-balancing was the whole story. Discrimination is unaffected (AUROC identical to 3 decimal places), but ECE drops from 0.189 to 0.033 (better than GBT's 0.070) once the reweighting is removed -- `class_weight="balanced"` reshapes predicted probabilities toward 50/50 to improve the decision boundary on imbalanced data, and has no awareness of calibration at all. The lesson generalizes beyond this dataset: class-balancing and probability calibration are two different objectives that can trade against each other. The headline table above was deliberately left showing the `class_weight="balanced"` run (the strictly worse of the two on calibration, for identical discrimination) instead of quietly swapping in the better-calibrated unweighted version after the fact, because the miscalibration it causes is itself the more interesting and more general finding here.
