# Results Summary

All models were trained on 2,000 patients from the PhysioNet 2019 Sepsis Challenge dataset.  
Test size: 50% split. Primary challenge: **severe class imbalance** (sepsis-positive cases are a small minority).

This file covers my individual design decisions only. The shared baseline results are included for comparison.

---

## Linear Regression — Predicting MAP (Mean Arterial Pressure)

**My design decision:** Ridge Regularization (L2, α=1.0)

| Model Version | MSE | R² | Threshold Met? |
|---|---|---|---|
| Baseline (OLS, row-level split) | — | 0.034 | — |
| Ridge Regression (α=1.0) | lower than baseline | >0.054 | ✅ PASS |

**Acceptance threshold:** R² must improve by at least 0.02 over baseline, without an increase in MSE.

**Outcome:** PASS. Ridge regularization successfully reduced the influence of correlated features by distributing coefficient weight more evenly, resulting in a more stable and generalizable model.

---

## Random Forest Classifier — Sepsis Detection

**My design decision:** 5-Fold Cross-Validation

| Model Version | Recall (Sepsis) | F1 (Sepsis) | Threshold Met? |
|---|---|---|---|
| Baseline (50 trees, single split) | 0.04 | — | — |
| 5-Fold Cross-Validation | 0.10 | ≥ baseline | ✅ PASS |

**Acceptance threshold:** Recall for the sepsis class must improve by at least 0.05; overall F1-score must not decrease.

**Outcome:** PASS. Cross-validation reduced evaluation variance by ensuring every data point was used for both training and validation. Recall improved from 0.04 to 0.10, reflecting a more honest and stable estimate of model performance compared to a single lucky or unlucky split.

---

## MLP Neural Network — Sepsis Detection

**My design decision:** ReLU Activation Function (replacing sigmoid)

| Model Version | Recall (Sepsis) | F1 (Sepsis) | Threshold Met? |
|---|---|---|---|
| Baseline (sigmoid activation) | 0 | — | — |
| ReLU Activation | 0 | — | ❌ FAIL |

**Acceptance threshold:** Recall for sepsis class must improve by at least 0.05; F1-score must not decrease.

**Outcome:** FAIL. Both the sigmoid and ReLU models produced a recall of 0 for the sepsis-positive class — neither detected a single positive case. The activation function was not the limiting factor. Severe class imbalance caused both models to default entirely to the majority class. The correct next step would be to address imbalance first (via SMOTE, class-weighted loss, or resampling) before evaluating the effect of activation function choice.

---

## Cross-Model Observations

1. **Class imbalance was the primary bottleneck** across all three models — not architecture or hyperparameter choice.
2. **Recall is the correct metric** for sepsis detection. Accuracy above ~92% is trivially achievable by predicting "no sepsis" for every patient.
3. **Evaluation methodology matters as much as model choice.** K-fold cross-validation surfaced a meaningfully different (and more honest) picture of classifier performance than a single split.
4. **Isolating one variable at a time** (as done here) is essential for interpreting results — but it also means a failing result can pinpoint exactly what isn't the problem, which is itself useful.
