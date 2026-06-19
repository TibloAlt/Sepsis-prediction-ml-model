# Results Summary

All models were trained on 2,000 patients from the PhysioNet 2019 Sepsis Challenge dataset.  
Test size: 50% split. Primary challenge: **severe class imbalance** (sepsis-positive cases are a small minority).

---

## Linear Regression — Predicting MAP (Mean Arterial Pressure)

| Model Version | MSE | R² | Threshold Met? |
|---|---|---|---|
| Baseline (OLS, row-level split) | — | 0.034 | — |
| + Feature Scaling (StandardScaler) | — | 0.034 | ❌ No change |
| + Ridge Regularization (α=1.0) | lower than baseline | >0.054 | ✅ Pass (Δ > 0.02) |
| + Lasso Regression | 88.46 | 0.607 | ✅ Pass (R² > 0) |
| + Patient-level Split | higher than baseline | lower | ✅ Pass (valid trade-off) |

**Key finding:** Lasso regression achieved the strongest predictive performance (R² = 0.607) while also performing implicit feature selection. Patient-level splitting produced a more realistic evaluation by eliminating data leakage.

---

## Random Forest Classifier — Sepsis Detection

| Model Version | Accuracy | Recall (Sepsis) | F1 (Sepsis) | Threshold Met? |
|---|---|---|---|---|
| Baseline (50 trees) | ~0.92 | 0.04–0.05 | — | — |
| + Class Weighting | — | 0.06 | — | ❌ Below 0.25 |
| + 5-Fold Cross-Validation | — | 0.10 | ≥ baseline | ✅ Pass (Δ > 0.05) |
| + Threshold = 0.3 | 0.9081 | 0.16 | 0.25 | ✅ Partial |
| + 75 Trees | higher | higher | higher | ✅ Pass |

**Key finding:** High accuracy is misleading here — models defaulted to the majority class. Cross-validation and increasing tree count produced the most reliable improvements. Threshold lowering improved recall but didn't solve underlying imbalance.

---

## MLP Neural Network — Sepsis Detection

| Model Version | Accuracy | Recall (Sepsis) | F1 (Sepsis) | Threshold Met? |
|---|---|---|---|---|
| Baseline (1 hidden layer, sigmoid) | — | ~0 | — | — |
| + Optimised Training Params | — | 0.00 | — | ✅ Efficiency (+25% fewer iterations) |
| + ReLU Activation | — | 0.00 | — | ❌ No recall improvement |
| + Custom Threshold (0.2) | 0.9799 | 0.02 | 0.03 | ❌ Recall near-zero |
| + Two Hidden Layers (32 → 16) | higher | higher | higher | ✅ Pass (F1 > 5% improvement) |

**Key finding:** All neural network variants struggled with recall due to class imbalance dominating model behaviour. Training optimisation reduced iterations by 25% (above the 20% threshold), confirming efficiency gains. Architectural improvements (two hidden layers) produced the most meaningful classification gains.

---

## Cross-Model Observations

1. **Class imbalance was the primary bottleneck** across all classification models — not architecture or hyperparameters.
2. **Recall is the right metric** for sepsis detection; accuracy above ~92% is achievable by predicting "no sepsis" for every patient.
3. **Data leakage via row-level splits** inflated regression performance — patient-level splits are necessary for honest clinical evaluation.
4. **Feature scaling mattered for neural networks** but had no effect on linear regression coefficients or R².
5. **Future work:** SMOTE or class-weighted loss functions should be applied before further architectural experimentation.
