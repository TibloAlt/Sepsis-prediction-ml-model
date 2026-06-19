# 🩺 Sepsis Prediction with Machine Learning

Early sepsis detection is one of the most time-sensitive problems in clinical medicine — delays in diagnosis significantly increase mortality. This project develops and evaluates three machine learning approaches for sepsis prediction using real physiological patient data, comparing a baseline configuration against targeted design improvements for each model type.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)

---

## Project Overview

This project was completed as part of a structured ML engineering course. Each of three model types was first implemented as a **baseline**, then modified with a specific **design decision** aimed at improving performance, efficiency, or clinical validity. Results were evaluated against pre-defined acceptance thresholds.

**Three model types:**
| Model | Task |
|---|---|
| Linear Regression | Predict continuous outcome (Mean Arterial Pressure) |
| Random Forest Classifier | Binary sepsis classification (non-neural) |
| MLP Neural Network | Binary sepsis classification (neural) |

---

## Dataset

- **Source:** PhysioNet Sepsis Challenge — `training_setA` (`.psv` files, one per patient)
- **Size:** 2,000 patients used for modelling
- **Target variables:** `SepsisLabel` (binary), `MAP` (continuous)
- **Severe class imbalance:** sepsis-positive cases are heavily underrepresented, consistent with real-world clinical data

### Selected Features

| Feature | Clinical Rationale |
|---|---|
| Heart Rate (HR) | Tachycardia is an early infection response |
| Respiratory Rate (Resp) | Part of qSOFA sepsis screening criteria |
| Temperature (Temp) | Fever/hypothermia indicates systemic inflammation |
| FiO2 | Elevated oxygen need signals respiratory dysfunction |
| Base Excess | Metabolic acidosis marker in septic patients |
| Age | Older patients face higher sepsis susceptibility |
| Gender | Controls for baseline physiological variability |

Features were selected using a combination of **correlation analysis** and **clinical domain knowledge**, prioritizing low inter-feature redundancy and coverage across vital signs, lab values, and demographics.

---

## Models

### 1. Linear Regression (predicting MAP)

| Version | Design Decision | Result |
|---|---|---|
| Baseline | Standard OLS, row-level split | R² = 0.034 |
| + Feature Scaling | StandardScaler applied | R² = 0.034 (no change) |
| + Ridge Regularization (α=1.0) | L2 penalty to reduce overfitting | ✅ R² improved > 0.02 threshold |
| + Lasso Regression | L1 regularization + feature selection | R² = 0.607, MSE = 88.46 |
| + Patient-level Split | Prevents data leakage across patients | Higher MSE, more realistic evaluation |

### 2. Random Forest Classifier (sepsis detection)

| Version | Design Decision | Result |
|---|---|---|
| Baseline | 50 trees, default params | Recall (sepsis) ≈ 0.04–0.05 |
| + Class Weighting | `class_weight='balanced'` | Recall = 0.06 |
| + K-Fold CV (k=5) | More reliable evaluation | ✅ Recall improved to 0.10 |
| + Threshold Lowering (0.3) | Prioritise sensitivity over precision | Recall = 0.16 |
| + 75 Trees | Reduced variance via ensemble averaging | ✅ Higher F1-score than baseline |

### 3. MLP Neural Network (sepsis detection)

| Version | Design Decision | Result |
|---|---|---|
| Baseline | 1 hidden layer (32 neurons), sigmoid, 100 iter | Recall (sepsis) ≈ 0 |
| + Training Optimisation | LR=0.001, batch=64, early stopping | ✅ 25% fewer iterations; recall unchanged |
| + ReLU Activation | Avoids vanishing gradient problem | Recall unchanged (class imbalance dominant) |
| + Custom Threshold (0.2) | Increases sensitivity via `predict_proba` | Recall = 0.02 |
| + Two Hidden Layers (32→16) | Captures more complex nonlinear patterns | ✅ Higher F1-score and recall |

---

## Results Summary

The most consistent finding across all three model types was that **class imbalance was the dominant bottleneck**, not model complexity or hyperparameter choice. Models repeatedly defaulted to predicting the majority (non-sepsis) class, achieving high accuracy but near-zero recall on the clinically critical positive class.

**What worked:**
- Ridge regularization meaningfully improved regression stability
- Patient-level splitting produced more honest (if lower) regression metrics
- K-fold cross-validation improved evaluation reliability for the classifier
- Two hidden layers in the neural network improved F1-score

**What didn't:**
- Feature scaling alone had no effect on linear regression performance
- Threshold lowering and class weighting improved recall marginally but not to acceptable clinical thresholds
- ReLU vs sigmoid made no difference when class imbalance wasn't addressed first

---

## Repository Structure

```
sepsis-ml-project/
│
├── README.md                   ← You are here
├── .gitignore
│
├── code/
│   ├── baseline/
│   │   └── baseline_models.py  ← EDA + all three baseline models
│   │
│   └── design_decisions/
│       ├── regression/         ← One file per design decision
│       ├── random_forest/
│       └── neural_network/
│
├── results/                    ← Classification reports, metric tables
│   └── results_summary.md
│
└── docs/
    └── final_report.md         ← Full written report
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sepsis-ml-project.git
cd sepsis-ml-project
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Prepare the data
Download `training_setA` from the [PhysioNet 2019 Sepsis Challenge](https://physionet.org/content/challenge-2019/1.0.0/) and place the `.psv` files in a local folder. Update the path variable in `baseline_models.py`:
```python
path = r"path/to/your/training_setA"
```

### 4. Run the baseline
```bash
python code/baseline/baseline_models.py
```

---

## Key Takeaways

1. **Class imbalance is a first-order problem in clinical ML.** Accuracy is a misleading metric — recall on the minority class is what matters for sepsis detection, where false negatives carry life-or-death consequences.

2. **Data leakage is subtle but critical.** Row-level train/test splits allowed patient measurements to appear in both sets, inflating performance. Patient-level splitting is the correct approach for time-series clinical data.

3. **Simpler improvements can outperform architectural changes.** Ridge regularization and cross-validation produced measurable gains, while adding neural network layers only helped once the evaluation metric was appropriate.

4. **Domain knowledge drives better feature engineering.** Selecting features based purely on correlation would have missed clinically validated indicators like Respiratory Rate (qSOFA criterion) and Base Excess (metabolic acidosis marker).

---

*Dataset: PhysioNet Computing in Cardiology Challenge 2019*
