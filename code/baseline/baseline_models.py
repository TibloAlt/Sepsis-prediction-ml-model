"""
baseline_models.py
==================
Sepsis Prediction — Baseline Models
Covers: EDA, Linear Regression, Random Forest Classifier, MLP Neural Network

Data: PhysioNet 2019 Sepsis Challenge (training_setA)
Target variables:
    - MAP         (continuous) → regression task
    - SepsisLabel (binary)    → classification task

Features used:
    HR, FiO2, Temp, Age, Resp, Gender, BaseExcess
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────

DATA_PATH = r"path/to/your/training_setA"   # ← update this
N_PATIENTS = 2000                            # number of patient files to load
RANDOM_STATE = 42
TEST_SIZE = 0.5

FEATURES = ["HR", "FiO2", "Temp", "Age", "Resp", "Gender", "BaseExcess"]
TARGET_CLASS = "SepsisLabel"
TARGET_REG   = "MAP"

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: str, n_patients: int) -> pd.DataFrame:
    """
    Load and combine individual patient .psv files into a single DataFrame.
    Adds a patient_id column to preserve patient-level identity.
    """
    files = glob.glob(os.path.join(path, "*.psv"))[:n_patients]

    frames = []
    for f in files:
        df = pd.read_csv(f, sep="|")
        df["patient_id"] = os.path.basename(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} patients → {combined.shape[0]:,} rows, {combined.shape[1]} columns")
    return combined


df_raw = load_data(DATA_PATH, N_PATIENTS)

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

def plot_sepsis_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(5, 4))
    sns.countplot(x=TARGET_CLASS, data=df)
    plt.title("Sepsis vs Non-Sepsis Distribution")
    plt.xlabel("Sepsis Label (0 = No, 1 = Yes)")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Histogram + boxplot for a selected set of EDA features."""
    eda_features = {
        "Glucose": "Glucose (mg/dL)",
        "Temp":    "Temperature (°C)",
        "WBC":     "WBC (×10³/μL)",
        "Gender":  "Gender (1 = Male, 0 = Female)",
    }

    for col, xlabel in eda_features.items():
        if col not in df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col].dropna(), bins=50, ax=axes[0])
        axes[0].set_title(f"{col} Distribution")
        axes[0].set_xlabel(xlabel)

        sns.boxplot(x=TARGET_CLASS, y=col, data=df, ax=axes[1])
        axes[1].set_title(f"{col} vs Sepsis Label")
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlation Heatmap") -> None:
    corr = df.select_dtypes(include="number").corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.show()


print("\n── EDA ──────────────────────────────────────")
plot_sepsis_distribution(df_raw)
plot_feature_distributions(df_raw)
plot_correlation_heatmap(df_raw, "Raw Feature Correlation Heatmap")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop columns with >90% missing values.
    2. Impute remaining numeric NaNs with column medians (robust to outliers).
    3. Drop rows missing the target variables.
    """
    # Drop high-missingness columns
    missing_frac = df.isnull().mean()
    to_drop = missing_frac[missing_frac > 0.9].index
    df = df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} columns with >90% missing values")

    # Median imputation for numeric columns
    df = df.fillna(df.median(numeric_only=True))

    # Drop rows missing either target
    df = df.dropna(subset=[TARGET_CLASS, TARGET_REG])
    print(f"After preprocessing: {df.shape[0]:,} rows remaining")
    return df


df = preprocess(df_raw)
plot_correlation_heatmap(df, "Cleaned Feature Correlation Heatmap")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

X = df[FEATURES]
y_class = df[TARGET_CLASS]
y_reg   = df[TARGET_REG]

X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ─────────────────────────────────────────────
# 5. MODEL 1 — LINEAR REGRESSION (baseline)
# ─────────────────────────────────────────────

print("\n── Linear Regression (Baseline) ────────────")

reg = LinearRegression()
reg.fit(X_train, y_train_r)
y_pred_r = reg.predict(X_test)

print(f"MSE : {mean_squared_error(y_test_r, y_pred_r):.4f}")
print(f"R²  : {r2_score(y_test_r, y_pred_r):.4f}")

# ─────────────────────────────────────────────
# 6. MODEL 2 — RANDOM FOREST CLASSIFIER (baseline)
# ─────────────────────────────────────────────

print("\n── Random Forest Classifier (Baseline) ─────")

rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
rf.fit(X_train, y_train_c)
y_pred_c = rf.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test_c, y_pred_c):.4f}")
print(classification_report(y_test_c, y_pred_c, target_names=["No Sepsis", "Sepsis"]))

# ─────────────────────────────────────────────
# 7. MODEL 3 — MLP NEURAL NETWORK (baseline)
# ─────────────────────────────────────────────

print("\n── MLP Neural Network (Baseline) ────────────")

# Scale features — required for stable MLP training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

nn = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="logistic",      # sigmoid
    max_iter=100,
    random_state=RANDOM_STATE,
)
nn.fit(X_train_scaled, y_train_c)
y_pred_nn = nn.predict(X_test_scaled)

print(f"Accuracy : {accuracy_score(y_test_c, y_pred_nn):.4f}")
print(classification_report(y_test_c, y_pred_nn, target_names=["No Sepsis", "Sepsis"]))
