"""
nn_relu_activation.py
=====================
Design Decision — ReLU Activation Function for MLP Neural Network

Motivation:
    The baseline MLP uses the sigmoid (logistic) activation function, which
    compresses inputs into the range (0, 1). For deep networks, this causes
    gradients to become very small during backpropagation — the "vanishing
    gradient" problem — slowing or stalling learning entirely.

    ReLU (Rectified Linear Unit) outputs max(0, x), preserving gradients for
    positive inputs and allowing the network to learn faster and more reliably.
    This is particularly valuable when modelling complex, nonlinear
    relationships between clinical variables.

Design choice:
    activation='relu', all other parameters held constant (hidden_layer_sizes,
    max_iter, preprocessing, train/test split) to isolate the effect.

Acceptance threshold:
    - F1-score for the sepsis class must be higher than the sigmoid baseline
    - Recall for the sepsis-positive class must improve by at least 0.05
    - Overall F1-score must not decrease

Result: FAIL — Both sigmoid and ReLU models produced recall = 0 for the
        sepsis class. Class imbalance was the dominant issue, not the
        activation function. See reflections in the final report.
"""

import os
import glob

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_PATH = r"path/to/your/training_setA"   # ← update this
N_PATIENTS = 2000
RANDOM_STATE = 42
TEST_SIZE = 0.5

FEATURES = ["HR", "FiO2", "Temp", "Age", "Resp", "Gender", "BaseExcess"]
TARGET_CLASS = "SepsisLabel"

# ─────────────────────────────────────────────
# LOAD & PREPROCESS
# ─────────────────────────────────────────────

files = glob.glob(os.path.join(DATA_PATH, "*.psv"))[:N_PATIENTS]

frames = []
for f in files:
    patient_df = pd.read_csv(f, sep="|")
    patient_df["patient_id"] = os.path.basename(f)
    frames.append(patient_df)

df = pd.concat(frames, ignore_index=True)

missing_frac = df.isnull().mean()
df = df.drop(columns=missing_frac[missing_frac > 0.9].index)
df = df.fillna(df.median(numeric_only=True))
df = df.dropna(subset=[TARGET_CLASS])

X = df[FEATURES]
y = df[TARGET_CLASS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Feature scaling — required for stable MLP training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# BASELINE — Sigmoid activation
# ─────────────────────────────────────────────

nn_sigmoid = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="logistic",      # sigmoid
    max_iter=100,
    random_state=RANDOM_STATE,
)
nn_sigmoid.fit(X_train_scaled, y_train)
y_pred_sigmoid = nn_sigmoid.predict(X_test_scaled)

print("── Baseline MLP (sigmoid activation) ──────")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_sigmoid):.4f}")
print(classification_report(y_test, y_pred_sigmoid,
                             target_names=["No Sepsis", "Sepsis"]))

# ─────────────────────────────────────────────
# DESIGN DECISION — ReLU activation
# ─────────────────────────────────────────────

nn_relu = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",          # ReLU — avoids vanishing gradient
    max_iter=100,
    random_state=RANDOM_STATE,
)
nn_relu.fit(X_train_scaled, y_train)
y_pred_relu = nn_relu.predict(X_test_scaled)

print("── Design Decision MLP (ReLU activation) ──")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_relu):.4f}")
print(classification_report(y_test, y_pred_relu,
                             target_names=["No Sepsis", "Sepsis"]))

print("── Verification ────────────────────────────")
print("  Compare Sepsis recall and F1 between both reports above.")
print("  Threshold: +0.05 recall for Sepsis class, no F1 decrease.")
