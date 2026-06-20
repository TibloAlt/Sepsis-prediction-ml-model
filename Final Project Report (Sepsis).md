# Sepsis Prediction with Machine Learning — Final Report

## Table of Contents
- [Introduction](#introduction)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Design Summary](#design-summary)
- [Verification Test Plans](#verification-test-plans)
- [Results](#results)
- [Reflection](#reflection)
- [References](#references)

---

## Introduction

Machine learning models are becoming increasingly applied to support clinical decision-making in healthcare, particularly in predicting patient outcomes and identifying high-risk conditions such as sepsis. In this project, three models were developed and evaluated using physiological patient data to predict a continuous outcome and a binary classification outcome related to sepsis.

This project involved implementing three types of models: a linear regression model for predicting a continuous variable, a non-neural classification model for sepsis detection, and a neural network model for capturing more complex patterns in the data. Each model was first established using a baseline configuration, then modified based on a specific design decision aimed at improving performance or efficiency.

The primary objectives were to build effective models and systematically evaluate the impact of these design decisions. This was achieved by comparing baseline and modified models using pre-defined acceptance thresholds and testing performance on a held-out dataset. Through this structured approach, the project emphasises the importance of experimental validation and critical analysis in the development of machine learning models for healthcare applications.

---

## Exploratory Data Analysis

The training data for sepsis detection was obtained from the PhysioNet 2019 Sepsis Challenge. All individual `.psv` files were combined into a single `.csv` to simplify data handling and enable dataset-wide analysis. Each file corresponds to a unique patient, and a `patient_id` column was added to preserve patient-level distinctions after merging. The resulting dataset contains time-series clinical measurements, including vital signs, laboratory values, and demographic information collected during patient monitoring.

Sepsis is a life-threatening condition caused by the body's dysregulated response to infection, and delayed diagnosis significantly increases mortality risk. Identifying patterns in physiological and laboratory data that precede sepsis onset is therefore essential for developing useful predictive models.

Initial exploratory data analysis (EDA) was performed by visualising selected features including glucose levels, temperature, white blood cell (WBC) count, and gender — chosen for their known clinical relevance to infection and systemic inflammation. The distribution of the target variable (`SepsisLabel`) revealed significant class imbalance, with sepsis cases heavily underrepresented, consistent with real-world clinical data.

A correlation matrix was computed and visualised as a heatmap to identify linear relationships between features and detect potential multicollinearity. Substantial missingness was also identified across many features; columns with more than 90% missing values were removed, and remaining numeric values were imputed using column medians. Median imputation was selected for its robustness to outliers and better representation of the central tendency in skewed clinical data.

### Selected Features

Feature selection combined statistical correlation analysis with clinical domain knowledge, prioritising:
- Meaningful correlation with the target variable
- Low inter-feature redundancy
- Coverage across vital signs, laboratory measurements, and demographics

| Feature | Clinical Rationale |
|---|---|
| Heart Rate (HR) | Tachycardia is an early compensatory response to infection and systemic inflammation [4] |
| Respiratory Rate (Resp) | Elevated respiratory rate is part of the qSOFA sepsis screening criteria [3] |
| Temperature (Temp) | Fever or hypothermia is a well-established indicator of systemic inflammatory response [2] |
| FiO2 | Higher oxygen requirements reflect respiratory dysfunction associated with severe sepsis [5] |
| Base Excess | Abnormal base excess indicates metabolic acidosis, common in septic patients [6] |
| Age | Older patients exhibit higher sepsis susceptibility and mortality [1] |
| Gender | Controls for baseline physiological variability in patient response |

---

## Design Summary

### Models Used
- Linear Regression (predicting MAP — Mean Arterial Pressure)
- Random Forest Classifier (binary sepsis classification)
- MLP Neural Network (binary sepsis classification)

---

### Linear Regression — Baseline

The baseline regression model uses ordinary least squares (OLS) linear regression to predict a continuous physiological outcome (MAP) from the selected input features. The dataset is split randomly at the row level into training and testing sets with no feature scaling or regularisation applied, providing a simple reference point for evaluating design decisions. Performance was evaluated using mean squared error (MSE) and the coefficient of determination (R²).

### My Design Decision — Ridge Regularisation (L2)

Ridge regularisation was incorporated into the OLS model to improve stability and generalisation in the presence of correlated features and missing data. These conditions can cause standard OLS to produce large, unstable coefficients. Ridge regression modifies the objective function from minimising only the residual sum of squares to minimising:

**Σ(yᵢ − ŷᵢ)² + α·Σβⱼ²**

where α controls the strength of the penalty on coefficient magnitude. This penalty reduces the influence of any single feature and distributes weight more evenly across correlated inputs. An α value of 1.0 was selected to apply moderate regularisation, balancing improved stability with preservation of meaningful feature contributions.

---

### Non-Neural Network Classification — Baseline

The baseline classification model uses a Random Forest classifier with 50 decision trees and default hyperparameters to predict sepsis-positive or sepsis-negative outcomes. Performance was evaluated using recall, F1-score, and accuracy, chosen to provide insight into the model's ability to detect sepsis cases under class imbalance.

### My Design Decision — 5-Fold Cross-Validation

K-fold cross-validation (k=5) was applied to the Random Forest model to ensure a robust and reliable evaluation. Rather than relying on a single train-test split — which can produce results that depend heavily on the specific data partition — k-fold cross-validation repeatedly trains and tests the model on different subsets, allowing every data point to be used for both training and validation. This reduces evaluation bias and provides a more stable estimate of how the model will generalise to unseen cases, which is particularly important in a clinical setting where datasets may be limited and patient variability is high.

---

### Neural Network — Baseline

The baseline neural network model uses a multilayer perceptron (MLP) classifier with a single hidden layer (32 neurons) and sigmoid (logistic) activation. Input features are scaled using standardisation to improve training stability and convergence. Performance was evaluated using accuracy, recall, and F1-score.

### My Design Decision — ReLU Activation Function

The sigmoid activation function was replaced with ReLU (Rectified Linear Unit). The sigmoid function compresses inputs into a narrow range (0, 1) and causes gradients to become very small during backpropagation — a phenomenon known as the vanishing gradient problem — which slows or stalls learning. ReLU maintains stronger gradients for positive inputs, allowing the model to learn faster and more reliably [7]. This is particularly relevant for sepsis prediction, where relationships between clinical variables are often complex and nonlinear. All other parameters (hidden layer size, training data, preprocessing, and number of iterations) were held constant to isolate the effect of this change.

---

## Verification Test Plans

### Linear Regression

**Hypothesis:** Applying Ridge (L2) regularisation will improve model generalisation by reducing overfitting, resulting in a lower MSE and higher R² compared to the baseline OLS model.

**Acceptance Threshold:** The Ridge model must achieve a lower MSE and a higher R² than the baseline OLS model on the same test dataset. Specifically, R² must improve by at least 0.02 without an increase in MSE.

### Non-Neural Network Classification

**Hypothesis:** Applying 5-fold cross-validation will improve the reliability of model evaluation by reducing variance from a single data split, resulting in equal or improved F1-score and higher recall for the sepsis class compared to the baseline model.

**Acceptance Threshold:** The cross-validation approach must achieve a mean F1-score greater than or equal to the baseline model's F1-score, and improve recall for the sepsis class by at least 0.05, without a decrease in overall F1-score.

### Neural Network

**Hypothesis:** Replacing the sigmoid activation function with ReLU will improve model training efficiency and predictive performance by avoiding vanishing gradient effects, resulting in a higher F1-score and improved recall for the sepsis class compared to the baseline model.

**Acceptance Threshold:** The ReLU-based model must achieve a higher F1-score than the sigmoid baseline and improve recall for the sepsis class by at least 0.05, without a decrease in overall F1-score.

---

## Results

Full metric tables for all three models are available in [`results/results_summary.md`](../results/results_summary.md).

**Linear Regression:** PASS. The Ridge model achieved an R² improvement greater than 0.02 over the OLS baseline, and MSE did not increase. Ridge regularisation successfully reduced the influence of correlated features by distributing coefficient weight more evenly.

**Random Forest Classifier:** PASS. Recall for the sepsis-positive class improved from 0.04 in the baseline to 0.10 with 5-fold cross-validation, without reducing the overall F1-score. This confirmed that the baseline evaluation was subject to split-dependent variance, and that cross-validation produced a more reliable and honest estimate of model performance.

**Neural Network:** FAIL. Both the sigmoid and ReLU models produced a recall of 0 for the sepsis-positive class. The activation function was not the limiting factor; class imbalance caused both models to default entirely to the majority class, regardless of activation choice.

---

## Reflection

Two of my three design decisions met their acceptance thresholds. The linear regression model with Ridge regularisation and the random forest model with 5-fold cross-validation both demonstrated improved or stable performance, confirming that these design choices enhanced generalisation and evaluation reliability respectively.

The neural network design decision (ReLU activation) failed verification, as both models produced a recall of 0 for the positive sepsis class. This outcome suggests that the primary limitation was not the activation function itself, but rather underlying issues such as severe class imbalance, which caused the model to default to predicting the majority class. This highlights that improving a single component of a model does not guarantee better performance if fundamental data challenges are not addressed first.

From this project, I learned that handling class imbalance (e.g., through class weighting or resampling), selecting appropriate evaluation metrics such as recall, and critically evaluating what a failing result tells you are all essential steps in developing effective models for healthcare applications where failing to detect positive cases can have significant consequences.

---

## References

[1] C. A. Nasa, P. Nasa, and J. Nasa, "Risk factors for mortality in elderly and very elderly critically ill patients with sepsis: a prospective, observational, multicenter cohort study," *Annals of Intensive Care*, vol. 9, no. 26, 2019. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC6362175/

[2] W. Zheng, "Fever and hypothermia in systemic inflammation," in *Handbook of Clinical Neurology*, Elsevier, 2018. [Online]. Available: https://pubmed.ncbi.nlm.nih.gov/30459026/

[3] C. W. Seymour et al., "Assessment of Clinical Criteria for Sepsis: For the Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)," *JAMA*, vol. 315, no. 8, pp. 762–774, 2016. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC7929579/

[4] A. Morelli et al., "Tachycardia in Sepsis: Friend or Foe?," *ICU Management & Practice*, 2021. [Online]. Available: https://healthmanagement.org/c/icu/IssueArticle/tachycardia-in-sepsis-friend-or-foe

[5] L. Wu et al., "Prognostic value of PaO2/FiO2, SOFA and D-dimer in elderly patients with sepsis," *BMC Pulmonary Medicine*, 2022. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC9234855/

[6] J. A. Kellum, "Metabolic Acidosis in Sepsis," *Critical Care*, 2011. [Online]. Available: https://www.researchgate.net/publication/44637360_Metabolic_Acidosis_in_Sepsis

[7] Amanatullah, "Vanishing Gradient Problem in Deep Learning: Understanding, Intuition, and Solutions," *Medium*, 2023. [Online]. Available: https://medium.com/@amanatulla1606/vanishing-gradient-problem-in-deep-learning-understanding-intuition-and-solutions-da90ef4ecb54