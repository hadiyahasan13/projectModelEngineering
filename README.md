# Project Model Engineering
# 🔍 Credit Card Fraud Detection — Model Engineering

**Course:** Model Engineering | M.Sc. Data Science, IU International University of Applied Sciences
**Author:** Hadiya Hasan | [GitHub: hadiyahasan13](https://github.com/hadiyahasan13)
**Submitted:** April 2025

---

## Overview

This project tackles one of the most critical challenges in financial data science: **detecting fraudulent credit card transactions in a highly imbalanced dataset**. Using the ULB Credit Card Fraud Detection dataset (284,807 transactions, only 0.172% fraud), this case study implements, optimizes, and compares two machine learning classifiers — **Random Forest** and **XGBoost** — across multiple resampling strategies and hyperparameter tuning setups.

The project focuses on real-world constraints: class imbalance, pipeline design to prevent data leakage, and choosing evaluation metrics that actually matter in fraud detection (Precision, Recall, F1-Score).

---

## Key Results

| Model | Configuration | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | Baseline | 0.87 | 0.76 | 0.81 |
| Random Forest | + Hyperparameter Tuning | 0.94 | 0.77 | 0.85 |
| **Random Forest** | **+ Random Undersampling (RUS)** | **1.00** | **0.97** | **0.98** |
| Random Forest | + SMOTE | 0.80 | 0.82 | 0.81 |
| XGBoost | Baseline | 0.94 | 0.78 | 0.85 |
| XGBoost | + Hyperparameter Tuning | 0.95 | 0.78 | 0.86 |
| XGBoost | + SMOTE | 0.79 | 0.83 | 0.81 |

> **Best performer:** Random Forest with Random Undersampling — F1-Score of **0.98** on the fraud (minority) class.

---

## Dataset

- **Source:** [Kaggle — ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions over 2 days (September 2013, European cardholders)
- **Features:** 28 PCA-transformed features (V1–V28) + `Time`, `Amount`, `Class`
- **Class imbalance:** 492 fraud cases (0.172%) vs. 284,315 legitimate transactions

---

## Methodology

### Preprocessing
- Checked and confirmed no missing values
- Removed duplicate rows
- Normalized the `Amount` column using `StandardScaler`
- Dropped the `Time` column (no predictive correlation with fraud class)

### Handling Class Imbalance
Two strategies were compared across both classifiers:

- **Random Undersampling (RUS):** Majority class reduced to match minority. Applied inside CV folds to prevent leakage. Used with Random Forest.
- **SMOTE (Synthetic Minority Oversampling Technique):** Synthetic fraud samples generated via interpolation. Applied with both RF and XGBoost for comparison.

### Cross-Validation
- **Stratified 5-Fold Cross-Validation** used throughout — ensures class proportions are preserved in every fold, critical for imbalanced data.

### Hyperparameter Optimization
- **RandomizedSearchCV** used for both models (10 iterations) — faster than grid search while exploring broad parameter space effectively.
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` (RF); `learning_rate`, `subsample`, `colsample_bytree` (XGBoost).

### Pipelines
All preprocessing (resampling) and model training steps were wrapped in `imblearn.pipeline.Pipeline` to ensure:
- No data leakage between train and validation folds
- Reproducibility across CV folds
- Clean, modular code structure

Three pipelines were implemented:

```
Pipeline A: RandomUnderSampler → RandomForestClassifier
Pipeline B: SMOTE → RandomForestClassifier
Pipeline C: SMOTE → XGBClassifier
```

All wrapped with `RandomizedSearchCV` for end-to-end hyperparameter tuning.

---

## Models

### Random Forest
Ensemble of decision trees trained on bootstrap samples. Handles feature selection implicitly, robust to class imbalance when paired with resampling. Achieves best fraud detection performance with RUS (F1 = 0.98), though overfitting risk exists due to aggressive undersampling.

### XGBoost
Gradient boosted decision trees with built-in regularization. Well-suited for large, imbalanced datasets. With SMOTE, achieves higher recall (0.83) than RF+SMOTE, with faster training time — making it more practical for real-world deployment.

---

## Evaluation Metrics

Standard accuracy is misleading for imbalanced datasets. This project uses:

- **Precision** — of all transactions flagged as fraud, how many actually were?
- **Recall** — of all actual frauds, how many did the model catch?
- **F1-Score** — harmonic mean of precision and recall; the primary metric here

Recall is particularly important in fraud detection — missing a fraud (false negative) is typically more costly than a false alarm.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| scikit-learn | ML models, pipelines, CV, metrics |
| imbalanced-learn | SMOTE, RandomUnderSampler, ImbPipeline |
| XGBoost | XGBClassifier |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |
| Jupyter Notebook | Development environment |

---

## Repository Structure

```
projectModelEngineering/
│
├── modelengg.ipynb       # Full notebook: EDA, preprocessing, RF + XGBoost experiments
├── rf.ipynb              # Focused Random Forest implementation with RUS and SMOTE
└── README.md
```

---

## Key Takeaways

- **Pipeline design is non-negotiable** when working with resampling — applying SMOTE before splitting causes data leakage and inflated metrics.
- **Stratified K-Fold** is essential for imbalanced classification; standard K-Fold can produce folds with no fraud samples at all.
- **Random Undersampling + RF** achieved the best fraud recall but risks overfitting; Stratified CV mitigates this.
- **SMOTE + XGBoost** is the more production-realistic combination — balanced performance, faster training, and better generalization.
- The "right" model depends on the deployment context: if false negatives are very costly (real-time fraud prevention), maximize recall. If false positives are costly (customer experience), maximize precision.
