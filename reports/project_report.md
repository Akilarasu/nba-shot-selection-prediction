# NBA Shot Selection Prediction

## Problem Statement

Shot selection plays a critical role in basketball performance. The goal of this project is to predict whether a shot attempt will be successful based on historical shot data.

## Challenges

- Missing target labels
- Imbalanced classes
- High dimensional spatial features

## Approach

A semi-supervised learning approach was used to leverage both labeled and unlabeled data.

Steps:

1. Data preprocessing
2. Feature engineering
3. Semi-supervised learning
4. Model training and evaluation

Models used:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Model Evaluation

Evaluation metrics:

- Precision
- Recall
- F1 Score

These metrics were selected due to class imbalance.

## Why SMOTE Was Not Used

SMOTE generates synthetic samples which can introduce noise when combined with pseudo-labeled data from semi-supervised learning.

Instead, tree-based models were used with class imbalance handling.

## Why Outliers Were Not Removed

In basketball shot data, extreme shot distances represent real game events.

Removing them would distort the true distribution of shot attempts.

## Business Impact

This model can help teams and analysts:

- Identify high probability shooting zones
- Improve offensive decision making
- Analyze player shot efficiency
- Reduce low probability shot attempts
