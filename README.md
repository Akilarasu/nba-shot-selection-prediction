# NBA Shot Selection Prediction

## Overview

This project builds a machine learning model to predict whether an NBA shot attempt will be successful using historical shot data.

The project demonstrates the full machine learning workflow including data preprocessing, feature engineering, semi-supervised learning, model training, and evaluation.

## Problem Statement

Shot selection significantly impacts team performance. Predicting shot success can help teams optimize offensive strategy and player decision making.

## Methodology

Steps followed in the project:

1. Data preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Semi-supervised learning
5. Model training and comparison
6. Model evaluation

## Models Used

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Why Semi-Supervised Learning

The dataset contained missing labels for shot outcomes.

Semi-supervised learning allowed the model to leverage both labeled and unlabeled data by generating pseudo labels.

## Why SMOTE Was Not Used

Synthetic oversampling can introduce noise when combined with pseudo-labeled data.

Instead, tree-based models with class imbalance handling were used.

## Why Outliers Were Not Removed

Extreme shot distances represent real basketball scenarios.

Removing them would bias the model and reduce real-world applicability.

