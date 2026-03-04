import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from scipy.stats import randint, uniform


#############################################
# Logistic Regression
#############################################

def train_logistic_regression(X_train, y_train):

    lr = LogisticRegression(max_iter=1000)

    lr.fit(X_train, y_train)

    joblib.dump(lr, "../models/logistic_regression_model.pkl")

    return lr


#############################################
# Random Forest with GridSearch
#############################################

def train_random_forest(X_train, y_train):

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='accuracy'
    )

    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_

    joblib.dump(best_rf, "../models/random_forest_model.pkl")

    return best_rf


#############################################
# XGBoost with Random Search
#############################################

def train_xgboost(X_train, y_train):

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False
    )

    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': uniform(0, 5),
        'min_child_weight': randint(1, 10)
    }

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=50,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    joblib.dump(best_model, "../models/xgboost_model.pkl")

    return best_model


#############################################
# LightGBM with Random Search
#############################################

def train_lightgbm(X_train, y_train):

    lgb = LGBMClassifier(
        objective='binary',
        random_state=42,
        n_jobs=-1
    )

    param_dist_lgb = {
        'n_estimators': randint(300, 1500),
        'max_depth': randint(3, 12),
        'learning_rate': uniform(0.01, 0.15),
        'num_leaves': randint(20, 150),
        'min_child_samples': randint(10, 80),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    lgb_search = RandomizedSearchCV(
        lgb,
        param_distributions=param_dist_lgb,
        n_iter=50,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    lgb_search.fit(X_train, y_train)

    best_lgb = lgb_search.best_estimator_

    joblib.dump(best_lgb, "../models/lightgbm_model.pkl")

    return best_lgb
