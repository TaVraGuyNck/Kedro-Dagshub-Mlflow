import os
import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def grid_search_xgb_model(X, y, param_grid, cv_folds):
    results = []
    best_score = -np.inf
    best_params = None
    best_model = None

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for params in ParameterGrid(param_grid):
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_train_fold, y_train_fold)
            score = model.score(X_val_fold, y_val_fold)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        results.append({"params": params, "cv_r2": avg_score})

        param_str = "_".join(f"{k}-{v}" for k, v in params.items())
        mlflow.log_metric(f"cv_r2_{param_str}", avg_score)

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # Fit the best model on the full training data
    best_model = XGBRegressor(**best_params).fit(X, y)
    os.makedirs("data/06_models", exist_ok=True)
    joblib.dump(best_model, "data/06_models/xgb_best_model.pkl")

    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_r2", best_score)
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")

    # Save grid results as artifact
    os.makedirs("data/08_reporting", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_parquet = "data/08_reporting/xgb_grid_results.parquet"
    results_df.to_parquet(results_parquet)
    mlflow.log_artifact(results_parquet)

    return results_df, best_model

def grid_search_cb_model(X, y, cat_features, param_grid, cv_folds):
    results = []
    best_score = -np.inf
    best_params = None
    best_model = None

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for params in ParameterGrid(param_grid):
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostRegressor(**params, verbose=0)
            model.fit(X_train_fold, y_train_fold, cat_features=cat_features)
            score = model.score(X_val_fold, y_val_fold)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        results.append({"params": params, "cv_r2": avg_score})

        param_str = "_".join(f"{k}-{v}" for k, v in params.items())
        mlflow.log_metric(f"cv_r2_{param_str}", avg_score)

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # Fit the best model on the full training data
    best_model = CatBoostRegressor(**best_params, verbose=0).fit(X, y, cat_features=cat_features)
    os.makedirs("data/06_models", exist_ok=True)
    joblib.dump(best_model, "data/06_models/cb_best_model.pkl")

    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_r2", best_score)
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")

    # Save grid results as artifact
    os.makedirs("data/08_reporting", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_parquet = "data/08_reporting/cb_grid_results.parquet"
    results_df.to_parquet(results_parquet)
    mlflow.log_artifact(results_parquet)

    return results_df, best_model
