import os
import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

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
        mlflow.log_metric(f"xgb_cv_r2_{param_str}", avg_score)

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # Fit the best model on the full training data
    best_model = XGBRegressor(**best_params).fit(X, y)
    os.makedirs("data/06_models", exist_ok=True)
    joblib.dump(best_model, "data/06_models/xgb_best_model.pkl")

    mlflow.log_params({f"xgb_best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("xgb_best_cv_r2", best_score)
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
        results.append({"params": params, "cb_cv_r2": avg_score})

        param_str = "_".join(f"{k}-{v}" for k, v in params.items())
        mlflow.log_metric(f"cb_cv_r2_{param_str}", avg_score)

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # Fit the best model on the full training data
    best_model = CatBoostRegressor(**best_params, verbose=0).fit(X, y, cat_features=cat_features)
    os.makedirs("data/06_models", exist_ok=True)
    joblib.dump(best_model, "data/06_models/cb_best_model.pkl")

    mlflow.log_params({f"cb_best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("cb_best_cv_r2", best_score)
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")

    # Save grid results as artifact
    os.makedirs("data/08_reporting", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_parquet = "data/08_reporting/cb_grid_results.parquet"
    results_df.to_parquet(results_parquet)
    mlflow.log_artifact(results_parquet)

    return results_df, best_model

def grid_search_ridge_pca_model(X, y, param_grid, cv_folds):
    """
    Grid search over Ridge alpha and PCA n_components.
    X: pd.DataFrame
    y: pd.Series or pd.DataFrame with one column
    """
    results = []
    best_score = -np.inf
    best_params = None
    best_model = None
    best_pca_model = None

    # Make sure y is a Series
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    # don't flatten to numpy!

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for params in ParameterGrid(param_grid):
        n_components = params["n_components"]
        alpha = params["alpha"]
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            pca = PCA(n_components=n_components, random_state=42)
            X_train_fold_pca = pca.fit_transform(X_train_fold)
            X_val_fold_pca = pca.transform(X_val_fold)

            model = Ridge(alpha=alpha)
            model.fit(X_train_fold_pca, y_train_fold)
            score = model.score(X_val_fold_pca, y_val_fold)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        results.append({"params": params, "lr_cv_r2": avg_score})

        param_str = "_".join(f"{k}-{v}" for k, v in params.items())
        mlflow.log_metric(f"lr_cv_r2_{param_str}", avg_score)

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    # Fit the best model and PCA on the full training set
    n_components = best_params["n_components"]
    alpha = best_params["alpha"]
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    best_model = Ridge(alpha=alpha).fit(X_pca, y)
    best_pca_model = pca

    os.makedirs("data/06_models", exist_ok=True)
    joblib.dump(best_model, "data/06_models/lr_best_model.pkl")
    joblib.dump(best_pca_model, "data/06_models/lr_best_pca.pkl")

    mlflow.log_params({f"lr_best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("lr_best_cv_r2", best_score)
    mlflow.sklearn.log_model(best_model, artifact_path="best_model")

    # Save grid results
    os.makedirs("data/08_reporting", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_parquet = "data/08_reporting/lr_grid_results.parquet"
    results_df.to_parquet(results_parquet)
    mlflow.log_artifact(results_parquet)

    return results_df, best_model, best_pca_model
