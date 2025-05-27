import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_lr_regression(model, preprocessor, pca, X_test_raw, y_test):
    """
    Evaluates the best linear Ridge model on the test set.
    Applies pre-fitted preprocessor (scaling+encoding) and PCA, then predicts.
    Returns a DataFrame of metrics (1 row).
    """
    # 1. Apply scaling/dummy encoding
    X_test_trans = preprocessor.transform(X_test_raw)
    # 2. Apply PCA
    X_test_pca = pca.transform(X_test_trans)
    # 3. Predict
    y_pred = model.predict(X_test_pca)
    # 4. Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    mlflow.set_tag("pipeline_step", "evaluate_lr_model")
    mlflow.log_metric("lr_mae", mae)
    mlflow.log_metric("lr_rmse", rmse)
    mlflow.log_metric("lr_r2", r2)
    metrics_df = pd.DataFrame([metrics])  # single row DataFrame
    return metrics_df

def report_lr_coefficients(model, pca=None):
    """
    Returns a DataFrame of Ridge coefficients.
    If PCA was used, coefficients correspond to principal components.
    """
    if pca is not None:
        pc_labels = [f"PC{i+1}" for i in range(pca.n_components_)]
        coefs = pd.Series(model.coef_, index=pc_labels)
    else:
        coefs = pd.Series(model.coef_)
    coefs_df = coefs.sort_values(ascending=False).to_frame(name="coefficient")
    coefs_df.index.name = "feature"
    return coefs_df

def plot_pca_2d(preprocessor, pca, X_raw, y=None):
    """
    Plots the first two principal components of X_raw after preprocessing and PCA.
    Optionally colors by y (if given).
    Returns the Matplotlib Figure for Kedro's MatplotlibWriter.
    """
    X_trans = preprocessor.transform(X_raw)
    X_pca = pca.transform(X_trans)
    fig, ax = plt.subplots(figsize=(8, 6))
    if y is not None:
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=np.ravel(y), cmap='viridis', alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Target/label")  # <-- the FIX!
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("First two principal components")
    plt.tight_layout()
    return fig

def compute_shap_values(model, preprocessor, X_test_raw):
    """
    Computes and plots mean absolute SHAP values for XGBoost model.
    Returns the Matplotlib Figure object (to be handled by MatplotlibWriter or as artifact).
    """
    X_test = preprocessor.transform(X_test_raw)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap_df = pd.DataFrame(
        np.abs(shap_values.values), columns=X_test.columns
    ).mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, len(shap_df) * 0.3)))
    shap_df.plot.barh(ax=ax)
    ax.set_title("Mean |SHAP| values (XGBoost)")
    ax.set_xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    output_path = "data/08_reporting/shap_xgb_summary.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    mlflow.set_tag("pipeline_step", "shap_xgb")
    mlflow.set_tag("model_type", "xgboost")
    mlflow.set_tag("explanation_type", "bar_plot")
    mlflow.log_artifact(output_path)
    return fig

def compute_cb_shap_values(model, preprocessor, X_test_raw):
    """
    Computes and plots mean absolute SHAP values for CatBoost model.
    Returns the Matplotlib Figure object (to be handled by MatplotlibWriter or as artifact).
    """
    X_test = preprocessor.transform(X_test_raw)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap_df = pd.DataFrame(
        np.abs(shap_values.values), columns=X_test.columns
    ).mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, len(shap_df) * 0.3)))
    shap_df.plot.barh(ax=ax)
    ax.set_title("Mean |SHAP| values (CatBoost)")
    ax.set_xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    output_path = "data/08_reporting/shap_cb_summary.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    mlflow.set_tag("pipeline_step", "shap_cb")
    mlflow.set_tag("model_type", "catboost")
    mlflow.set_tag("explanation_type", "bar_plot")
    mlflow.log_artifact(output_path)
    return fig

def evaluate_regression(model, preprocessor, X_test_raw, y_test):
    """
    Evaluates a regression model (e.g., XGBoost, CatBoost) on test set.
    Returns a DataFrame of metrics (1 row).
    """
    X_test = preprocessor.transform(X_test_raw)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
    mlflow.set_tag("pipeline_step", "evaluate_model")
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    metrics_df = pd.DataFrame([metrics])
    return metrics_df
