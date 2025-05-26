import pandas as pd
import numpy as np
import mlflow
import shap
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def compute_shap_values(model, preprocessor, X_test_raw):
    # Apply preprocessing
    X_test = preprocessor.transform(X_test_raw)

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Compute mean absolute SHAP values per feature
    shap_df = pd.DataFrame(
        np.abs(shap_values.values), columns=X_test.columns
    ).mean().sort_values(ascending=True)

    # Bar plot (horizontal)
    fig, ax = plt.subplots(figsize=(8, max(6, len(shap_df) * 0.3)))
    shap_df.plot.barh(ax=ax)
    ax.set_title("Mean |SHAP| values (XGBoost)")
    ax.set_xlabel("Mean absolute SHAP value")
    plt.tight_layout()

    # Log to MLflow
    output_path = "data/08_reporting/shap_xgb_summary.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")

    mlflow.set_tag("pipeline_step", "shap_xgb")
    mlflow.set_tag("model_type", "xgboost")
    mlflow.set_tag("explanation_type", "bar_plot")
    mlflow.log_artifact(output_path)

    return fig


def compute_cb_shap_values(model, preprocessor, X_test_raw):
    # Apply preprocessing
    X_test = preprocessor.transform(X_test_raw)

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Compute mean absolute SHAP values per feature
    shap_df = pd.DataFrame(
        np.abs(shap_values.values), columns=X_test.columns
    ).mean().sort_values(ascending=True)

    # Bar plot (horizontal)
    fig, ax = plt.subplots(figsize=(8, max(6, len(shap_df) * 0.3)))
    shap_df.plot.barh(ax=ax)
    ax.set_title("Mean |SHAP| values (CatBoost)")
    ax.set_xlabel("Mean absolute SHAP value")
    plt.tight_layout()

    # Log to MLflow
    output_path = "data/08_reporting/shap_cb_summary.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")

    mlflow.set_tag("pipeline_step", "shap_cb")
    mlflow.set_tag("model_type", "catboost")
    mlflow.set_tag("explanation_type", "bar_plot")
    mlflow.log_artifact(output_path)

    return fig

def evaluate_regression(model, preprocessor, X_test_raw, y_test):
    # Transform raw test input
    X_test = preprocessor.transform(X_test_raw)
    y_pred = model.predict(X_test)

    # Compute metrics
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

    return metrics