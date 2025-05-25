import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from startupdelay_horizon.metrics import regression_metrics
import mlflow
import shap
import matplotlib.pyplot as plt
import os

def evaluate_xgb_model(
    xgb_model: XGBRegressor,
    xgb_X_test: pd.DataFrame,
    xgb_y_test: pd.Series
) -> pd.DataFrame:
    y_pred = xgb_model.predict(xgb_X_test)
    
    # Compute metrics (assuming this is a custom util that returns a dict)
    metrics = regression_metrics(xgb_y_test, y_pred)

    # Log each metric to MLflow
    for key, value in metrics.items():
        mlflow.log_metric(f"xgb_{key}", value)

    return pd.DataFrame([metrics])


def evaluate_cb_point_model(
    cb_model: CatBoostRegressor,
    cb_X_test: pd.DataFrame,
    cb_y_test: pd.Series
) -> pd.DataFrame:
    y_pred = cb_model.predict(cb_X_test)

    # Compute regression metrics using your custom utility
    metrics = regression_metrics(cb_y_test, y_pred)

    # Log each metric to MLflow
    for key, value in metrics.items():
        mlflow.log_metric(f"cb_{key}", value)

    return pd.DataFrame([metrics])

def compute_shap_values(model, preprocessor, X_test_raw):
    # Apply preprocessing
    X_test = preprocessor.transform(X_test_raw)

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Generate beeswarm plot
    plt.figure()
    shap_plot = shap.plots.beeswarm(shap_values, show=False)

    # Log to MLflow (optional)
    
    plt.savefig("data/08_reporting/shap_beeswarm_xgb.png", bbox_inches="tight")
    mlflow.log_artifact("data/08_reporting/shap_beeswarm_xgb.png")
    
    return plt.gcf()  

def compute_cb_shap_values(model, preprocessor, X_test_raw):
    X_test = preprocessor.transform(X_test_raw)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("data/08_reporting/shap_beeswarm_cb.png", bbox_inches="tight")
    mlflow.log_artifact("data/08_reporting/shap_beeswarm_cb.png")

    return plt.gcf()

def transform_cb_X_test(cb_preprocessor, cb_X_test_raw):
    return cb_preprocessor.transform(cb_X_test_raw)



