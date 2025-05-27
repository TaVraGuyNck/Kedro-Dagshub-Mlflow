from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    evaluate_regression,
    evaluate_lr_regression,
    report_lr_coefficients,
    compute_shap_values,
    compute_cb_shap_values,
    plot_pca_2d
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Evaluate CatBoost
        node(
            func=evaluate_regression,
            inputs=["cb_model", "cb_preprocessor", "X_test_raw", "y_test"],
            outputs="cb_eval_metrics",
            name="cb_eval_metrics_node",
        ),
        node(
            func=compute_cb_shap_values,
            inputs=["cb_model", "cb_preprocessor", "X_test_raw"],
            outputs="shap_beeswarm_cb_plot",
            name="cb_shap_node",
        ),
        # Evaluate XGBoost
        node(
            func=evaluate_regression,
            inputs=["xgb_model", "xgb_preprocessor", "X_test_raw", "y_test"],
            outputs="xgb_eval_metrics",
            name="xgb_eval_metrics_node",
        ),
        node(
            func=compute_shap_values,
            inputs=["xgb_model", "xgb_preprocessor", "X_test_raw"],
            outputs="shap_beeswarm_xgb_plot",
            name="xgb_shap_node",
        ),
        # Evaluate Linear Ridge + PCA
        node(
            func=evaluate_lr_regression,
            inputs=["lr_best_model", "lr_preprocessor", "lr_best_pca_model", "X_test_raw", "y_test"],
            outputs="lr_eval_metrics",
            name="evaluate_lr_regression_node",
        ),
        node(
            func=report_lr_coefficients,
            inputs=["lr_best_model", "lr_best_pca_model"],
            outputs="lr_coefficients",
            name="report_lr_coefficients_node",
        ),

        node(
            func=plot_pca_2d,
            inputs=["lr_preprocessor", "lr_best_pca_model", "X_test_raw", "y_test"],  # or X_train_raw, y_train
            outputs="pca_2d_plot",
            name="plot_pca_2d_node",
        )
    ])
