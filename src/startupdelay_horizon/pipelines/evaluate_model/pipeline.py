from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    compute_cb_shap_values,
    compute_shap_values,
    evaluate_regression,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # 📊 CATBOOST: SHAP
        node(
            func=compute_cb_shap_values,
            inputs=[
                "cb_model",
                "cb_preprocessor",
                "X_test_raw"
            ],
            outputs="shap_beeswarm_cb_plot",
            name="cb_shap_node",
        ),

        # 📊 XGBOOST: SHAP
        node(
            func=compute_shap_values,
            inputs=[
                "xgb_model",
                "xgb_preprocessor",
                "X_test_raw"
            ],
            outputs="shap_beeswarm_xgb_plot",
            name="xgb_shap_node",
        ),

        # 📈 CATBOOST: EVALUATION METRICS
        node(
            func=evaluate_regression,
            inputs=[
                "cb_model",
                "cb_preprocessor",
                "X_test_raw",
                "y_test"
            ],
            outputs="cb_eval_metrics",
            name="cb_eval_metrics_node",
        ),

        # 📈 XGBOOST: EVALUATION METRICS
        node(
            func=evaluate_regression,
            inputs=[
                "xgb_model",
                "xgb_preprocessor",
                "X_test_raw",
                "y_test"
            ],
            outputs="xgb_eval_metrics",
            name="xgb_eval_metrics_node",
        ),
    ])
