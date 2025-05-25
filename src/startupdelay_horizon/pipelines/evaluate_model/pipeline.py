from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    evaluate_xgb_model,
    evaluate_cb_point_model,
    compute_shap_values,
    transform_cb_X_test,
    compute_cb_shap_values  # <-- NEW
)

def create_pipeline() -> Pipeline:
    return pipeline([
        # XGBoost evaluation
        node(
            func=evaluate_xgb_model,
            inputs=["xgb_model", "xgb_X_test", "xgb_y_test"],
            outputs="metrics_xgb",
            name="evaluate_xgb_node",
        ),

        # CatBoost test preprocessing
        node(
            func=transform_cb_X_test,
            inputs=["cb_preprocessor", "cb_X_test_raw"],
            outputs="cb_X_test_eval",
            name="transform_cb_X_test_node",
        ),

        # CatBoost evaluation
        node(
            func=evaluate_cb_point_model,
            inputs=["cb_model", "cb_X_test_eval", "cb_y_test"],
            outputs="metrics_cb",
            name="evaluate_cb_point_node",
        ),

        # SHAP for XGBoost
        node(
            func=compute_shap_values,
            inputs=["xgb_model", "xgb_preprocessor", "xgb_X_test_raw"],
            outputs="shap_beeswarm_plot",
            name="xgb_shap_node",
        ),

        # SHAP for CatBoost
        node(
            func=compute_cb_shap_values,
            inputs=["cb_model", "cb_preprocessor", "cb_X_test_raw"],
            outputs="shap_beeswarm_cb_plot",
            name="cb_shap_node",
        ),
        
        # Parameter grid xgb model
        node(
            func=grid_search_xgb_model,
            inputs=["xgb_X_train", "xgb_y_train", "xgb_X_test", "xgb_y_test", "params:xgb_param_grid"],
            outputs="xgb_grid_search_results",
            name="grid_search_xgb_model_node"
        )
    ])
