from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    split_xgboost_data,
    apply_xgb_transformer,
    train_xgboost_model,
    )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_xgboost_data,
            inputs="model_input_table",
            outputs=["xgb_X_train_raw", "xgb_X_test_raw", "xgb_y_train", "xgb_y_test"],
            name="xgb_split_data_node"
        ),
        node(
            func=apply_xgb_transformer,
            inputs=["xgb_X_train_raw", "xgb_X_test_raw"],
            outputs=["xgb_X_train", "xgb_X_test", "xgb_preprocessor"],
            name="xgb_preprocessing_node"
        ),
        node(
            func=train_xgboost_model,
            inputs=["xgb_X_train", "xgb_y_train", "params:xgb_params", "xgb_preprocessor"],
            outputs=["xgb_model", "xgb_model_mlflow"],
            name="xgb_training_node"
        )

    ])