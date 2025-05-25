from kedro.pipeline import Pipeline, node, pipeline
import mlflow
from .nodes import (
    split_catboost_data,
    apply_cb_transformer,
    train_catboost_model,
)

def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=split_catboost_data,
            inputs="model_input_table",
            outputs=[
                "cb_X_train_raw",
                "cb_X_test_raw",
                "cb_y_train",
                "cb_y_test",
            ],
            name="cb_split_data_node",
        ),
        node(
            func=apply_cb_transformer,
            inputs=["cb_X_train_raw", "cb_X_test_raw"],
            outputs=[
                "cb_X_train",
                "cb_X_test",
                "cb_cat_features",
                "cb_preprocessor"
            ],
            name="cb_preprocessing_node",
        ),
        node(
            func=train_catboost_model,
            inputs=[
                "cb_X_train",
                "cb_y_train",
                "cb_X_test",
                "cb_y_test",
                "params:cb_params",
                "cb_cat_features"
            ],
            outputs=["cb_model", "cb_model_mlflow"],
            name="cb_training_node",
        ),
    ])
