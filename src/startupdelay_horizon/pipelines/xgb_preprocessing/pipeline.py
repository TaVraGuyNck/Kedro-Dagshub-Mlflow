from kedro.pipeline import Pipeline, node, pipeline
from .nodes import apply_xgb_transformer

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=apply_xgb_transformer,
            inputs=["X_train_raw", "X_test_raw"],
            outputs=[
                "xgb_X_train_transformed",
                "xgb_X_test_transformed",
                "xgb_preprocessor"
            ],
            name="xgb_preprocessing_node"
        ),
    ])