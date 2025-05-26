from kedro.pipeline import Pipeline, node, pipeline
from .nodes import apply_cb_transformer

def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=apply_cb_transformer,
            inputs=["X_train_raw", "X_test_raw"],
            outputs=[
                "cb_X_train_transformed",
                "cb_X_test_transformed",
                "cb_cat_features",
                "cb_preprocessor"
            ],
            name="cb_preprocessing_node"
        )
    ])

