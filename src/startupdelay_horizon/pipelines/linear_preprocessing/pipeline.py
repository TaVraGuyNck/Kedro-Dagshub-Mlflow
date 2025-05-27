from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fit_transform_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fit_transform_features,
            inputs=["X_train_no_outliers", "X_test_raw"],
            outputs=["lr_X_train_transformed", "lr_X_test_transformed", "lr_preprocessor"],
            name="fit_transform_features_node",
        )
    ])
