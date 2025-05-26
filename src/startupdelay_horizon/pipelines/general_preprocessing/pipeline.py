from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess, impute_missing, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess,
            inputs=["project_raw", "programme_raw", "organization_raw"],
            outputs="preprocessed_table",
            name="preprocess_node",
        ),
        node(
            func=impute_missing,
            inputs="preprocessed_table",
            outputs="model_input_table",
            name="impute_missing_node",
        ),
        node(
            func=split_data,
            inputs="model_input_table",
            outputs=[
                "X_train_raw",
                "X_test_raw",
                "y_train",
                "y_test",
            ],
            name="split_data_node",
        ),
    ])
