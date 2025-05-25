from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import preprocess, impute_missing

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess,
            inputs=["project_raw", "programme_raw", "organization_raw"],
            outputs="preprocessed_table",
            name="preprocess_node"
        ),
        node(
            func=impute_missing,
            inputs="preprocessed_table",
            outputs="model_input_table",
            name="impute_missing_node"
        )
    ])
