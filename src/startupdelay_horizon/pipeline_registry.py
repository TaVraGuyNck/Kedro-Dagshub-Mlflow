from kedro.pipeline import Pipeline

from startupdelay_horizon.pipelines.general_preprocessing import (
    create_pipeline as create_general_preprocessing_pipeline,
)
from startupdelay_horizon.pipelines.xgb_preprocessing import (
    create_pipeline as create_xgb_preprocessing_pipeline,
)
from startupdelay_horizon.pipelines.cb_preprocessing import (
    create_pipeline as create_cb_preprocessing_pipeline,
)
from startupdelay_horizon.pipelines.model_training import (
    create_pipeline as create_model_training_pipeline,
)
from startupdelay_horizon.pipelines.evaluate_model.pipeline import (
    create_pipeline as create_evaluate_model_pipeline,
)
from startupdelay_horizon.pipelines.linear_preprocessing.pipeline import (
    create_pipeline as create_linear_preprocessing_pipeline
)

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "general_preprocessing": create_general_preprocessing_pipeline(),
        "linear_preprocessing": create_linear_preprocessing_pipeline(),
        "xgb_preprocessing": create_xgb_preprocessing_pipeline(),
        "cb_preprocessing": create_cb_preprocessing_pipeline(),
        "model_training": create_model_training_pipeline(),
        "evaluate_model": create_evaluate_model_pipeline(),
        "__default__": (
            create_general_preprocessing_pipeline()
            + create_linear_preprocessing_pipeline()
            + create_xgb_preprocessing_pipeline()
            + create_cb_preprocessing_pipeline()
            + create_model_training_pipeline()
            + create_evaluate_model_pipeline()
            
        ),
    }
