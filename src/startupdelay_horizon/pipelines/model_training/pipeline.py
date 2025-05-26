from kedro.pipeline import Pipeline, node, pipeline
from .nodes import grid_search_cb_model, grid_search_xgb_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=grid_search_cb_model,
            inputs=[
                "cb_X_train_transformed",
                "y_train",
                "cb_cat_features",
                "params:cb_param_grid",
                "params:cv_folds",
            ],
            outputs=["cb_grid_search_results", "cb_model"],  
            name="grid_search_cb_model_node",
        ),
        node(
            func=grid_search_xgb_model,
            inputs=[
                "xgb_X_train_transformed",
                "y_train",
                "params:xgb_param_grid",
                "params:cv_folds",
            ],
            outputs=["xgb_grid_search_results", "xgb_model"],  
            name="grid_search_xgb_model_node",
        ),
    ])
