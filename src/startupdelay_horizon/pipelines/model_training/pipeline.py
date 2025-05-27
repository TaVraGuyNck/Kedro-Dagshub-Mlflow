from kedro.pipeline import Pipeline, node, pipeline
from .nodes import grid_search_cb_model, grid_search_xgb_model, grid_search_ridge_pca_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=grid_search_cb_model,
            inputs=[
                "cb_X_train_transformed",
                "y_train_no_outliers",
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
                "y_train_no_outliers",
                "params:xgb_param_grid",
                "params:cv_folds",
            ],
            outputs=["xgb_grid_search_results", "xgb_model"],  
            name="grid_search_xgb_model_node",
        ),

        node(
            func=grid_search_ridge_pca_model,
            inputs=[
                "lr_X_train_transformed",  # Should be numpy array (after fit_transform_features)
                "y_train_no_outliers",                # Should be 1d or DataFrame with 1 column
                "params:ridge_pca_param_grid",
                "params:ridge_cv_folds",
            ],
            outputs=["lr_grid_results", "lr_best_model", "lr_best_pca_model"],
            name="grid_search_ridge_pca_model_node",
        ),
    ])



