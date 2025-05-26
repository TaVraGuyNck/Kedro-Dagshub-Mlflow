from kedro.framework.hooks import hook_impl
import mlflow

class MLflowPipelineHook:
    def __init__(self):
        self.active_run = None

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        # Here is where a pipeline-wide MLflow run starts!
        self.active_run = mlflow.start_run(run_name=run_params["pipeline_name"])
        mlflow.set_tag("kedro.pipeline", run_params["pipeline_name"])

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    @hook_impl
    def on_pipeline_error(self, error, run_params, pipeline, catalog):
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
