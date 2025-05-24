from kedro.framework.hooks import hook_impl
from kedro_mlflow.framework.hooks import MlflowHook
from pyspark.sql import SparkSession
from pyspark import SparkConf
import mlflow
import warnings


class SparkHooks:
    """Configure Spark session after Kedro context is created."""

    @hook_impl
    def after_context_created(self, context):
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())
        spark = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")


class CustomMlflowHook(MlflowHook):
    @hook_impl
    def after_context_created(self, context):
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            warnings.warn("MLFLOW_TRACKING_URI not set — using default local MLflow store.")
        super().after_context_created(context)


class MLflowTagHook:
    """
    Add custom MLflow tags before pipeline run.
    """

    @hook_impl
    def before_pipeline_run(self, run_params):
        mlflow.set_tag("experiment", "baseline")
        mlflow.set_tag("run_type", "catboost")  # Change if needed
        mlflow.set_tag("owner", "tanguy")
        mlflow.set_tag("pipeline", run_params.get("pipeline_name", "__default__"))
        mlflow.set_tag("run_id", run_params["run_id"])


# Register hooks
mlflow_hook = CustomMlflowHook()
HOOKS = (SparkHooks(), mlflow_hook, MLflowTagHook())
