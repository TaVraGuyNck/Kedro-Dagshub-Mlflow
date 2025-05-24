from kedro.framework.hooks import hook_impl
import logging

logger = logging.getLogger(__name__)

# Optional Spark hook, keep only if you're using Spark
class SparkHooks:
    @hook_impl
    def after_context_created(self, context):
        from pyspark.sql import SparkSession
        from pyspark import SparkConf

        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())
        spark = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

# Register only Spark hooks (safe to extend later)
HOOKS = ()
