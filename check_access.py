from dagshub import dagshub_logger, init
import mlflow

init(repo_owner="TaVraGuyNck", repo_name="Kedro-Dagshub-Mlflow", mlflow=True)

with mlflow.start_run(experiment_id="0"):
    mlflow.log_param("test", 1)