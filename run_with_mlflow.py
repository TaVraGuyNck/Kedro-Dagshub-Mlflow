from dagshub import init
import mlflow
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from pathlib import Path
import sys
import logging

# Add src to path manually
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

# Replace with your Kedro project name
configure_project("startupdelay_horizon")

# Set DagsHub tracking URI
try:
    init(repo_owner="TaVraGuyNck", repo_name="Kedro-Dagshub-Mlflow", mlflow=True)
except Exception as e:
    logging.warning(f"[DagsHub init failed] {e}")

# Try to start MLflow run
try:
    with mlflow.start_run():
        run_pipeline = True
except Exception as e:
    logging.warning(f"[MLflow start_run failed] {e}")
    run_pipeline = False

# Run Kedro pipeline, with or without MLflow
project_path = Path.cwd()
with KedroSession.create(project_path=project_path, env="local") as session:
    if run_pipeline:
        session.run(pipeline_name="__default__")
    else:
        # Still run without explicit MLflow context
        session.run(pipeline_name="__default__")
