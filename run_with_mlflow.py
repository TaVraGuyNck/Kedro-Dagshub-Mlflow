from dagshub import init
import mlflow
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from pathlib import Path
import sys


# Add src to path manually
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

# Replace with your Kedro project name (check src/<your_project_name>)
configure_project("startupdelay_horizon")

# Set DagsHub tracking URI
init(repo_owner="TaVraGuyNck", repo_name="Kedro-Dagshub-Mlflow", mlflow=True)

# Start MLflow run manually
with mlflow.start_run():
    project_path = Path.cwd()
    with KedroSession.create(project_path=project_path, env="local") as session:
        session.run(pipeline_name="__default__")
