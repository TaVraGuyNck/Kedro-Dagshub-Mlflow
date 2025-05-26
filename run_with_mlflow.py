import dagshub
dagshub.init(repo_owner='tavraguynck',
             repo_name='Kedro-Dagshub-Mlflow',
             mlflow=True)

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from pathlib import Path
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

configure_project("startupdelay_horizon")

project_path = Path.cwd()
with KedroSession.create(project_path=project_path, env="local") as session:
    session.run(pipeline_name="__default__")