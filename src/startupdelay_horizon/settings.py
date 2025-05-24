"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

import os
from pathlib import Path
from dotenv import load_dotenv

# ✅ Load .env from the project root (above `src/`)
project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path)

# Optional debug print to verify it worked:
# print("MLFLOW URI:", os.getenv("MLFLOW_TRACKING_URI"))

# Class that manages how configuration is loaded.
from kedro.config import OmegaConfigLoader  # noqa: E402

CONFIG_LOADER_CLASS = OmegaConfigLoader

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
    }
}

# Other optional settings (hooks, context, catalog) can be defined here as needed.
