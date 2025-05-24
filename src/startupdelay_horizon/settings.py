"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

import os
from pathlib import Path
from dotenv import load_dotenv
import warnings

# ✅ Load .env from the project root (above `src/`)
project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path)
    if not os.getenv("MLFLOW_TRACKING_URI"):
        warnings.warn(
            "⚠️ .env file loaded but MLFLOW_TRACKING_URI is not set. "
            "MLflow may fall back to the default local store."
        )
else:
    warnings.warn("⚠️ .env file not found. Environment variables (like MLFLOW_TRACKING_URI) will not be loaded.")

# Optional debug print
# print("MLFLOW_TRACKING_URI =", os.getenv("MLFLOW_TRACKING_URI"))

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
