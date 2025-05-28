import joblib
from kedro.framework.context import KedroContext

# Replace this with the actual name of your Kedro package (check pyproject.toml)
context = KedroContext(package_name="your_package_name")

# Load the catalog and extract the preprocessor
catalog = context.catalog
cb_preprocessor = catalog.load("cb_preprocessor")

# Save the preprocessor
joblib.dump(cb_preprocessor, "cb_preprocessor.pkl")
print("cb_preprocessor.pkl saved successfully.")
