# 📘 Kedro Project Report

## 📊 Pipelines

### `train_xgboost`

- **Node:** `xgb_split_data_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: xgb_X_train_raw, xgb_X_test_raw, xgb_y_train, xgb_y_test
  - 🧠 Function: `split_xgboost_data`

- **Node:** `xgb_preprocessing_node`
  - 📥 Inputs: xgb_X_train_raw, xgb_X_test_raw
  - 📤 Outputs: xgb_X_train, xgb_X_test, xgb_preprocessor
  - 🧠 Function: `apply_xgb_transformer`

- **Node:** `xgb_training_node`
  - 📥 Inputs: xgb_X_train, xgb_y_train, params:xgb_params, xgb_preprocessor
  - 📤 Outputs: xgb_model, xgb_model_mlflow
  - 🧠 Function: `train_xgboost_model`

### `train_catboost`

- **Node:** `cb_split_data_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: cb_X_train_raw, cb_X_test_raw, cb_y_train, cb_y_test
  - 🧠 Function: `split_catboost_data`

- **Node:** `cb_preprocessing_node`
  - 📥 Inputs: cb_X_train_raw, cb_X_test_raw
  - 📤 Outputs: cb_X_train, cb_X_test, cb_cat_features, cb_preprocessor
  - 🧠 Function: `apply_cb_transformer`

- **Node:** `cb_training_node`
  - 📥 Inputs: cb_X_train, cb_y_train, cb_X_test, cb_y_test, params:cb_params, cb_cat_features
  - 📤 Outputs: cb_model, cb_model_mlflow
  - 🧠 Function: `train_catboost_model`

### `evaluate_model`

- **Node:** `evaluate_cb_point_node`
  - 📥 Inputs: cb_model, cb_X_test, cb_y_test
  - 📤 Outputs: metrics_cb
  - 🧠 Function: `evaluate_cb_point_model`

- **Node:** `evaluate_xgb_node`
  - 📥 Inputs: xgb_model, xgb_X_test, xgb_y_test
  - 📤 Outputs: metrics_xgb
  - 🧠 Function: `evaluate_xgb_model`

- **Node:** `xgb_shap_node`
  - 📥 Inputs: xgb_model, xgb_preprocessor, xgb_X_test_raw
  - 📤 Outputs: shap_beeswarm_plot
  - 🧠 Function: `compute_shap_values`

### `compare_models`

- **Node:** `compare_model_metrics_node`
  - 📥 Inputs: metrics_xgb, metrics_cb
  - 📤 Outputs: model_comparison_table
  - 🧠 Function: `compare_model_metrics`

### `general_preprocessing`

- **Node:** `preprocess_node`
  - 📥 Inputs: project_raw, programme_raw, organization_raw
  - 📤 Outputs: preprocessed_table
  - 🧠 Function: `preprocess`

- **Node:** `impute_missing_node`
  - 📥 Inputs: preprocessed_table
  - 📤 Outputs: model_input_table
  - 🧠 Function: `impute_missing`

### `__default__`

- **Node:** `preprocess_node`
  - 📥 Inputs: project_raw, programme_raw, organization_raw
  - 📤 Outputs: preprocessed_table
  - 🧠 Function: `preprocess`

- **Node:** `impute_missing_node`
  - 📥 Inputs: preprocessed_table
  - 📤 Outputs: model_input_table
  - 🧠 Function: `impute_missing`

- **Node:** `cb_split_data_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: cb_X_train_raw, cb_X_test_raw, cb_y_train, cb_y_test
  - 🧠 Function: `split_catboost_data`

- **Node:** `xgb_split_data_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: xgb_X_train_raw, xgb_X_test_raw, xgb_y_train, xgb_y_test
  - 🧠 Function: `split_xgboost_data`

- **Node:** `cb_preprocessing_node`
  - 📥 Inputs: cb_X_train_raw, cb_X_test_raw
  - 📤 Outputs: cb_X_train, cb_X_test, cb_cat_features, cb_preprocessor
  - 🧠 Function: `apply_cb_transformer`

- **Node:** `xgb_preprocessing_node`
  - 📥 Inputs: xgb_X_train_raw, xgb_X_test_raw
  - 📤 Outputs: xgb_X_train, xgb_X_test, xgb_preprocessor
  - 🧠 Function: `apply_xgb_transformer`

- **Node:** `cb_training_node`
  - 📥 Inputs: cb_X_train, cb_y_train, cb_X_test, cb_y_test, params:cb_params, cb_cat_features
  - 📤 Outputs: cb_model, cb_model_mlflow
  - 🧠 Function: `train_catboost_model`

- **Node:** `xgb_training_node`
  - 📥 Inputs: xgb_X_train, xgb_y_train, params:xgb_params, xgb_preprocessor
  - 📤 Outputs: xgb_model, xgb_model_mlflow
  - 🧠 Function: `train_xgboost_model`

- **Node:** `evaluate_cb_point_node`
  - 📥 Inputs: cb_model, cb_X_test, cb_y_test
  - 📤 Outputs: metrics_cb
  - 🧠 Function: `evaluate_cb_point_model`

- **Node:** `evaluate_xgb_node`
  - 📥 Inputs: xgb_model, xgb_X_test, xgb_y_test
  - 📤 Outputs: metrics_xgb
  - 🧠 Function: `evaluate_xgb_model`

- **Node:** `xgb_shap_node`
  - 📥 Inputs: xgb_model, xgb_preprocessor, xgb_X_test_raw
  - 📤 Outputs: shap_beeswarm_plot
  - 🧠 Function: `compute_shap_values`

- **Node:** `compare_model_metrics_node`
  - 📥 Inputs: metrics_xgb, metrics_cb
  - 📤 Outputs: model_comparison_table
  - 🧠 Function: `compare_model_metrics`


## 📁 Data Catalog

- `project_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/project.json`
- `organization_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/organization.json`
- `programme_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/programme.json`
- `model_input_table`: **kedro_datasets.pandas.ParquetDataset** → `data/03_primary/selected_features.parquet`
- `xgb_y_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_y_train.parquet`
- `xgb_y_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_y_test.parquet`
- `cb_X_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_X_train.parquet`
- `cb_X_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_X_test.parquet`
- `cb_y_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_y_train.parquet`
- `cb_y_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_y_test.parquet`
- `cb_cat_features`: **kedro_datasets.json.JSONDataset** → `data/05_model_input/cb_cat_features.json`
- `metrics_xgb`: **kedro_datasets.pandas.ParquetDataset** → `data/08_reporting/metrics_xgb.parquet`
- `metrics_cb`: **kedro_datasets.pandas.ParquetDataset** → `data/08_reporting/metrics_cb.parquet`
- `model_comparison_table`: **kedro_datasets.pandas.ParquetDataset** → `data/08_reporting/model_comparison.parquet`
- `confusion_matrix_plot`: **kedro_datasets.matplotlib.MatplotlibWriter** → `data/08_reporting/confusion_matrix.png`
- `xgb_X_train_raw`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_X_train_raw.parquet`
- `xgb_X_test_raw`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_X_test_raw.parquet`
- `xgb_X_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_X_train.parquet`
- `xgb_X_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_X_test.parquet`
- `shap_beeswarm_plot`: **kedro_datasets.matplotlib.MatplotlibWriter** → `data/08_reporting/shap_beeswarm_xgb.png`
- `cb_X_train_raw`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_X_train_raw.parquet`
- `cb_X_test_raw`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_X_test_raw.parquet`
- `cb_preprocessor`: **pickle.PickleDataset** → `data/06_models/cb_preprocessor.pkl`
- `cb_model`: **pickle.PickleDataset** → `data/06_models/cb_model.pkl`
- `cb_model_mlflow`: **pickle.PickleDataset** → `data/06_models/cb_model_mlflow.pkl`
- `xgb_model`: **pickle.PickleDataset** → `data/06_models/xgb_model.pkl`
- `xgb_model_mlflow`: **pickle.PickleDataset** → `data/06_models/xgb_model_mlflow.pkl`
- `xgb_preprocessor`: **pickle.PickleDataset** → `data/06_models/xgb_preprocessor.pkl`

## 🧠 Node Function Code (Top-Level Only)

### `split_xgboost_data`
```python
def split_xgboost_data(df: pd.DataFrame):
    X = df.drop(columns=["startupDelay", "id"])
    y = df["startupDelay"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        X_train,
        X_test,
        y_train.to_frame(name="startupDelay"),
        y_test.to_frame(name="startupDelay"),
    )
```

### `apply_xgb_transformer`
```python
def apply_xgb_transformer(X_train: pd.DataFrame, X_test: pd.DataFrame):
    transformer = XGBoostPreprocessor()
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    return X_train_transformed, X_test_transformed, transformer
```

### `train_xgboost_model`
```python
def train_xgboost_model(X_train, y_train, xgb_params, transformer):
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    # Prepare save paths
    os.makedirs("data/06_models", exist_ok=True)
    kedro_model_path = "data/06_models/xgb_model.pkl"
    mlflow_model_path = "data/06_models/xgb_model_mlflow.pkl"
    preproc_path = "data/06_models/xgb_preprocessor.pkl"

    # Save model twice and transformer once
    joblib.dump(model, kedro_model_path)         # For Kedro
    joblib.dump(model, mlflow_model_path)        # For MLflow
    joblib.dump(transformer, preproc_path)       # Shared

    # MLflow logging
    with mlflow.start_run(nested=True):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("preprocessor", "XGBoostPreprocessor")
        mlflow.set_tag("pipeline_step", "train_xgboost_model")

        for key, value in xgb_params.items():
            mlflow.log_param(f"xgb_{key}", value)

        mlflow.log_artifact(mlflow_model_path)
        mlflow.log_artifact(preproc_path)

        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_score", train_score)

    return model, model  # Matches outputs=["xgb_model", "xgb_model_mlflow"]
```

### `split_catboost_data`
```python
def split_catboost_data(df: pd.DataFrame):
    with mlflow.start_run(nested=True):
        mlflow.set_tag("pipeline_step", "cb_split_data_node")

        X = df.drop(columns=["startupDelay", "id"])
        y = df["startupDelay"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        mlflow.log_param("split_ratio", 0.2)
        mlflow.log_metric("num_train_samples", len(X_train))
        mlflow.log_metric("num_test_samples", len(X_test))

        return (
            X_train,
            X_test,
            y_train.to_frame(name="startupDelay"),
            y_test.to_frame(name="startupDelay")
        )
```

### `apply_cb_transformer`
```python
def apply_cb_transformer(X_train: pd.DataFrame, X_test: pd.DataFrame):
    with mlflow.start_run(nested=True):
        mlflow.set_tag("pipeline_step", "cb_preprocessing_node")

        transformer = CatBoostPreprocessor()
        X_train_transformed = transformer.fit_transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        mlflow.set_tag("cat_features", ", ".join(transformer.cat_features))
        mlflow.log_param("top_countries", transformer.top_countries_)
        mlflow.log_param("top_funding", transformer.top_funding_)

        return (
            X_train_transformed,
            X_test_transformed,
            transformer.cat_features,
            transformer
        )
```

### `train_catboost_model`
```python
def train_catboost_model(X_train, y_train, X_valid, y_valid, catboost_params, cat_features):
    with mlflow.start_run(nested=True):
        mlflow.set_tag("pipeline_step", "cb_training_node")
        mlflow.set_tag("model_type", "catboost")

        model = CatBoostRegressor(**catboost_params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features,
            verbose=0
        )

        # Save two versions
        os.makedirs("data/06_models", exist_ok=True)
        kedro_path = "data/06_models/cb_model.pkl"
        mlflow_path = "data/06_models/cb_model_mlflow.pkl"

        joblib.dump(model, kedro_path)
        joblib.dump(model, mlflow_path)

        # Log MLflow artifacts and params
        mlflow.log_artifact(mlflow_path)
        for key, value in catboost_params.items():
            mlflow.log_param(f"cb_{key}", value)
        mlflow.log_param("cb_cat_features_used", cat_features)
        mlflow.log_metric("train_score", model.score(X_train, y_train))

        return model, model
```

### `evaluate_cb_point_model`
```python
def evaluate_cb_point_model(
    cb_model: CatBoostRegressor,
    cb_X_test: pd.DataFrame,
    cb_y_test: pd.Series
) -> pd.DataFrame:
    y_pred = cb_model.predict(cb_X_test)

    # Compute regression metrics using your custom utility
    metrics = regression_metrics(cb_y_test, y_pred)

    # Log each metric to MLflow
    for key, value in metrics.items():
        mlflow.log_metric(f"cb_{key}", value)

    return pd.DataFrame([metrics])
```

### `evaluate_xgb_model`
```python
def evaluate_xgb_model(
    xgb_model: XGBRegressor,
    xgb_X_test: pd.DataFrame,
    xgb_y_test: pd.Series
) -> pd.DataFrame:
    y_pred = xgb_model.predict(xgb_X_test)
    
    # Compute metrics (assuming this is a custom util that returns a dict)
    metrics = regression_metrics(xgb_y_test, y_pred)

    # Log each metric to MLflow
    for key, value in metrics.items():
        mlflow.log_metric(f"xgb_{key}", value)

    return pd.DataFrame([metrics])
```

### `compute_shap_values`
```python
def compute_shap_values(model, preprocessor, X_test_raw):
    # Apply preprocessing
    X_test = preprocessor.transform(X_test_raw)

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Generate beeswarm plot
    plt.figure()
    shap_plot = shap.plots.beeswarm(shap_values, show=False)

    # Log to MLflow (optional)
    
    plt.savefig("data/08_reporting/shap_beeswarm_xgb.png", bbox_inches="tight")
    mlflow.log_artifact("data/08_reporting/shap_beeswarm_xgb.png")
    

    print("SHAP values computed and plot returned.")
    return plt.gcf()
```

### `compare_model_metrics`
```python
def compare_model_metrics(metrics_xgb, metrics_cb):
    df_xgb = metrics_xgb.copy()
    df_cb = metrics_cb.copy()

    df_xgb["model"] = "xgboost"
    df_cb["model"] = "catboost_point"
    

    result = pd.concat([df_xgb, df_cb], ignore_index=True)
    return result  # guaranteed to be a single DataFrame
```

### `preprocess`
```python
def preprocess(project_df, programme_df, org_df) -> pd.DataFrame:
    # --- Ensure all inputs are DataFrames ---
    project_df = pd.DataFrame(project_df)
    programme_df = pd.DataFrame(programme_df)
    org_df = pd.DataFrame(org_df)

    # --- Ensure expected numeric fields are properly typed ---
    numeric_cols = ["ecMaxContribution", "totalCost"]
    for col in numeric_cols:
        if col in project_df.columns:
            project_df[col] = pd.to_numeric(project_df[col], errors="coerce")
        else:
            project_df[col] = np.nan  # Create it to avoid downstream key errors

    # --- Parse dates safely ---
    for date_col in ["startDate", "endDate", "ecSignatureDate"]:
        if date_col in project_df.columns:
            project_df[date_col] = pd.to_datetime(project_df[date_col], errors="coerce")
        else:
            project_df[date_col] = pd.NaT

    # --- Derived features ---
    project_df["startupDelay"] = (project_df["startDate"] - project_df["ecSignatureDate"]).dt.days
    project_df["duration"] = (project_df["endDate"] - project_df["startDate"]).dt.days
    project_df["totalCostzero"] = project_df["totalCost"].fillna(0).eq(0).astype(int)

    # Safe ratio calculation
    def safe_ratio(row):
        try:
            if pd.notnull(row["ecMaxContribution"]) and pd.notnull(row["totalCost"]) and row["totalCost"] != 0:
                return float(row["ecMaxContribution"]) / float(row["totalCost"])
        except Exception:
            pass
        return None

    project_df["contRatio"] = project_df.apply(safe_ratio, axis=1)

    # --- Map legal basis to pillar ---
    mapping = {
        "HORIZON.1.1": "Pillar 1 - European Research Council (ERC)",
        "HORIZON.1.2": "Pillar 1 - Marie Sklodowska-Curie Actions (MSCA)",
        "HORIZON.1.3": "Pillar 1 - Research infrastructures",
        "HORIZON.2.1": "Pillar 2 - Health",
        "HORIZON.2.2": "Pillar 2 - Culture, creativity and inclusive society",
        "HORIZON.2.3": "Pillar 2 - Civil Security for Society",
        "HORIZON.2.4": "Pillar 2 - Digital, Industry and Space",
        "HORIZON.2.5": "Pillar 2 - Climate, Energy and Mobility",
        "HORIZON.2.6": "Pillar 2 - Food, Bioeconomy Natural Resources, Agriculture and Environment",
        "HORIZON.3.1": "Pillar 3 - The European Innovation Council (EIC)",
        "HORIZON.3.2": "Pillar 3 - European innovation ecosystems",
        "HORIZON.3.3": "Pillar 3 - Cross-cutting call topics",
        "EURATOM2027": "EURATOM2027",
        "EURATOM.1.1": "Improve and support nuclear safety...",
        "EURATOM.1.2": "Maintain and further develop expertise...",
        "EURATOM.1.3": "Foster the development of fusion energy...",
    }
    project_df["pillar"] = project_df.get("legalBasis", pd.Series(dtype=object)).map(mapping)

    # --- Merge with organization data ---
    org_df["projectID"] = org_df.get("projectID", pd.Series(dtype=object)).astype(str)
    project_df["id"] = project_df.get("id", pd.Series(dtype=object)).astype(str)

    # Coordinator country
    coor_map = org_df[org_df["role"] == "coordinator"][["projectID", "country"]]
    coor_map = coor_map.drop_duplicates("projectID").set_index("projectID")["country"]
    project_df["countryCoor"] = project_df["id"].map(coor_map)

    # Number of participating organizations
    number_org = org_df.groupby("projectID").size()
    project_df["numberOrg"] = project_df["id"].map(number_org).fillna(0).astype(int)

    # --- Drop unreasonable values ---
    project_df = project_df[project_df["startupDelay"] >= 0]

    # --- Final feature selection (drop any missing columns) ---
    expected_cols = [
        "id", "startupDelay", "totalCost", "totalCostzero",
        "ecMaxContribution", "duration", "contRatio",
        "pillar", "countryCoor", "numberOrg", "fundingScheme" 
    ]
    selected = project_df.reindex(columns=[col for col in expected_cols if col in project_df.columns])

    return selected
```

### `impute_missing`
```python
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing target
    df = df[df["startupDelay"].notna()]

    # Separate by dtype
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Impute numerics with median
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Impute categoricals with "missing"
    for col in cat_cols:
        df[col] = df[col].fillna("missing")

    # Remove index feature    

    df = df.reset_index(drop=True)

    return df
```
