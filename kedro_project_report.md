# 📘 Kedro Project Report

## 📊 Pipelines

### `general_preprocessing`

- **Node:** `preprocess_node`
  - 📥 Inputs: project_raw, programme_raw, organization_raw
  - 📤 Outputs: preprocessed_table
  - 🧠 Function: `preprocess`

- **Node:** `impute_missing_node`
  - 📥 Inputs: preprocessed_table
  - 📤 Outputs: model_input_table
  - 🧠 Function: `impute_missing`

- **Node:** `split_data_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: X_train_raw, X_test_raw, y_train, y_test
  - 🧠 Function: `split_data`

### `xgb_preprocessing`

- **Node:** `xgb_preprocessing_node`
  - 📥 Inputs: X_train_raw, X_test_raw
  - 📤 Outputs: xgb_X_train_transformed, xgb_X_test_transformed, xgb_preprocessor
  - 🧠 Function: `apply_xgb_transformer`

### `cb_preprocessing`

- **Node:** `cb_preprocessing_node`
  - 📥 Inputs: X_train_raw, X_test_raw
  - 📤 Outputs: cb_X_train_transformed, cb_X_test_transformed, cb_cat_features, cb_preprocessor
  - 🧠 Function: `apply_cb_transformer`

### `model_training`

- **Node:** `grid_search_cb_model_node`
  - 📥 Inputs: cb_X_train_transformed, y_train, cb_cat_features, params:cb_param_grid, params:cv_folds
  - 📤 Outputs: cb_grid_search_results, cb_model
  - 🧠 Function: `grid_search_cb_model`

- **Node:** `grid_search_xgb_model_node`
  - 📥 Inputs: xgb_X_train_transformed, y_train, params:xgb_param_grid, params:cv_folds
  - 📤 Outputs: xgb_grid_search_results, xgb_model
  - 🧠 Function: `grid_search_xgb_model`

### `evaluate_model`

- **Node:** `cb_eval_metrics_node`
  - 📥 Inputs: cb_model, cb_preprocessor, X_test_raw, y_test
  - 📤 Outputs: cb_eval_metrics
  - 🧠 Function: `evaluate_regression`

- **Node:** `cb_shap_node`
  - 📥 Inputs: cb_model, cb_preprocessor, X_test_raw
  - 📤 Outputs: shap_beeswarm_cb_plot
  - 🧠 Function: `compute_cb_shap_values`

- **Node:** `xgb_eval_metrics_node`
  - 📥 Inputs: xgb_model, xgb_preprocessor, X_test_raw, y_test
  - 📤 Outputs: xgb_eval_metrics
  - 🧠 Function: `evaluate_regression`

- **Node:** `xgb_shap_node`
  - 📥 Inputs: xgb_model, xgb_preprocessor, X_test_raw
  - 📤 Outputs: shap_beeswarm_xgb_plot
  - 🧠 Function: `compute_shap_values`

### `__default__`

- **Node:** `preprocess_node`
  - 📥 Inputs: project_raw, programme_raw, organization_raw
  - 📤 Outputs: preprocessed_table
  - 🧠 Function: `preprocess`

- **Node:** `impute_missing_node`
  - 📥 Inputs: preprocessed_table
  - 📤 Outputs: model_input_table
  - 🧠 Function: `impute_missing`

- **Node:** `split_data_node`
  - 📥 Inputs: model_input_table
  - 📤 Outputs: X_train_raw, X_test_raw, y_train, y_test
  - 🧠 Function: `split_data`

- **Node:** `cb_preprocessing_node`
  - 📥 Inputs: X_train_raw, X_test_raw
  - 📤 Outputs: cb_X_train_transformed, cb_X_test_transformed, cb_cat_features, cb_preprocessor
  - 🧠 Function: `apply_cb_transformer`

- **Node:** `xgb_preprocessing_node`
  - 📥 Inputs: X_train_raw, X_test_raw
  - 📤 Outputs: xgb_X_train_transformed, xgb_X_test_transformed, xgb_preprocessor
  - 🧠 Function: `apply_xgb_transformer`

- **Node:** `grid_search_cb_model_node`
  - 📥 Inputs: cb_X_train_transformed, y_train, cb_cat_features, params:cb_param_grid, params:cv_folds
  - 📤 Outputs: cb_grid_search_results, cb_model
  - 🧠 Function: `grid_search_cb_model`

- **Node:** `grid_search_xgb_model_node`
  - 📥 Inputs: xgb_X_train_transformed, y_train, params:xgb_param_grid, params:cv_folds
  - 📤 Outputs: xgb_grid_search_results, xgb_model
  - 🧠 Function: `grid_search_xgb_model`

- **Node:** `cb_eval_metrics_node`
  - 📥 Inputs: cb_model, cb_preprocessor, X_test_raw, y_test
  - 📤 Outputs: cb_eval_metrics
  - 🧠 Function: `evaluate_regression`

- **Node:** `cb_shap_node`
  - 📥 Inputs: cb_model, cb_preprocessor, X_test_raw
  - 📤 Outputs: shap_beeswarm_cb_plot
  - 🧠 Function: `compute_cb_shap_values`

- **Node:** `xgb_eval_metrics_node`
  - 📥 Inputs: xgb_model, xgb_preprocessor, X_test_raw, y_test
  - 📤 Outputs: xgb_eval_metrics
  - 🧠 Function: `evaluate_regression`

- **Node:** `xgb_shap_node`
  - 📥 Inputs: xgb_model, xgb_preprocessor, X_test_raw
  - 📤 Outputs: shap_beeswarm_xgb_plot
  - 🧠 Function: `compute_shap_values`


## 📁 Data Catalog

- `project_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/project.json`
- `organization_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/organization.json`
- `programme_raw`: **kedro_datasets.json.JSONDataset** → `data/01_raw/programme.json`
- `model_input_table`: **kedro_datasets.pandas.ParquetDataset** → `data/03_primary/selected_features.parquet`
- `X_train_raw`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/X_train_raw.parquet`
- `X_test_raw`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/X_test_raw.parquet`
- `y_train`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/y_train.parquet`
- `y_test`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/y_test.parquet`
- `cb_X_train_transformed`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_X_train_transformed.parquet`
- `cb_X_test_transformed`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/cb_X_test_transformed.parquet`
- `cb_cat_features`: **kedro_datasets.json.JSONDataset** → `data/05_model_input/cb_cat_features.json`
- `cb_preprocessor`: **pickle.PickleDataset** → `data/06_models/cb_preprocessor.pkl`
- `xgb_X_train_transformed`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_X_train_transformed.parquet`
- `xgb_X_test_transformed`: **kedro_datasets.pandas.ParquetDataset** → `data/05_model_input/xgb_X_test_transformed.parquet`
- `xgb_preprocessor`: **pickle.PickleDataset** → `data/06_models/xgb_preprocessor.pkl`
- `cb_eval_metrics`: **kedro_datasets.json.JSONDataset** → `data/08_reporting/cb_eval_metrics.json`
- `xgb_eval_metrics`: **kedro_datasets.json.JSONDataset** → `data/08_reporting/xgb_eval_metrics.json`
- `cb_model`: **pickle.PickleDataset** → `data/06_models/cb_best_model.pkl`
- `xgb_model`: **pickle.PickleDataset** → `data/06_models/xgb_best_model.pkl`
- `shap_beeswarm_cb_plot`: **kedro_datasets.matplotlib.MatplotlibWriter** → `data/08_reporting/shap_cb_summary.png`
- `shap_beeswarm_xgb_plot`: **kedro_datasets.matplotlib.MatplotlibWriter** → `data/08_reporting/shap_xgb_summary.png`
- `cb_grid_search_results`: **kedro_datasets.pandas.ParquetDataset** → `data/08_reporting/cb_grid_results.parquet`
- `xgb_grid_search_results`: **kedro_datasets.pandas.ParquetDataset** → `data/08_reporting/xgb_grid_results.parquet`

## 🧠 Node Function Code (Top-Level Only)

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

### `split_data`
```python
def split_data(df: pd.DataFrame):
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
    """
    Applies XGBoost-specific preprocessing to already-split data and logs metadata to MLflow.
    Returns transformed train/test sets and the fitted preprocessor.
    """
    with mlflow.start_run(nested=True):
        mlflow.set_tag("pipeline_step", "xgb_preprocessing_node")

        transformer = XGBoostPreprocessor()
        X_train_transformed = transformer.fit_transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        mlflow.log_param("top_countries", transformer.top_countries_)
        mlflow.log_param("top_funding", transformer.top_funding_)

        return X_train_transformed, X_test_transformed, transformer
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

### `grid_search_cb_model`
```python
def grid_search_cb_model(X, y, cat_features, param_grid, cv_folds):
    results = []
    best_score = -np.inf
    best_params = None
    best_model = None

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="cb_grid_search"):
        for params in ParameterGrid(param_grid):
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model = CatBoostRegressor(**params, verbose=0)
                model.fit(X_train_fold, y_train_fold, cat_features=cat_features)
                score = model.score(X_val_fold, y_val_fold)
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            results.append({"params": params, "cv_r2": avg_score})

            param_str = "_".join(f"{k}-{v}" for k, v in params.items())
            mlflow.log_metric(f"cv_r2_{param_str}", avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        # Fit the best model on the full training data
        best_model = CatBoostRegressor(**best_params, verbose=0).fit(X, y, cat_features=cat_features)
        os.makedirs("data/06_models", exist_ok=True)
        joblib.dump(best_model, "data/06_models/cb_best_model.pkl")

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_r2", best_score)
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        # Save grid results as artifact
        os.makedirs("data/08_reporting", exist_ok=True)
        results_df = pd.DataFrame(results)
        results_parquet = "data/08_reporting/cb_grid_results.parquet"
        results_df.to_parquet(results_parquet)
        mlflow.log_artifact(results_parquet)

    return results_df, best_model
```

### `grid_search_xgb_model`
```python
def grid_search_xgb_model(X, y, param_grid, cv_folds):
    results = []
    best_score = -np.inf
    best_params = None
    best_model = None

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="xgb_grid_search"):
        for params in ParameterGrid(param_grid):
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                model = XGBRegressor(**params)
                model.fit(X_train_fold, y_train_fold)
                score = model.score(X_val_fold, y_val_fold)
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            results.append({"params": params, "cv_r2": avg_score})

            param_str = "_".join(f"{k}-{v}" for k, v in params.items())
            mlflow.log_metric(f"cv_r2_{param_str}", avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        # Fit the best model on the full training data
        best_model = XGBRegressor(**best_params).fit(X, y)
        os.makedirs("data/06_models", exist_ok=True)
        joblib.dump(best_model, "data/06_models/xgb_best_model.pkl")

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_r2", best_score)
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        # Save grid results as artifact
        os.makedirs("data/08_reporting", exist_ok=True)
        results_df = pd.DataFrame(results)
        results_parquet = "data/08_reporting/xgb_grid_results.parquet"
        results_df.to_parquet(results_parquet)
        mlflow.log_artifact(results_parquet)

    return results_df, best_model
```

### `evaluate_regression`
```python
def evaluate_regression(model, preprocessor, X_test_raw, y_test):
    # Transform raw test input
    X_test = preprocessor.transform(X_test_raw)
    y_pred = model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    # Log to MLflow
    with mlflow.start_run(nested=True):
        mlflow.set_tag("pipeline_step", "evaluate_model")
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    return metrics
```

### `compute_cb_shap_values`
```python
def compute_cb_shap_values(model, preprocessor, X_test_raw):
    # Apply preprocessing
    X_test = preprocessor.transform(X_test_raw)

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Compute mean absolute SHAP values per feature
    shap_df = pd.DataFrame(
        np.abs(shap_values.values), columns=X_test.columns
    ).mean().sort_values(ascending=True)

    # Bar plot (horizontal)
    fig, ax = plt.subplots(figsize=(8, max(6, len(shap_df) * 0.3)))
    shap_df.plot.barh(ax=ax)
    ax.set_title("Mean |SHAP| values (CatBoost)")
    ax.set_xlabel("Mean absolute SHAP value")
    plt.tight_layout()

    # Log to MLflow
    output_path = "data/08_reporting/shap_cb_summary.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")

    with mlflow.start_run(nested=True):
        mlflow.set_tag("pipeline_step", "shap_cb")
        mlflow.set_tag("model_type", "catboost")
        mlflow.set_tag("explanation_type", "bar_plot")
        mlflow.log_artifact(output_path)

    return fig
```

### `compute_shap_values`
```python
def compute_shap_values(model, preprocessor, X_test_raw):
    # Apply preprocessing
    X_test = preprocessor.transform(X_test_raw)

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Compute mean absolute SHAP values per feature
    shap_df = pd.DataFrame(
        np.abs(shap_values.values), columns=X_test.columns
    ).mean().sort_values(ascending=True)

    # Bar plot (horizontal)
    fig, ax = plt.subplots(figsize=(8, max(6, len(shap_df) * 0.3)))
    shap_df.plot.barh(ax=ax)
    ax.set_title("Mean |SHAP| values (XGBoost)")
    ax.set_xlabel("Mean absolute SHAP value")
    plt.tight_layout()

    # Log to MLflow
    output_path = "data/08_reporting/shap_xgb_summary.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")

    with mlflow.start_run(nested=True):
        mlflow.set_tag("pipeline_step", "shap_xgb")
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("explanation_type", "bar_plot")
        mlflow.log_artifact(output_path)

    return fig
```
