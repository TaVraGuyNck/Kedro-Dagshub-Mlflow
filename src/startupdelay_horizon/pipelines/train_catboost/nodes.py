import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import joblib
import mlflow
import os

# --- STEP 1: Split data first ---
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

# --- STEP 2: Transformer class ---
class CatBoostPreprocessor:
    def __init__(self, top_n_countries=10, top_n_funding=10):
        self.top_n_countries = top_n_countries
        self.top_n_funding = top_n_funding
        self.top_countries_ = None
        self.top_funding_ = None
        self.cat_features = []

    def fit(self, X: pd.DataFrame, y=None):
        self.top_countries_ = (
            X["countryCoor"]
            .fillna("missing")
            .value_counts()
            .nlargest(self.top_n_countries)
            .index
            .tolist()
        )
        self.top_funding_ = (
            X["fundingScheme"]
            .fillna("missing")
            .value_counts()
            .nlargest(self.top_n_funding)
            .index
            .tolist()
        )
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # Fill missing values
        X["pillar"] = X["pillar"].fillna("missing")
        X["countryCoor"] = X["countryCoor"].fillna("missing")
        X["fundingScheme"] = X["fundingScheme"].fillna("missing")

        # Reduce cardinality
        X["countryCoor"] = X["countryCoor"].where(
            X["countryCoor"].isin(self.top_countries_), "Other"
        )
        X["fundingScheme"] = X["fundingScheme"].where(
            X["fundingScheme"].isin(self.top_funding_), "Other"
        )

        # Convert to strings
        X["pillar"] = X["pillar"].astype(str)
        X["countryCoor"] = X["countryCoor"].astype(str)
        X["fundingScheme"] = X["fundingScheme"].astype(str)

        return X

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X, y)
        transformed = self.transform(X)
        self.cat_features = ["pillar", "countryCoor", "fundingScheme"]
        return transformed


# --- STEP 3: Apply transformer ---
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

# --- STEP 4: Train model ---
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