import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import mlflow
import os


# --- Custom Preprocessor ---
class XGBoostPreprocessor:
    def __init__(self, top_n_countries=10, top_n_funding=10):
        self.top_n_countries = top_n_countries
        self.top_n_funding = top_n_funding
        self.top_countries_ = None
        self.top_funding_ = None

    def fit(self, X, y=None):
        self.top_countries_ = (
            X["countryCoor"]
            .value_counts()
            .nlargest(self.top_n_countries)
            .index
            .tolist()
        )
        self.top_funding_ = (
            X["fundingScheme"]
            .value_counts()
            .nlargest(self.top_n_funding)
            .index
            .tolist()
        )
        return self

    def transform(self, X):
        X = X.copy()

        # Fill missing values for safety
        X["pillar"] = X["pillar"].fillna("missing")
        X["countryCoor"] = X["countryCoor"].fillna("missing")
        X["fundingScheme"] = X["fundingScheme"].fillna("missing")

        # Encode pillar as category codes
        X["pillar_encoded"] = X["pillar"].astype("category").cat.codes

        # --- Handle country ---
        X["country_clean"] = X["countryCoor"].where(
            X["countryCoor"].isin(self.top_countries_), "Other"
        )
        country_dummies = pd.get_dummies(X["country_clean"], prefix="country")

        # --- Handle funding scheme ---
        X["funding_clean"] = X["fundingScheme"].where(
            X["fundingScheme"].isin(self.top_funding_), "Other"
        )
        funding_dummies = pd.get_dummies(X["funding_clean"], prefix="funding")

        # Combine everything
        X = pd.concat([
            X.drop(columns=[
                "pillar", "countryCoor", "country_clean",
                "fundingScheme", "funding_clean"
            ]),
            country_dummies,
            funding_dummies
        ], axis=1)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# --- Node 1: Split raw data into train/test ---
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


def apply_xgb_transformer(X_train: pd.DataFrame, X_test: pd.DataFrame):
    transformer = XGBoostPreprocessor()
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    return X_train_transformed, X_test_transformed, transformer



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
