import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import mlflow
import joblib
import os

def preprocess_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    le_pillar = LabelEncoder()
    df["pillar_encoded"] = le_pillar.fit_transform(df["pillar"].astype(str))

    # save encoders for lambda function
    os.makedirs("data/06_models/encoders", exist_ok=True)
    joblib.dump(le_pillar, "data/06_models/encoders/pillar_encoder.pkl")

    top_countries = df["countryCoor"].value_counts().nlargest(10).index
    
    # save country dummies for lambda function
    joblib.dump(top_countries.tolist(), "data/06_models/encoders/top_countries.pkl")

    df["country_clean"] = df["countryCoor"].where(df["countryCoor"].isin(top_countries), "Other")
    country_dummies = pd.get_dummies(df["country_clean"], prefix="country")

    df_xgb = pd.concat([
        df.drop(columns=["pillar", "countryCoor", "country_clean"]),
        country_dummies
    ], axis=1)

    return df_xgb


def split_xgboost_data(df: pd.DataFrame):
    X = df.drop(columns=["startupDelay", "id"])
    y = df["startupDelay"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wrap y Series as DataFrames
    return (
        X_train,
        X_test,
        y_train.to_frame(name="startupDelay"),
        y_test.to_frame(name="startupDelay"),
    )

def train_xgboost_model(X_train, y_train, xgb_params):
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    # Save 2 versions of model locally
    os.makedirs("data/06_models", exist_ok=True)
    kedro_path = "data/06_models/xgb_model.pkl"
    mlflow_path= "data/06_models/xgb_model_mlflow.pkl"

    joblib.dump(model, kedro_path)
    joblib.dump(model, mlflow_path)
 
    # save feature list for lambda function
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, "data/06_models/expected_columns.pkl")

    # Log to MLflow
    mlflow.set_tag("model_type", "xgboost")
    for key, value in xgb_params.items():
        mlflow.log_param(f"xgb_{key}", value)
    mlflow.log_artifact(mlflow_path)
    mlflow.log_metric("train_score", model.score(X_train, y_train))

    return model, model
