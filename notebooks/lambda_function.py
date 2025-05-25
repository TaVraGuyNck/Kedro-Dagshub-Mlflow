import os
import json
import joblib
import boto3
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def preprocess_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:

    # TEMP for test:
    
    #le_pillar = joblib.load("/tmp/pillar_encoder.pkl")
    le_pillar = joblib.load("models/encoders/pillar_encoder.pkl")
    df["pillar_encoded"] = le_pillar.transform(df["pillar"].astype(str))

    # TEMP for test: 
    top_countries = joblib.load("models/encoders/top_countries.pkl")
    #top_countries = joblib.load("/tmp/top_countries.pkl")

    df["country_clean"] = df["countryCoor"].where(df["countryCoor"].isin(top_countries), "Other")
    country_dummies = pd.get_dummies(df["country_clean"], prefix="country")

    df_xgb = pd.concat([
        df.drop(columns=["pillar", "countryCoor", "country_clean"]),
        country_dummies
    ], axis=1)

    return df_xgb


def lambda_handler(event, context):
    try:
        required_fields = [
            "totalCost", "ecMaxContribution", "duration",
            "pillar", "countryCoor", "numberOrg"
        ]

        for field in required_fields:
            if field not in event:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": f"Missing required field: {field}"})
                }

        # --- Validate and convert numeric inputs ---
        try:
            total_cost = float(event["totalCost"])
            ec_contrib = float(event["ecMaxContribution"])
            duration = int(event["duration"])
            number_org = int(event["numberOrg"])
        except Exception:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Numeric inputs must be valid numbers."})
            }

        if total_cost < 0 or ec_contrib < 0 or duration < 0 or number_org < 0:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Numeric inputs must be non-negative."})
            }

        input_df = pd.DataFrame([{
            "totalCost": total_cost,
            "ecMaxContribution": ec_contrib,
            "duration": duration,
            "pillar": event["pillar"],
            "countryCoor": event["countryCoor"],
            "numberOrg": number_org
        }])


        # TEMP for local test:
        model = joblib.load("models/xgb_model.pkl")
        expected_columns = joblib.load("models/expected_columns.pkl")
        #s3 = boto3.client("s3")
        #bucket = "your-s3-bucket-name"  # Replace with your bucket name
        #s3.download_file(bucket, "models/xgb_model.pkl", "/tmp/xgb_model.pkl")
        #s3.download_file(bucket, "models/expected_columns.pkl", "/tmp/expected_columns.pkl")
        #s3.download_file(bucket, "models/encoders/pillar_encoder.pkl", "/tmp/pillar_encoder.pkl")
        #s3.download_file(bucket, "models/encoders/top_countries.pkl", "/tmp/top_countries.pkl")
        #model = joblib.load("/tmp/xgb_model.pkl")
        #expected_columns = joblib.load("/tmp/expected_columns.pkl")


        # --- Derived features from preprocessing ---
        input_df["totalCostzero"] = input_df["totalCost"].fillna(0).eq(0).astype(int)

        def safe_ratio(row):
            try:
                if pd.notnull(row["ecMaxContribution"]) and pd.notnull(row["totalCost"]) and row["totalCost"] != 0:
                    return float(row["ecMaxContribution"]) / float(row["totalCost"])
            except Exception:
                pass
            return None

        input_df["contRatio"] = input_df.apply(safe_ratio, axis=1)

        processed_df = preprocess_for_xgboost(input_df)

        for col in expected_columns:
            if col not in processed_df.columns:
                processed_df[col] = 0

        processed_df = processed_df[expected_columns]

        prediction = model.predict(processed_df)[0]

        response = {"prediction": float(prediction)}

        if "true_startupDelay" in event:
            try:
                true_val = float(event["true_startupDelay"])
                mae = mean_absolute_error([true_val], [prediction])
                response["mae"] = mae
            except Exception:
                response["mae_error"] = "Invalid true_startupDelay. Must be a number."

        if input_df["totalCost"].iloc[0] == 0:
            response["warning"] = (
                "You entered a totalCost of 0. "
                "Please note that when not provided with the correct totalCost, "
                "the prediction of the start-up delay for your project will be of lower quality."
            )

        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
