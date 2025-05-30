import os
import json
import joblib
import boto3
import pandas as pd
import base64
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import logging

# for lambda to explicitly log prediction
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def preprocess_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:

    le_pillar = joblib.load("models/encoders/pillar_encoder.pkl")
    df["pillar_encoded"] = le_pillar.transform(df["legalBasis"].astype(str))

    top_countries = joblib.load("models/encoders/top_countries.pkl")
    df["country_clean"] = df["countryCoor"].where(df["countryCoor"].isin(top_countries), "Other")
    country_dummies = pd.get_dummies(df["country_clean"], prefix="country")

    df_xgb = pd.concat([
        df.drop(columns=["legalBasis", "countryCoor", "country_clean"]),
        country_dummies
    ], axis=1)

    return df_xgb


def lambda_handler(event, context):
    try:
        print("Raw event:", json.dumps(event))

        # for lambda to be able to read all kinds of formats delivered by the api
        if "body" in event:
            body = event["body"]
            try:
                if event.get("isBase64Encoded"):
                    body = base64.b64decode(body).decode("utf-8")
                input_data = json.loads(body) if isinstance(body, str) else body
       
            except Exception as e:
                logger.error(f"Error parsing body: {str(e)}")
                return {     
                    "statusCode": 400,
                    "body": json.dumps({"error": f"Invalid JSON body: {str(e)}"})
                }
        else:
            input_data = event

        # check if all fields contain data
        required_fields = [
            "totalCost", "ecMaxContribution", "duration",
            "legalBasis", "countryCoor", "numberOrg"
        ]

        for field in required_fields:
            if field not in input_data:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": f"Missing required field: {field}"})
                }

        # validating and converting numeric inputs
        try:
            total_cost = float(input_data["totalCost"])
            ec_contrib = float(input_data["ecMaxContribution"])
            duration = int(input_data["duration"])
            number_org = int(input_data["numberOrg"])
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
            "legalBasis": input_data["legalBasis"],
            "countryCoor": input_data["countryCoor"],
            "numberOrg": number_org
        }])

        model = joblib.load("models/xgb_model.pkl")
        expected_columns = joblib.load("models/expected_columns.pkl")


        # ----derived from preprocessing pipelines -----
        # totalCost
        input_df["totalCostzero"] = input_df["totalCost"].fillna(0).eq(0).astype(int)

        #calculation safe_ratio:
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

        # for lambda to explicitly log prediction
        logger.info(f"Prediction: {prediction}")

        response = {"prediction": float(prediction)}

        if "true_startupDelay" in input_data:
            try:
                true_val = float(input_data["true_startupDelay"])
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
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }