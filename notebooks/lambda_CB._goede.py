import os
import json
import joblib
import boto3
import pandas as pd
import base64
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import logging

logger = logging.getLogger()                                                                 #logging
logger.setLevel(logging.INFO)


def legal_basis_to_pillar(legal_basis_code):
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
            return mapping.get(legal_basis_code, "missing")


  def safe_ratio(row):
            try:
                if pd.notnull(row["ecMaxContribution"]) and pd.notnull(row["totalCost"]) and row["totalCost"] != 0:
                    return float(row["ecMaxContribution"]) / float(row["totalCost"])
            except Exception:
                pass
            return None
    

def lambda_handler(event, context):
    try: print("raw event:", json.dumps(event))                                                #logging

        # to ensure lambda will be able to read all kinds of data formats
        if "body" in event:
            body = event["body"]
            try:
                if event.get("isBase64Encoded"):
                    body = base64.b64decode(body).decode("utf-8")
                input_data = json.loads(body) if isinstance(body, str) else body
            except Exception as e:
                return {
                    "statusCode": 400,                                                         #logging
                    "body": json.dumps({"error": f"Invalid JSON body: {str(e)}"})
                }
        else: input_data = event

        # ensuring all necessary data input is present 
        required_fields = ["totalCost", "ecMaxContribution", "duration",
            "LegalBasis", "countryCoor", "numberOrg", "fundingScheme"]

        for field in required_fields:
            if field not in input_data:
                return {
                    "statusCode": 400,                                                         #logging
                    "body": json.dumps({"error": f"Missing required field: {field}"})
                }

        # validation dnd formatting of numerical data
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
    
        # assembling all features in dataframe for input to CBmodel  
        pillar = map_legal_basis_to_pillar(input_data["legalBasis"])

        input_df = pd.DataFrame([{
            "totalCost": total_cost,
            "ecMaxContribution": ec_contrib,
            "duration": duration,
            "pillar": pillar,
            "countryCoor": input_data["countryCoor"],
            "fundingScheme": input_data["fundingScheme"],
            "numberOrg": number_org}])
        

        # "manual" feature engineering before feeding to preprossing.pkl 
        input_df["totalCostzero"] = (input_df["totalCost"] == 0).astype(int)
        input_df["contRatio"] = input_df["ecMaxContribution"] / input_df["totalCost"]
    
        
        # loading pkl files
        cb_best_model = joblib.load("models/cb_best_model.pkl")
        cb_preprocessor = joblib.load("models/cb_preprocessor.pkl")
    
    
        # applying preprocesinig and transformer
        X_transformed = cb_preprocessor.transform(input_df)
        prediction = cb_model.predict(X_transformed, cat_features=cb_preprocessor.cat_features)[0]
    
        logger.info(f"Prediction: {prediction}")                                          # logging

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": float(prediction)})
        }

except Exception as e:
    logger.error(f"Error: {str(e)}", exc_info=True)                                        # logging
    return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }