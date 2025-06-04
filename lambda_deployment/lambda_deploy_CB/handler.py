import json
import joblib
import pandas as pd
import base64
from catboost import Pool
import logging
from startupdelay_horizon.pipelines.cb_preprocessing.nodes import CatBoostPreprocessor

# instantiating logger - all to be logged from INFO level onwards
logger = logging.getLogger()                                                                 
logger.setLevel(logging.INFO)

# function - calculation safe_ratio - idem general preprocessing pipeline
def safe_ratio(row):
        if pd.notnull(row["ecMaxContribution"]) and pd.notnull(row["totalCost"]) and row["totalCost"] != 0:
            return float(row["ecMaxContribution"]) / float(row["totalCost"])
        return None

# lambda handler     
def lambda_handler(event, context):

    # parsing input_data from raw event to python dictionry (input_data)
    # when received from http api - is in json string (event = "body")
    # when lambda is invoked in other way - not in json string (input_data = event)

    try:
        # json string (%s = lazy string formatting) to cloudwatch in case of issues
        logger.info("raw event: %s", json.dumps(event))                                             

        # if event has key called "body" or not (if comes via http api, or e.g. via test)
        if "body" in event:
            body = event["body"]

            try: 
                # in case event (and body) is encoded base64 => to be decoded
                if event.get("isBase64Encoded"):
                    body = base64.b64decode(body).decode("utf-8")

                #  if body = a string (not a dictionary) => load body from the string as a dict
                # else dict = input_data
                if isinstance(body, str): 
                    input_data = json.loads(body)
                else: 
                    input_data = body

            except Exception as e:
                return {
                    "statusCode": 400,  
                    "headers": { "Content-Type": "application/json" },                                                       #logging
                    "body": json.dumps({"error": f"Lambda did not receive a valid data format from htt api: {str(e)}"})
                }
        else:
            input_data = event

        # ensuring all necessary data input is present 
        required_fields = ["totalCost", "ecMaxContribution", "duration",
            "pillar", "countryCoor", "numberOrg", "fundingScheme"]

        for field in required_fields: 
            if field not in input_data:
                logger.warning(f"Missing required field: {field} in input: {input_data}")
                return {
                    "statusCode": 400, 
                    "headers": { "Content-Type": "application/json" },                                                        #logging
                    "body": json.dumps({"error": f"Missing required field: {field}"})
                }

        # formatting of numerical data to float or int
        try:
            total_cost = float(input_data["totalCost"])
            ec_contrib = float(input_data["ecMaxContribution"])
            duration = int(input_data["duration"])
            number_org = int(input_data["numberOrg"])

        except Exception as e:
            logger.error(f"Numeric validation failed: {e}", exc_info=True)
            return {
                "statusCode": 400,
                "headers": { "Content-Type": "application/json" },
                "body": json.dumps({"error": "Numeric inputs must be valid numbers."})
            }

        # validating if numerical data is not negative or 0:
        if total_cost < 0 or ec_contrib <= 0 or duration <= 0 or number_org <= 0:
            logger.warning(f"Negative value found in numeric input: {input_data}")
            return {
                "statusCode": 400,
                "headers": { "Content-Type": "application/json" },
                "body": json.dumps({"error": "Numeric inputs must be non-negative and not 0."})
            }

    
        # assembling all features in dataframe for input to CBmodel  
        data_for_lambda = pd.DataFrame([{
            "totalCost": total_cost,
            "ecMaxContribution": ec_contrib,
            "duration": duration,
            "pillar": input_data["pillar"],
            "countryCoor": input_data["countryCoor"],
            "fundingScheme": input_data["fundingScheme"],
            "numberOrg": number_org}])
        

        # calculating totalCostzero and contRatio and adding to data_for_lambda 
        data_for_lambda["totalCostzero"] = (data_for_lambda["totalCost"] == 0).astype(int)
        
        try: 
            data_for_lambda["contRatio"] = data_for_lambda.apply(safe_ratio, axis=1) 
        except Exception as e: 
            logger.error(f"Error calculating contRatio: {e}", exc_info=True)
            
            return {
                "statusCode": 400,
                "headers": { "Content-Type": "application/json" },
                "body": json.dumps({"error": f"Failed to calculate contRatio due invalid value provided\
                                     for Total Cost and or Maximum EC contribution: {str(e)}"})
                }
    
        
        # loading model and encoders 
        try:
            cb_best_model = joblib.load("models/cb_best_model.pkl")
            cb_preprocessor = joblib.load("models/cb_preprocessor.pkl")
            cb_preprocessor.cat_features = ["pillar", "countryCoor", "fundingScheme"]
            
        except Exception as e: 
            logger.error(f"Model/Preprocessor .pkl not loaded.", exc_info=True)
            return {
                "statusCode": 500,
                "headers": { "Content-Type": "application/json" },
                "body": json.dumps({"error": f"Failed to load model or encoder: {str(e)}"})
                }
        
        # extra verification whether all columns present for prediction (set does not take order into account)
        missing_cols = set(cb_preprocessor.feature_names_) - set(data_for_lambda.columns)
        if missing_cols:
             return {
                "statusCode": 400,
                "headers": { "Content-Type": "application/json" },
                "body": json.dumps({"error": f"Missing data in data input"})
                }
        
        # reindexing columns to the order of columns when model was trained
        data_for_lambda = data_for_lambda.reindex(columns=cb_preprocessor.feature_names_)

        # applying preprocesinig and transformer
        X_transformed = cb_preprocessor.transform(data_for_lambda)

        # blockages -> need to tell catboost which are categorical features
        pool = Pool(X_transformed, cat_features=cb_preprocessor.cat_features)

        # generating prediction by running model vs. pool input data
        prediction = cb_best_model.predict(pool)[0]
    
        logger.info(f"Prediction: {prediction}")                                         

        return {
            "statusCode": 200,
            "headers": { "Content-Type": "application/json" },
            "body": json.dumps({"prediction": float(prediction)})
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)                                        
        return {
            "statusCode": 500,
            "headers": { "Content-Type": "application/json" },
            "body": json.dumps({"error": str(e)})
        }