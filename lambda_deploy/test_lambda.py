import json
from lambda_function import lambda_handler

event = {
    "body": json.dumps({
        "totalCost": 100000,
        "ecMaxContribution": 50000,
        "duration": 24,
        "legalBasis": "HORIZON.3.1",
        "countryCoor":"FR",
        "numberOrg": 5
    }),
    "isBase64Encoded": False
}

response = lambda_handler(event, None)
print("Response:")
print(json.dumps(response, indent=2))