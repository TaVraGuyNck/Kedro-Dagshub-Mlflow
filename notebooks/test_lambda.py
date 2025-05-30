import json
from handler import lambda_handler  # adjust if needed

test_event = {
    "body": json.dumps({
        "legalBasis": "HORIZON.2.1",
        "countryCoor": "DE",
        "fundingScheme": "HORIZON-RIA",
        "totalCost": 2000000,
        "ecMaxContribution": 1500000,
        "numberOrg": 5,
        "duration": 730
    }),
    "isBase64Encoded": False
}

response = lambda_handler(test_event, None)

print("Lambda Response:")
print(response)

print("\nParsed Response Body:")
print(json.dumps(json.loads(response["body"]), indent=4))