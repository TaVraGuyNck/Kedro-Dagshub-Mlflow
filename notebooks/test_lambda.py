import json
import lambda_function  # this assumes lambda_function.py is in the same directory

def test_lambda():
    # Define test input (user-provided values only)
    test_event = {
        "totalCost": 1000000.0,
        "ecMaxContribution": 750000.0,
        "duration": 900,
        "pillar": "Pillar 2 - Health",
        "countryCoor": "FR",
        "numberOrg": 4,

    }

    # Simulate AWS Lambda context (not used)
    context = {}

    # Call the lambda handler directly
    result = lambda_function.lambda_handler(test_event, context)

    # Print result nicely
    print("Response:")
    print(json.dumps(json.loads(result["body"]), indent=4))


if __name__ == "__main__":
    test_lambda()
