import httpx

url = "https://8icrl41qp8.execute-api.eu-west-3.amazonaws.com/prod/predict"
payload = {
    "totalCost": 100000,
    "ecMaxContribution": 750000.0,
    "duration": 900,
    "pillar": "Pillar 2 - Health",
    "countryCoor": "FR",
    "numberOrg": 4
}

response = httpx.post(url, json=payload, timeout=30.0)

print("Status:", response.status_code)
print("Raw response text:", response.text)

# Try to parse as JSON, safely
try:
    print("Parsed JSON:", response.json())
except Exception as e:
    print("Could not parse JSON:", e)