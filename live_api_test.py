import requests

URL = "https://census-income-api-3bd2.onrender.com/predict"

payload = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 284582,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}

r = requests.post(URL, json=payload)

print("Status:", r.status_code)
print("Raw:", r.text)

if r.headers.get("content-type", "").startswith("application/json"):
    print("JSON:", r.json())
