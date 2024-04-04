import os
import requests
import pandas as pd

# Load test data from score.csv
test_data = pd.read_csv(os.path.join("data", "score.csv"))

# Define API URL based on environment variable or default to localhost
api_url = os.getenv("API_URL", "http://localhost:8888/score/")

# Select a single row for testing
single_row = test_data.iloc[0]
print(single_row)
print(single_row.to_list())
payload = {"data": single_row}  # Convert row to list
response = requests.post(api_url, json=payload)
print(response.text)
if response.status_code == 200:
    predictions = response.json()["predictions"]
    if all(0 <= pred <= 1 for pred in predictions):
        print("Test passed: Predictions are within range [0, 1].")
    else:
        print("Test failed: Predictions are not within range [0, 1].")
else:
    print("Test failed: Unexpected status code:", response.status_code)
