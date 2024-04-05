import os
import requests
import pandas as pd

# Load test data from score.csv
test_data = pd.read_csv(os.path.join("tests/data", "score.csv"))

# Define API URL based on environment variable or default to localhost
api_url = os.getenv("API_URL", "http://localhost:8888/score/")


request_data = {"inputs": test_data.head(2).to_dict(orient="records")}
response = requests.post(api_url, json=request_data)


print(response.text)
