import os
import pytest
import requests
import pandas as pd

# Load test data from score.csv
test_data = pd.read_csv(os.path.join("data", "score.csv"))


# Define API URL based on environment variable or default to localhost
@pytest.fixture
def api_url():
    return os.getenv("API_URL", "http://localhost:8888/score/")


def test_score_endpoint(api_url):
    for _, row in test_data.iterrows():
        payload = {"data": row.tolist()}  # Convert row to list
        response = requests.post(api_url, json=payload)
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert all(0 <= pred <= 1 for pred in predictions)
