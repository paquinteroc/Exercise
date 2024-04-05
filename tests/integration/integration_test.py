import os
import requests
import pandas as pd


def test_score_endpoint():
    # Load test data from score.csv
    test_data = pd.read_csv(os.path.join("tests/data", "test_score.csv"))

    api_url = os.getenv("API_URL", "http://localhost:8888/score/")

    request_data = {"inputs": test_data.to_dict(orient="records")}

    response = requests.post(api_url, json=request_data)

    assert response.status_code == 200, "API response should be 200 OK"
    response_data = response.json()
    assert "predictions" in response_data, "Response should contain predictions"

    # Assert that there are exactly 2 predictions returned
    assert (
        len(response_data["predictions"]) == 2
    ), "There should be 2 predictions in the response"

    # Assert that each prediction is either 1 or 0
    for prediction in response_data["predictions"]:
        assert prediction in [0, 1], "Each prediction should be either 1 or 0"
