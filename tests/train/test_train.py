import os
import pytest
from unittest.mock import MagicMock, patch
from train.train import main


@pytest.fixture
def config_path(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_content = """
    data:
        data_path: "data/train.csv"
        test_size: 0.2
    model:
        random_state: 42
        model_params:
            max_iter: 1000
        model_save_path: "models/trained_model.pkl"
    features:
        numeric: ["x1", "x2", "x4", "x5"]
        categorical: ["x3", "x6", "x7"]
    """
    with open(config_file, "w") as f:
        f.write(config_content)
    return config_file


@patch("train.train.mlflow")
def test_main(mock_mlflow, config_path):
    # Mocking mlflow functions
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

    # Running the main function
    main(config_path)

    # Asserting mlflow function calls
    mock_mlflow.log_metric.assert_called_once()
    mock_mlflow.sklearn.log_model.assert_called_once()
