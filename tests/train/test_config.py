import pytest

from train.config import load_config


def test_load_config():
    config = load_config("config.yaml")
    assert config is not None
    assert "data" in config
