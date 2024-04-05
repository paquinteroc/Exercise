import pandas as pd
from pandas import DataFrame


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df: DataFrame, label_column: str) -> DataFrame:
    """Preprocess the data."""
    if label_column in df.columns:
        df = df.pop(label_column)
    return df
