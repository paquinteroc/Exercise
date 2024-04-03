import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import mlflow
from config import load_config

mlflow.set_tracking_uri("http://localhost:5000")


def main(config_path):
    """
    Train a logistic regression model with specified preprocessing pipelines.

    Parameters:
    - config_path (str): Path to the configuration file in YAML format.
    """
    with mlflow.start_run():
        config = load_config(config_path)

        df = pd.read_csv(config["data"]["data_path"])
        label_column = config["data"].get("label_column", "y")
        df_label = df.pop(label_column)
        RANDOM_STATE = config["model"].get("random_state", 1337)

        numeric_features = config["features"]["numeric"]
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = config["features"]["categorical"]
        categorical_transformer = OneHotEncoder(handle_unknown="infrequent_if_exist")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=config["model"]["model_params"].get("max_iter", 10000),
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            df,
            df_label,
            test_size=config["data"].get("test_size", 0.2),
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        joblib.dump(clf, config["model"]["model_save_path"])
        training_accuracy = clf.score(X_train, y_train)
        mlflow.log_metric("training_accuracy", training_accuracy)

        # Log the model
        mlflow.sklearn.log_model(clf, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model based on a given configuration."
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")

    args = parser.parse_args()
    main(args.config_path)
