import joblib
from sklearn.linear_model import LogisticRegression
from data_preparation import load_data, preprocess_data


def train_model(X_train, y_train, **kwargs):
    """
    Trains the Logistic Regression model with the given training data and parameters.

    :param X_train: Training features
    :param y_train: Training labels
    :param kwargs: Additional arguments for the Logistic Regression model
    :return: Trained model
    """
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model


def main(config_path="config.yaml"):
    """
    Main training script that loads data, preprocesses it, trains the model, and saves the trained model.
    """
    # Assuming `load_config` is adjusted to be within this file or imported from `config.py`
    config = load_config(config_path)
    data = load_data(config["data_path"])
    data = preprocess_data(data)

    X_train, y_train = data.drop("target", axis=1), data["target"]
    model = train_model(X_train, y_train, max_iter=config["model_params"]["max_iter"])

    joblib.dump(model, config["model_save_path"])


if __name__ == "__main__":
    main()
