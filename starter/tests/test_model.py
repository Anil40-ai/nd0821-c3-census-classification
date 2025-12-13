import sys
import os
import pandas as pd
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from starter.starter.ml.data import process_data  # noqa: E402


def test_process_data_outputs():
    df = pd.read_csv("starter/data/census.csv")

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == df.shape[0]
    assert len(y) == df.shape[0]
    assert encoder is not None
    assert lb is not None


def test_model_artifacts_exist():
    assert os.path.exists("starter/model/model.joblib")
    assert os.path.exists("starter/model/encoder.joblib")
    assert os.path.exists("starter/model/lb.joblib")


def test_model_prediction_shape():
    model = joblib.load("starter/model/model.joblib")
    encoder = joblib.load("starter/model/encoder.joblib")
    lb = joblib.load("starter/model/lb.joblib")

    df = pd.read_csv("starter/data/census.csv").head(5)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = model.predict(X)

    assert preds.shape[0] == df.shape[0]
