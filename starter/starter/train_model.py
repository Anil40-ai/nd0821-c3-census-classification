import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from starter.starter.ml.data import process_data

data = pd.read_csv("starter/data/census.csv")

train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "starter/model/model.joblib")
joblib.dump(encoder, "starter/model/encoder.joblib")
joblib.dump(lb, "starter/model/lb.joblib")
