import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference


app = FastAPI()

model = None
encoder = None
lb = None


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.on_event("startup")
def load_artifacts() -> None:
    global model, encoder, lb
    model = joblib.load("starter/model/model.joblib")
    encoder = joblib.load("starter/model/encoder.joblib")
    lb = joblib.load("starter/model/lb.joblib")


@app.get("/")
def read_root() -> dict:
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
def predict(data: CensusData) -> dict:
    input_dict = data.dict(by_alias=True)
    df = pd.DataFrame([input_dict])

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
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = model.predict(X)[0]
    salary = ">50K" if pred == 1 else "<=50K"

    return {"prediction": salary}
