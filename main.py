import os
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml.data import process_data
from ml.model import inference

app = FastAPI()

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

model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
with open(model_path, "rb") as f:
    model = pickle.load(f)

lb_path = os.path.join(os.path.dirname(__file__), 'model', 'label_binarizer.pkl')
with open(lb_path, "rb") as f:
    lb = pickle.load(f)

encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'encoder.pkl')
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)


@app.get("/")
def welcome() -> dict:
    return {"message": "Welcome to the ML model inference API"}


class InferenceBody(BaseModel):
    # Columns: age,workclass,fnlgt,education,education-num,marital-status,occupation,relationship,race,sex,
    #          capital-gain,capital-loss,hours-per-week,native-country
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
        schema_extra = {
            "example": {
                "age": 36,
                "workclass": "Private",
                "fnlgt": 484024,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Divorced",
                "occupation": "Machine-op-inspct",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.post("/predict")
def predict(request: InferenceBody) -> dict:
    try:
        data_dict = request.dict(by_alias=True)
        data_df = pd.DataFrame([data_dict])
        X, _, _, _ = process_data(data_df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

        prediction = inference(model, X)
        return {"input": data_dict, "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)