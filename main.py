from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Wine Classifier API")
model = joblib.load("model.joblib")

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43,
                "alcalinity_of_ash": 15.6, "magnesium": 127,
                "total_phenols": 2.8, "flavanoids": 3.06,
                "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29,
                "color_intensity": 5.64, "hue": 1.04,
                "od280_od315_of_diluted_wines": 3.92, "proline": 1065
            }]
        }
    }

@app.get("/")
def root():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(wine: WineFeatures):
    features = [list(wine.model_dump().values())]
    prediction = int(model.predict(features)[0])
    return {"prediction": prediction, "class_name": f"class_{prediction}"}