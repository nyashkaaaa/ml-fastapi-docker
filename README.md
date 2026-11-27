# Wine Classifier — FastAPI + Docker + Streamlit + MLflow

A machine learning model (RandomForestClassifier) trained on the classic Wine dataset
from scikit-learn, deployed as a REST API with FastAPI and Docker, with a Streamlit
frontend for user interaction and MLflow for experiment tracking and model registry.

## Project Structure
```
ml-fastapi-docker/
├── main.py            # FastAPI application (prediction API)
├── front.py           # Streamlit frontend
├── train.py           # trains the model and logs it to MLflow
├── model.joblib       # saved trained model (used by the API)
├── requirements.txt   # Python dependencies
├── Dockerfile         # container instructions for the API
└── README.md          # project description and run steps
```

## Architecture
```
User → Streamlit (front.py, :8501) → FastAPI (main.py, :8000) → model.joblib
                                train.py → MLflow (tracking + registry, :5001)
```

## Model
- **Dataset:** Wine (scikit-learn), 178 samples, 13 numerical features, 3 classes
- **Algorithm:** RandomForestClassifier (n_estimators=100, random_state=42)
- **Test metrics:** accuracy = 1.0, F1-score (weighted) = 1.0 (80/20 train-test split)

## MLflow Integration
`train.py` logs every training run to MLflow:
- **Experiment:** `wine_classification`
- **Parameters:** `n_estimators`, `max_depth`, `random_state`
- **Metrics:** `accuracy`, `f1_score`
- **Artifacts:** `model.joblib` and the model in MLflow format
- **Model Registry:** model registered as `WineClassifier` (each run creates a new version)

## Run Locally
```bash
pip install -r requirements.txt

# 1. Train the model and log it to MLflow
python train.py

# 2. Start the API
uvicorn main:app --reload --port 8000

# 3. Start the frontend (in a new terminal)
streamlit run front.py

# 4. Open the MLflow UI (in a new terminal)
mlflow ui --port 5001
```

| Service        | URL                         |
|----------------|-----------------------------|
| Streamlit app  | http://localhost:8501       |
| FastAPI docs   | http://127.0.0.1:8000/docs  |
| MLflow UI      | http://127.0.0.1:5001       |

> On macOS port 5000 is used by AirPlay Receiver, so MLflow runs on port 5001.
> In the MLflow UI switch to **Model training** mode to see runs and registered models.

## Run API with Docker
```bash
docker build -t ml-fastapi .
docker run -p 8000:8000 ml-fastapi
```
The Streamlit frontend can then be started locally with `streamlit run front.py`.

## API Endpoints
| Method | Endpoint   | Description                                  |
|--------|------------|----------------------------------------------|
| GET    | `/`        | Health check — confirms the API is running   |
| POST   | `/predict` | Returns the predicted wine class             |
| GET    | `/docs`    | Interactive Swagger documentation            |

## Example Request
`POST /predict`
```json
{
  "alcohol": 14.23,
  "malic_acid": 1.71,
  "ash": 2.43,
  "alcalinity_of_ash": 15.6,
  "magnesium": 127,
  "total_phenols": 2.8,
  "flavanoids": 3.06,
  "nonflavanoid_phenols": 0.28,
  "proanthocyanins": 2.29,
  "color_intensity": 5.64,
  "hue": 1.04,
  "od280_od315_of_diluted_wines": 3.92,
  "proline": 1065
}
```

## Example Response
```json
{
  "prediction": 0,
  "class_name": "class_0"
}
```