# Wine Classifier API — FastAPI + Docker

A machine learning model (RandomForestClassifier) trained on the classic Wine dataset
from scikit-learn and deployed as a REST API using FastAPI and Docker.

## Project Structure
```
ml-fastapi-docker/
├── main.py            # FastAPI application
├── train.py           # trains and saves the model
├── model.joblib       # saved trained model
├── requirements.txt   # Python dependencies
├── Dockerfile         # container instructions
└── README.md          # project description and run steps
```

## Model
- **Dataset:** Wine (scikit-learn), 178 samples, 13 numerical features, 3 classes
- **Algorithm:** RandomForestClassifier
- **Test accuracy:** 1.0 (80/20 train-test split)

## API Endpoints
| Method | Endpoint   | Description                                  |
|--------|------------|----------------------------------------------|
| GET    | `/`        | Health check — confirms the API is running   |
| POST   | `/predict` | Returns the predicted wine class             |
| GET    | `/docs`    | Interactive Swagger documentation            |

## Run Locally
```bash
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```
Open: http://127.0.0.1:8000/docs

## Run with Docker
```bash
docker build -t ml-fastapi .
docker run -p 8000:8000 ml-fastapi
```
Open: http://127.0.0.1:8000/docs

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