from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

# Store MLflow data in the same database that `mlflow ui` reads from
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create the experiment (or reuse it if it already exists)
mlflow.set_experiment("wine_classification")

# Load data and split into train/test sets
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model hyperparameters (same as the original model)
params = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42,
}

# Everything inside this block is recorded as one MLflow run
with mlflow.start_run(run_name="random_forest"):
    # Train the model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")  # wine has 3 classes

    # 1. Log hyperparameters
    mlflow.log_params(params)

    # 2. Log evaluation metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # 3. Save the model locally (used by FastAPI) and log it as an artifact
    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")

    # 4. Log the model in MLflow format and register it in the Model Registry
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_test[:1],
        registered_model_name="WineClassifier",
        # We trained this model ourselves, so its tree structure is safe to load
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("Accuracy:", accuracy)
    print("F1-score:", f1)