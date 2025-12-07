# 1) Pipeline snippet – quick “does it fit & predict?” test

# Goal: Prove your build_titanic_pipeline works end-to-end.

print("Snippet 1 – Build preprocessing + model pipeline for Titanic dataset")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def build_titanic_pipeline(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])




    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return clf

def smoke_test_pipeline():
    # Tiny Titanic-like dataset
    df = pd.DataFrame({
        "Pclass": [1, 3, 2, 3],
        "Sex": ["male", "female", "female", "male"],
        "Age": [22, 38, 26, 35],
        "Fare": [7.25, 71.2833, 7.925, 8.05],
        "Survived": [0, 1, 1, 0],
    })

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    clf = build_titanic_pipeline(X)
    clf.fit(X, y)
    preds = clf.predict(X)  

    print("\nPredictions:", preds)
    assert len(preds) == len(y), "Length of predictions does not match length of labels"

if __name__ == "__main__":
    smoke_test_pipeline()

print('''\nSnippet 1 Key phrases:
•	“I use a ColumnTransformer to keep numeric and categorical flows separate.”
•	“Pipelines guarantee the exact same preprocessing happens at inference.''')





# How to talk about this in an interview

# You don’t have to run anything live, but you can say something like:

# “Whenever I write these snippets, I pair them with a tiny smoke test:
# – for the pipeline, a 4–6 row DataFrame and len(preds) == len(y),
# – for MLflow, a mini run that prints accuracy and creates an mlruns directory,
# – for FastAPI, a TestClient call that asserts status_code == 200 and returns a prediction.
# That way I know my wiring is correct even before I plug everything into a larger system.”


# 2) MLflow snippet – quick “does it log metrics & artifacts?” test


print("\nSnippet 2 – MLflow logging test")
import mlflow
import mlflow.sklearn

def train_with_mlflow(X, y):
    mlflow.set_experiment("smoke_test_experiment")

    with mlflow.start_run():
        clf = build_titanic_pipeline(X)
        clf.fit(X, y)

        accuracy = clf.score(X, y)
        mlflow.log_metric("accuracy", accuracy)

        # Add input example to avoid the warning
        input_example = X.head(1)
        mlflow.sklearn.log_model(
            clf,
            name="titanic_model",
            input_example=input_example,
        )
        print(f"Logged accuracy: {accuracy}")


    return accuracy



from sklearn.model_selection import train_test_split

def smoke_test_mlflow():
    # Tiny Titanic-like dataset
    df = pd.DataFrame({
        "Pclass": [1, 3, 2, 3],
        "Sex": ["male", "female", "female", "male"],
        "Age": [22, 38, 26, 35],
        "Fare": [7.25, 71.2833, 7.925, 8.05],
        "Survived": [0, 1, 1, 0],
    })

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    accuracy = train_with_mlflow(X_train, y_train)
    print(f"\nTraining accuracy from MLflow run: {accuracy}\n")

if __name__ == "__main__":
    smoke_test_mlflow()


print('''Snippet 2 Key phrases:
	•	“Every training run gets its own MLflow run with params, metrics, and model artifacts.”
	•	“In production I’d point MLflow to a DB and object store instead of local disk.''')








# 3) FastAPI snippet – quick “does it return a prediction?” test
# Use when they ask: “How do you serve it?”

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from pathlib import Path

print("\nSnippet 3 – FastAPI prediction test")
app = FastAPI()

class Features(BaseModel):
    data: Dict[str, Any]    

MODEL_PATH = Path(__file__).resolve().parent / "models" / "tabular_model.joblib"
model = joblib.load(MODEL_PATH)  # Ensure tabular_model.joblib exists from prior training

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Features):
    X = pd.DataFrame([payload.data])
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}


def smoke_test_fastapi():
    client = TestClient(app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    print("\nHealth check response:", health_response.json())

    test_payload = {
        "data": {
            "PassengerId": 1,
            "Pclass": 3,
            "Name": "Test Passenger",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": "",
            "Embarked": "S"
        }
    }

    predict_response = client.post("/predict", json=test_payload)
    assert predict_response.status_code == 200
    assert "prediction" in predict_response.json()
    print("Status:", predict_response.status_code, "Prediction response:", predict_response.json())

if __name__ == "__main__":
    smoke_test_fastapi()


print('''\nSnippet 3 Key phrases:
    I’d use FastAPI’s TestClient to hit /predict in-process. If I get a 200 and a prediction field, I know my model load + request/response schema is correct.''')