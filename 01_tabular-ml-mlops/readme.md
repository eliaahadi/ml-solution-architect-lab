# Project 1 – Tabular ML with MLOps flavor

Goal: Build an end to end tabular ML pipeline for a business problem such as credit risk scoring or housing price prediction.

This project covers:

- Data loading and exploratory data analysis (EDA)
- Feature engineering with scikit learn pipelines
- Model training and evaluation
- Experiment tracking with MLflow
- Serving the trained model via a FastAPI endpoint
- Basic tests for data and inference

## Architecture

High level flow:

1. `data.py` loads raw data from a CSV or local file.
2. `features.py` builds a `ColumnTransformer` for preprocessing.
3. `model.py` creates a `Pipeline` that joins preprocessing with the estimator.
4. `train.py` runs training, logs experiments to MLflow, and persists the best model.
5. `api.py` loads the persisted model and exposes a `/predict` endpoint.

## How to run

From the repo root, activate your virtualenv, then:

```bash
cd 01_tabular-ml-mlops

# train and log to MLflow
python -m src.train

# start API
uvicorn src.api:app --reload

# after you send a request
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.1, "feature_2": 3.14}}'
```

### Architect “checklist” for this project

As a solution architect, when you glance at this project, you want to be able to say:

1.. Data → Model → API is wired cleanly

- Data loader (data.py) uses a clear config path.
- Feature engineering is modular and testable (features.py).
- Model training is decoupled from serving (train.py vs api.py).
- Preprocessing is handled in a pipeline (features.py + model.py).
- Serving uses the exact same pipeline (no training/serving skew).

2.. Experiments are traceable

- MLflow logs every run with metrics and artifacts.
- You can compare experiments over time (e.g., baseline vs tuned model).

3.. It’s reproducible

- A new dev can:
- Clone repo
- pip install -r requirements.txt
- make train && make serve
- No hidden “magic” paths or manual steps.
  
4.. You can talk about tradeoffs

- Why a RandomForest as a baseline vs XGBoost/LightGBM.
- How you’d move MLflow from local filesystem → DB backend in prod.
- How you’d containerize this (later: Dockerfile) and deploy on AWS/Azure/GCP.
