import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from .config import config
from .data import load_data, train_test_split
from .model import TaskType, build_pipeline, evaluate


def train(task_type: TaskType, target_col: str) -> Path:
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df, target_col=target_col)

    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run():
        model = build_pipeline(X_train, task_type=task_type)
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test, task_type=task_type)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        config.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = config.model_dir / config.model_name
        joblib.dump(model, model_path)

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.sklearn.log_model(model, "sklearn-model")

    return model_path


if __name__ == "__main__":
    # Titanic is a binary classification problem
    # Ensure your CSV has a 'Survived' column as the target
    task_type: TaskType = "classification"
    target_col = "Survived"

    model_path = train(task_type=task_type, target_col=target_col)
    print(f"Model saved to: {model_path}")