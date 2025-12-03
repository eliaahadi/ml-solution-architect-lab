from typing import Literal

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import pandas as pd

from .features import build_preprocessor


TaskType = Literal["classification", "regression"]


def build_pipeline(
    X: pd.DataFrame, task_type: TaskType = "classification"
) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(X)

    if task_type == "classification":
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    return pipe


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: TaskType = "classification",
) -> dict:
    y_pred = model.predict(X_test)

    if task_type == "classification":
        metric_value = accuracy_score(y_test, y_pred)
        return {"accuracy": metric_value}
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        return {"rmse": rmse, "r2": r2}