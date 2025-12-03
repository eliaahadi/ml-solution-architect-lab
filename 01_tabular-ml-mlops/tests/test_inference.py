import pandas as pd

from src.model import build_pipeline


def test_pipeline_can_fit_and_predict():
    # Tiny fake dataset for sanity check
    df = pd.DataFrame(
        {
            "num_feature": [1.0, 2.0, 3.0, 4.0],
            "cat_feature": ["a", "b", "a", "b"],
            "target": [0, 1, 0, 1],
        }
    )

    X = df.drop(columns=["target"])
    y = df["target"]

    pipeline = build_pipeline(X, task_type="classification")
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == len(y)