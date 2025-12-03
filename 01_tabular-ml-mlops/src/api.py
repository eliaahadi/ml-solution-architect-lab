from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import config

app = FastAPI(title="Tabular ML Inference API", version="0.1.0")


class Features(BaseModel):
    features: Dict[str, Any]


_model = None


def load_model() -> Any:
    global _model
    if _model is None:
        model_path: Path = config.model_dir / config.model_name
        if not model_path.exists():
            raise RuntimeError(
                f"Model file not found at {model_path}. "
                "Run `python -m src.train` first."
            )
        _model = joblib.load(model_path)
    return _model


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Features) -> Dict[str, Any]:
    model = load_model()

    import pandas as pd

    X = pd.DataFrame([payload.features])
    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"prediction": pred[0]}