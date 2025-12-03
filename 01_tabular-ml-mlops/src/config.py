from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_path: Path = project_root / "data" / "train.csv"  # you can change this
    model_dir: Path = project_root / "models"
    model_name: str = "tabular_model.joblib"
    mlflow_experiment: str = "tabular-ml-mlops"


config = Config()