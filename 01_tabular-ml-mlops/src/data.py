from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import config


def load_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load raw data from CSV.

    For now this expects a CSV at config.data_path.
    You can swap in your own dataset later.
    """
    if path is None:
        path = config.data_path

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            "Place your dataset there or update config.data_path."
        )

    df = pd.read_csv(path)
    return df


def train_test_split(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split as sk_train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test