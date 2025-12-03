from pathlib import Path

import pytest

from src.config import config
from src.data import load_data


def test_load_data_raises_if_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_path = tmp_path / "missing.csv"
    monkeypatch.setattr(config, "data_path", fake_path)
    with pytest.raises(FileNotFoundError):
        load_data(fake_path)