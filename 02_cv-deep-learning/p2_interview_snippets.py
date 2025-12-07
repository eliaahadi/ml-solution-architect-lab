'''I built an end-to-end image classification pipeline on CIFAR-10 using a small CNN in PyTorch.

If I map it to my CATER framework:
C is collecting CIFAR-10 via torchvision and setting up train/val splits.
A is arranging the data with augmentation and normalization transforms plus DataLoaders.
T is a PyTorch training loop in `train.py` that logs loss and accuracy to MLflow and saves the best model as `models/cnn.pt`.
E is evaluation in MLflow UI, where I compare validation curves across runs and hyperparameters.
R is rele```asing the trained CNN behind a FastAPI `/predict` endpoint that takes a base64 image and returns a class label and confidence.'''

"""
Interview smoke tests for Project 2 (CV / CNN).

This file shows how I would quickly test the 3 core snippets:

1) Model definition (CnnClassifier) – does a forward pass work and have the right shape?
2) Training + MLflow logging – can I log a tiny run with loss to MLflow?
3) FastAPI predict endpoint – can I hit /health and /predict and get a valid response?
"""

import base64
import io

import torch
from torch import nn
from fastapi.testclient import TestClient
from torchvision.datasets import CIFAR10

from src import config
from src.model import CnnClassifier, create_model
from src.data import get_dataloaders
from src.transforms import get_val_transforms
from src.api import app

import mlflow
import mlflow.pytorch


# ---------- Snippet 1: CNN forward pass ----------

def test_snippet1_model_forward():
    """Smoke test for the CnnClassifier forward pass."""
    model = CnnClassifier(num_classes=config.num_classes)
    x = torch.randn(4, 3, 32, 32)  # batch of 4 CIFAR-like images

    with torch.no_grad():
        y = model(x)

    print("Snippet 1 - CnnClassifier output shape:", y.shape)
    assert y.shape == (4, config.num_classes), "Unexpected output shape for CNN"


# ---------- Snippet 2: Tiny MLflow logging run ----------

def test_snippet2_mlflow_logging():
    """
    Smoke test for MLflow logging.

    Uses a single batch from the train loader to:
    - run a forward pass
    - compute a loss
    - log a metric and the model to MLflow
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get one batch of real data
    train_loader, _ = get_dataloaders()
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    model = create_model(device=device)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        mlflow.log_param("smoke_test", "snippet2_single_batch")
        mlflow.log_metric("loss", float(loss.item()))

        mlflow.pytorch.log_model(model, "model")

    print("Snippet 2 - MLflow logging complete, loss:", float(loss.item()))


# ---------- Snippet 3: FastAPI /health and /predict ----------

def _encode_pil_to_base64(img) -> str:
    """Helper: encode a PIL image to base64 string (PNG)."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_snippet3_fastapi_predict():
    """
    Smoke test for FastAPI + model serving.

    - Uses TestClient to call /health
    - Grabs one image from CIFAR-10 test set
    - Encodes it to base64 and POSTs to /predict
    """
    client = TestClient(app)

    # 1) Health check
    resp_health = client.get("/health")
    print("Snippet 3 - /health status:", resp_health.status_code, resp_health.json())
    assert resp_health.status_code == 200

    # 2) Get a sample CIFAR-10 test image (PIL)
    dataset = CIFAR10(root=config.data_root, train=False, download=True)
    img, label_idx = dataset[0]
    true_label = config.class_names[label_idx]

    img_b64 = _encode_pil_to_base64(img)

    # 3) Call /predict
    payload = {"image_base64": img_b64}
    resp_pred = client.post("/predict", json=payload)
    print("Snippet 3 - /predict status:", resp_pred.status_code)

    assert resp_pred.status_code == 200, f"Predict error: {resp_pred.text}"

    data = resp_pred.json()
    print("Snippet 3 - prediction response:", data)
    print(f"Snippet 3 - true label: {true_label} (index {label_idx})")

    assert "class_name" in data
    assert "confidence" in data


# ---------- Run all three when executed directly ----------

if __name__ == "__main__":
    print("=== Snippet 1: CNN forward ===")
    test_snippet1_model_forward()

    print("\n=== Snippet 2: MLflow logging ===")
    test_snippet2_mlflow_logging()

    print("\n=== Snippet 3: FastAPI prediction ===")
    test_snippet3_fastapi_predict()