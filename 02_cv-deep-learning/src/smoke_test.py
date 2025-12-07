import base64
import io

import requests
from PIL import Image
from torchvision.datasets import CIFAR10

from . import config


def encode_image_to_base64(img: Image.Image) -> str:
    """Encode a PIL image as base64 string (PNG)."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main():
    # 1) Get a sample image + label from CIFAR-10
    dataset = CIFAR10(root=config.data_root, train=False, download=True)
    img, label_idx = dataset[0]  # first test image
    label_name = config.class_names[label_idx]

    # 2) Encode to base64
    img_b64 = encode_image_to_base64(img)

    # 3) Call /health
    health_resp = requests.get("http://127.0.0.1:8000/health")
    print("Health:", health_resp.status_code, health_resp.json())

    # 4) Call /predict with POST + JSON
    payload = {"image_base64": img_b64}
    pred_resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
    print("Predict status:", pred_resp.status_code)

    if pred_resp.status_code == 200:
        data = pred_resp.json()
        print("Prediction response:", data)
        print(f"True label: {label_name} (index {label_idx})")
    else:
        print("Error body:", pred_resp.text)


if __name__ == "__main__":
    main()