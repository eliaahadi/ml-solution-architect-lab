import base64
import io

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from . import config
from .model import CnnClassifier
from .transforms import get_val_transforms

app = FastAPI(title="CV Image Classification API")

# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CnnClassifier(num_classes=config.num_classes)
model_path = config.model_dir / "cnn.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

val_transforms = get_val_transforms()


class ImagePayload(BaseModel):
    image_base64: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: ImagePayload):
    # Decode base64 image
    image_bytes = base64.b64decode(payload.image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply transforms and run through model
    tensor = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = probs.max(dim=1)

    class_index = int(pred_idx.item())
    class_name = config.class_names[class_index]
    confidence = float(conf.item())

    return {
        "class_index": class_index,
        "class_name": class_name,
        "confidence": confidence,
    }