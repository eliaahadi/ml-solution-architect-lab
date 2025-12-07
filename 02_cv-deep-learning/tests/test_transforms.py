import numpy as np
from PIL import Image

from src.transforms import get_val_transforms


def test_val_transform_output_shape():
    transforms = get_val_transforms()

    # Create a random 32x32 RGB image
    arr = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)

    tensor = transforms(img)
    assert tensor.shape == (3, 32, 32)