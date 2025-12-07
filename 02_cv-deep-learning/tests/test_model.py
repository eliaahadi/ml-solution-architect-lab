import torch

from src.model import CnnClassifier
from src import config


def test_cnn_forward_shape():
    model = CnnClassifier(num_classes=config.num_classes)
    x = torch.randn(4, 3, 32, 32)  # batch of 4 CIFAR-like images
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, config.num_classes)