from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from . import config
from .transforms import get_train_transforms, get_val_transforms


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for CIFAR-10.

    Returns
    -------
    train_loader, val_loader : torch.utils.data.DataLoader
    """
    data_root = config.data_root
    data_root.mkdir(parents=True, exist_ok=True)

    full_train = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=get_train_transforms(),
    )

    # Simple train/val split (e.g., 45k / 5k)
    val_size = 5000
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # For validation, we want val transforms instead of train transforms
    val_dataset.dataset.transform = get_val_transforms()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader