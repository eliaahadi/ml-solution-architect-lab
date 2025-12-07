from torchvision import transforms as T

# CIFAR-10 normalization stats
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_train_transforms():
    """Transforms for training (augment + normalize)."""
    return T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_val_transforms():
    """Transforms for validation / test (normalize only)."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )