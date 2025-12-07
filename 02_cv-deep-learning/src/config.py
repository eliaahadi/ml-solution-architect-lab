from pathlib import Path

# Project root is the folder containing this "src" directory
project_root = Path(__file__).resolve().parents[1]

# Data and model directories
data_root = project_root / "data"
model_dir = project_root / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# MLflow experiment name
mlflow_experiment = "cv-image-classification"

# Training hyperparameters
batch_size: int = 64
num_epochs: int = 2  # keep small for local experiments
learning_rate: float = 1e-3
num_workers: int = 2

# CIFAR-10 specifics
num_classes: int = 10
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]