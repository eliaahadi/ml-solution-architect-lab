from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import config
from .data import get_dataloaders
from .model import create_model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

    epoch_loss = running_loss / total
    return epoch_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders()

    model = create_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    mlflow.set_experiment(config.mlflow_experiment)

    best_val_acc = 0.0
    best_model_path = config.model_dir / "cnn.pt"

    with mlflow.start_run():
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("num_epochs", config.num_epochs)

        for epoch in range(1, config.num_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch}/{config.num_epochs} "
                f"- train_loss: {train_loss:.4f} "
                f"- val_loss: {val_loss:.4f} "
                f"- val_acc: {val_acc:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

        # Log final model to MLflow (can be the last-epoch model)
        mlflow.pytorch.log_model(model, "model")

    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    train()