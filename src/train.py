from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from src.config import CONFIG, NUM_CLASSES, PLOT_DIR
from src.dataset import create_dataloaders
from src.model import build_criterion, build_model, build_optimizer, save_checkpoint
from src.utils import EarlyStopping, TrainingHistory, set_seed

@dataclass(slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    f1_macro: float

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: Optimizer,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train_one_epoch(self, dataloader: DataLoader) -> EpochMetrics:
        self.model.train()
        return self._run_epoch(dataloader, train=True)

    def evaluate(self, dataloader: DataLoader) -> EpochMetrics:
        self.model.eval()
        return self._run_epoch(dataloader, train=False)

    def _run_epoch(self, dataloader: DataLoader, train: bool) -> EpochMetrics:
        total_loss = 0.0
        total_samples = 0
        all_labels: list[int] = []
        all_predictions: list[int] = []

        with torch.set_grad_enabled(train):
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                predictions = outputs.argmax(dim=1)
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())
                total_samples += labels.size(0)
                total_loss += loss.item() * labels.size(0)

        correct_predictions = sum(
            prediction == label for prediction, label in zip(all_predictions, all_labels)
        )

        return EpochMetrics(
            loss=total_loss / total_samples,
            accuracy=correct_predictions / total_samples,
            f1_macro=f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        )

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        checkpoint_path: Path,
        early_stopping_patience: int,
    ) -> TrainingHistory:
        history = TrainingHistory()
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_f1_macro = 0.0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader)
            valid_metrics = self.evaluate(valid_loader)

            history.append(
                train_loss=train_metrics.loss,
                train_accuracy=train_metrics.accuracy,
                train_f1_macro=train_metrics.f1_macro,
                valid_loss=valid_metrics.loss,
                valid_accuracy=valid_metrics.accuracy,
                valid_f1_macro=valid_metrics.f1_macro,
            )

            print(
                f"epoch {epoch}/{epochs} "
                f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} train_f1={train_metrics.f1_macro:.4f} "
                f"valid_loss={valid_metrics.loss:.4f} valid_acc={valid_metrics.accuracy:.4f} valid_f1={valid_metrics.f1_macro:.4f}"
            )

            if valid_metrics.f1_macro > best_f1_macro:
                best_f1_macro = valid_metrics.f1_macro
                save_checkpoint(self.model, checkpoint_path)

            if early_stopping.step(valid_metrics.f1_macro):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return history

class TrainingPipeline:
    def __init__(self) -> None:
        set_seed(CONFIG.seed)
        self.device = CONFIG.device
        self.dataloaders = create_dataloaders()

        self.model = build_model(num_classes=NUM_CLASSES)
        self.criterion = build_criterion()
        self.optimizer = build_optimizer(
            self.model,
            learning_rate=CONFIG.learning_rate,
            weight_decay=CONFIG.weight_decay,
        )
        self.trainer = Trainer(
            model=self.model,
            device=self.device,
            criterion=self.criterion,
            optimizer=self.optimizer,
        )

    def run(self) -> TrainingHistory:
        history = self.trainer.fit(
            train_loader=self.dataloaders.train,
            valid_loader=self.dataloaders.valid,
            epochs=CONFIG.epochs,
            checkpoint_path=CONFIG.best_model_path,
            early_stopping_patience=CONFIG.early_stopping_patience,
        )
        history.plot(PLOT_DIR)
        return history