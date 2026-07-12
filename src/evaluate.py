from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from src.config import CLASS_NAMES, CONFIG, NUM_CLASSES, PLOT_DIR, REPORT_DIR
from src.dataset import create_dataloaders
from src.model import load_trained_model

@dataclass(slots=True)
class EvaluationResult:
    y_true: list[int]
    y_pred: list[int]

class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    def predict(self, dataloader: DataLoader) -> EvaluationResult:
        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = outputs.argmax(dim=1).cpu().tolist()

                y_true.extend(labels.tolist())
                y_pred.extend(predictions)

        return EvaluationResult(y_true=y_true, y_pred=y_pred)

    def save_classification_report(self, result: EvaluationResult, report_dir: Path) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        report = classification_report(
            result.y_true,
            result.y_pred,
            target_names=list(CLASS_NAMES),
        )
        (report_dir / "classification_report.txt").write_text(report)

    def save_confusion_matrix(self, result: EvaluationResult, plot_dir: Path) -> None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        matrix = confusion_matrix(result.y_true, result.y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(CLASS_NAMES))

        fig, ax = plt.subplots(figsize=(8, 8))
        display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
        fig.tight_layout()
        fig.savefig(plot_dir / "confusion_matrix.png")
        plt.close(fig)

class EvaluationPipeline:
    def __init__(self) -> None:
        self.device = CONFIG.device
        self.dataloaders = create_dataloaders()
        self.model = load_trained_model(CONFIG.best_model_path, NUM_CLASSES, self.device)
        self.evaluator = Evaluator(model=self.model, device=self.device)

    def run(self) -> None:
        result = self.evaluator.predict(self.dataloaders.test)
        self.evaluator.save_classification_report(result, REPORT_DIR)
        self.evaluator.save_confusion_matrix(result, PLOT_DIR)