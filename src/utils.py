from __future__ import annotations
import random
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass(slots=True)
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_score: float | None = field(default=None, init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

@dataclass(slots=True)
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    train_f1_macro: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    valid_accuracy: list[float] = field(default_factory=list)
    valid_f1_macro: list[float] = field(default_factory=list)

    def append(
        self,
        train_loss: float,
        train_accuracy: float,
        train_f1_macro: float,
        valid_loss: float,
        valid_accuracy: float,
        valid_f1_macro: float,
    ) -> None:
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.train_f1_macro.append(train_f1_macro)
        self.valid_loss.append(valid_loss)
        self.valid_accuracy.append(valid_accuracy)
        self.valid_f1_macro.append(valid_f1_macro)

    def plot(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(self.train_loss) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs, self.train_loss, label="train")
        axes[0].plot(epochs, self.valid_loss, label="valid")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("epoch")
        axes[0].legend()

        axes[1].plot(epochs, self.train_accuracy, label="train")
        axes[1].plot(epochs, self.valid_accuracy, label="valid")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].legend()

        axes[2].plot(epochs, self.train_f1_macro, label="train")
        axes[2].plot(epochs, self.valid_f1_macro, label="valid")
        axes[2].set_title("F1 (macro)")
        axes[2].set_xlabel("epoch")
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(output_dir / "training_history.png")
        plt.close(fig)