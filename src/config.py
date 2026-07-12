from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw" / "Data"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed" / "Data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
MODEL_DIR: Path = OUTPUT_DIR / "models"
PLOT_DIR: Path = OUTPUT_DIR / "plots"
REPORT_DIR: Path = OUTPUT_DIR / "reports"

SPLITS: tuple[str, ...] = ("train", "valid", "test")

CLASS_NAMES: tuple[str, ...] = (
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
)

CLASS_KEYWORDS: dict[str, str] = {
    "adenocarcinoma": "adenocarcinoma",
    "large.cell.carcinoma": "large.cell.carcinoma",
    "squamous.cell.carcinoma": "squamous.cell.carcinoma",
    "normal": "normal",
}

IMAGE_EXTENSION: str = ".png"
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")
NUM_CLASSES: int = len(CLASS_NAMES)

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_name: str = "efficientnet_b0"
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    early_stopping_patience: int = 5
    best_model_path: Path = MODEL_DIR / "best_efficientnet_b0.pth"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG: TrainingConfig = TrainingConfig()