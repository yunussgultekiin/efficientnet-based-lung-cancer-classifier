from __future__ import annotations
from pathlib import Path
from typing import NamedTuple
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.config import CONFIG, PROCESSED_DATA_DIR

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

class Dataloaders(NamedTuple):
    train: DataLoader
    valid: DataLoader
    test: DataLoader

def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

def build_dataset(split_dir: Path, image_size: int, train: bool) -> ImageFolder:
    return ImageFolder(root=str(split_dir), transform=build_transforms(image_size, train))

def create_dataloaders(
    processed_dir: Path = PROCESSED_DATA_DIR,
    image_size: int = CONFIG.image_size,
    batch_size: int = CONFIG.batch_size,
    num_workers: int = CONFIG.num_workers,
) -> Dataloaders:
    train_dataset = build_dataset(processed_dir / "train", image_size, train=True)
    valid_dataset = build_dataset(processed_dir / "valid", image_size, train=False)
    test_dataset = build_dataset(processed_dir / "test", image_size, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return Dataloaders(train=train_loader, valid=valid_loader, test=test_loader)