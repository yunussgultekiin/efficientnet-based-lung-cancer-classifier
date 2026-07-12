from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

def build_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()

def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> Optimizer:
    return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False

def unfreeze_backbone(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = True

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

def count_total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> nn.Module:
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def load_trained_model(checkpoint_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = build_model(num_classes=num_classes, pretrained=False)
    return load_checkpoint(model, checkpoint_path, device)