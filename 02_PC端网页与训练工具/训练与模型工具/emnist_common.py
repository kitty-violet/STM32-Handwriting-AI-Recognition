#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared helpers for EMNIST ByClass CNN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST


ROOT = Path(__file__).resolve().parent
EMNIST_CLASSES = EMNIST.classes_split_dict["byclass"]
NUM_EMNIST_CLASSES = len(EMNIST_CLASSES)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fix_emnist_orientation(data: torch.Tensor) -> torch.Tensor:
    """Convert torchvision EMNIST storage orientation to normal upright glyphs."""
    return torch.rot90(data, k=-1, dims=(-2, -1)).flip(-1).contiguous()


class EMNISTTensorDataset(Dataset):
    def __init__(
        self,
        root: Path,
        train: bool,
        download: bool,
        limit: int = 0,
    ) -> None:
        raw = EMNIST(root=str(root), split="byclass", train=train, download=download)
        data = fix_emnist_orientation(raw.data)
        targets = raw.targets.long()
        if limit > 0:
            data = data[:limit]
            targets = targets[:limit]
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data[index].unsqueeze(0).float().div(255.0)
        return image, self.targets[index]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EMNISTCNN(nn.Module):
    """CNN for 62-class EMNIST ByClass recognition."""

    def __init__(self, num_classes: int = NUM_EMNIST_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 192),
            ConvBlock(192, 192),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    batch = images.size(0)
    angles = torch.empty(batch, device=images.device).uniform_(-10.0, 10.0) * np.pi / 180.0
    scales = torch.empty(batch, device=images.device).uniform_(0.90, 1.10)
    tx = torch.empty(batch, device=images.device).uniform_(-2.0, 2.0) * (2.0 / 28.0)
    ty = torch.empty(batch, device=images.device).uniform_(-2.0, 2.0) * (2.0 / 28.0)
    cos_v = torch.cos(angles) / scales
    sin_v = torch.sin(angles) / scales
    theta = torch.zeros(batch, 2, 3, device=images.device)
    theta[:, 0, 0] = cos_v
    theta[:, 0, 1] = -sin_v
    theta[:, 1, 0] = sin_v
    theta[:, 1, 1] = cos_v
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def class_text(class_id: int) -> str:
    return EMNIST_CLASSES[int(class_id)]


def safe_class_text(class_id: int) -> str:
    text = class_text(class_id)
    if text.isalnum():
        return text
    return f"class{class_id:02d}"


def image_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((28, 28), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    correct: int
    total: int
    per_class: list[dict[str, object]]
    top_confusions: list[dict[str, object]]
    avg_infer_ms: float


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = NUM_EMNIST_CLASSES,
    class_names: list[str] | None = None,
) -> EvalResult:
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    class_total = torch.zeros(num_classes, dtype=torch.long)
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    infer_seconds = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_seconds += time.perf_counter() - start
            loss = loss_func(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += int((preds == labels).sum().item())
            total_count += int(labels.size(0))

            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()
            for label, pred in zip(labels_cpu.tolist(), preds_cpu.tolist()):
                class_total[label] += 1
                class_correct[label] += int(label == pred)
                confusion[label, pred] += 1

    per_class = []
    for class_id in range(num_classes):
        total = int(class_total[class_id].item())
        correct = int(class_correct[class_id].item())
        label_text = class_names[class_id] if class_names is not None else class_text(class_id)
        per_class.append(
            {
                "class_id": class_id,
                "label": label_text,
                "total": total,
                "correct": correct,
                "accuracy": (correct / total) if total else None,
            }
        )

    confusion_rows = []
    for label in range(num_classes):
        for pred in range(num_classes):
            if label != pred and confusion[label, pred] > 0:
                confusion_rows.append(
                    {
                        "label_id": label,
                        "label": class_names[label] if class_names is not None else class_text(label),
                        "pred_id": pred,
                        "pred": class_names[pred] if class_names is not None else class_text(pred),
                        "count": int(confusion[label, pred].item()),
                    }
                )
    confusion_rows.sort(key=lambda item: item["count"], reverse=True)

    return EvalResult(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
        correct=total_correct,
        total=total_count,
        per_class=per_class,
        top_confusions=confusion_rows[:30],
        avg_infer_ms=(infer_seconds * 1000.0 / total_count),
    )


def save_checkpoint(path: Path, model: EMNISTCNN, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(getattr(model, "emnist_classes", EMNIST_CLASSES))
    num_classes = len(classes)
    torch.save(
        {
            "model_name": "EMNISTCNN",
            "split": "byclass",
            "classes": classes,
            "input_shape": [1, 28, 28],
            "num_classes": num_classes,
            "state_dict": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> EMNISTCNN:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = EMNISTCNN(num_classes=int(checkpoint.get("num_classes", NUM_EMNIST_CLASSES)))
    model.load_state_dict(checkpoint["state_dict"])
    model.emnist_classes = checkpoint.get("classes", EMNIST_CLASSES)
    model.to(device)
    model.eval()
    return model

