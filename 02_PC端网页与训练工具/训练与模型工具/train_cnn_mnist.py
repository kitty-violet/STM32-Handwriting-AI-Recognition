#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a convolutional neural network on MNIST and save PyTorch weights."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_touch_export import DEFAULT_MNIST_RAW, load_mnist


ROOT = Path(__file__).resolve().parent


class MNISTCNN(nn.Module):
    """Compact CNN for 28x28 MNIST digits."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    image_tensor = torch.from_numpy(images.reshape(-1, 1, 28, 28)).float()
    label_tensor = torch.from_numpy(labels).long()
    return DataLoader(
        TensorDataset(image_tensor, label_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_func(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_func = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def save_checkpoint(
    path: Path,
    model: MNISTCNN,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "MNISTCNN",
            "state_dict": model.state_dict(),
            "input_shape": [1, 28, 28],
            "num_classes": 10,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_MNIST_RAW)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--target-train-acc", type=float, default=0.99)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "cnn_mnist.pth")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "cnn_metrics.json")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    set_seed(args.seed)
    train_x, train_y, test_x, test_y = load_mnist(args.raw_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    train_eval_loader = make_loader(train_x, train_y, args.batch_size * 2, shuffle=False)
    test_loader = make_loader(test_x, test_y, args.batch_size * 2, shuffle=False)

    model = MNISTCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_test_acc = 0.0
    best_metrics: dict | None = None
    start_time = time.perf_counter()

    print(f"device={device}")
    print(f"train={len(train_x)} test={len(test_x)} raw_dir={args.raw_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss_step, train_acc_step = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        train_loss, train_acc = evaluate(model, train_eval_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss_step": train_loss_step,
            "train_acc_step": train_acc_step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)

        print(
            f"epoch {epoch:02d}: "
            f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, "
            f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}"
        )

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_metrics = row
            save_checkpoint(args.weights, model, row)

        if epoch >= args.min_epochs and train_acc >= args.target_train_acc:
            break

    elapsed_sec = time.perf_counter() - start_time
    final_train_loss, final_train_acc = evaluate(model, train_eval_loader, device)
    final_test_loss, final_test_acc = evaluate(model, test_loader, device)

    final_metrics = {
        "model": "MNISTCNN",
        "framework": f"PyTorch {torch.__version__}",
        "device": str(device),
        "raw_dir": str(args.raw_dir),
        "epochs_run": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "target_train_acc": args.target_train_acc,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
        "best_epoch": best_metrics["epoch"] if best_metrics else None,
        "elapsed_sec": elapsed_sec,
        "weights": str(args.weights),
        "history": history,
    }

    save_checkpoint(args.weights, model, final_metrics)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"saved weights: {args.weights}")
    print(f"saved report : {args.report}")
    print(f"final_train_acc={final_train_acc:.4f}, final_test_acc={final_test_acc:.4f}")


if __name__ == "__main__":
    main()

