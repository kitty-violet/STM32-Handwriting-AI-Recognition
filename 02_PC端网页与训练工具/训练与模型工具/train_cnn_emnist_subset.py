#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a smaller EMNIST CNN for PC-side word recognition.

The full EMNIST ByClass task has unavoidable isolated-glyph ambiguities such as
0/O/o and 1/I/l. Mode 5 is an English-word workflow, so this script trains a
lowercase-only model that matches that workflow instead of forcing all 62
classes into one isolated classifier.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from emnist_common import (
    EMNIST_CLASSES,
    EMNISTCNN,
    EMNISTTensorDataset,
    ROOT,
    augment_batch,
    evaluate_model,
    save_checkpoint,
    set_seed,
)


CLASS_PRESETS = {
    "lower": list(range(36, 62)),
    "digit_lower": list(range(10)) + list(range(36, 62)),
    "digits": list(range(10)),
}


class EMNISTSubsetDataset(Dataset):
    def __init__(self, data_root: Path, train: bool, download: bool, class_ids: list[int]) -> None:
        base = EMNISTTensorDataset(data_root, train=train, download=download)
        ids = torch.tensor(class_ids, dtype=torch.long)
        mask = torch.isin(base.targets, ids)
        lookup = torch.full((len(EMNIST_CLASSES),), -1, dtype=torch.long)
        lookup[ids] = torch.arange(len(ids), dtype=torch.long)
        self.data = base.data[mask]
        self.targets = lookup[base.targets[mask]]
        self.classes = [EMNIST_CLASSES[i] for i in class_ids]

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.data[index].unsqueeze(0).float().div(255.0)
        return image, self.targets[index]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def make_class_weights(dataset: EMNISTSubsetDataset, power: float) -> torch.Tensor | None:
    if power <= 0:
        return None
    counts = torch.bincount(dataset.targets, minlength=len(dataset.classes)).float()
    weights = counts.sum() / (len(dataset.classes) * counts.clamp_min(1.0))
    weights = weights.pow(power)
    return weights / weights.mean().clamp_min(1.0e-6)



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    label_smoothing: float,
    class_weights: torch.Tensor | None,
    use_augment: bool,
) -> tuple[float, float]:
    model.train()
    loss_func = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        if use_augment:
            images = augment_batch(images)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = loss_func(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total_count += int(labels.size(0))

    return total_loss / total_count, total_correct / total_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--classes", choices=sorted(CLASS_PRESETS), default="lower")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=8.0e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--class-weight-power", type=float, default=0.20)
    parser.add_argument("--sampler-power", type=float, default=0.0)
    parser.add_argument("--boost-classes", type=str, default="")
    parser.add_argument("--save-score", choices=["macro", "min_class", "hybrid"], default="macro")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "cnn_emnist_lowercase.pth")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "emnist_lowercase_metrics.json")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    class_ids = CLASS_PRESETS[args.classes]
    train_set = EMNISTSubsetDataset(args.data_root, train=True, download=args.download, class_ids=class_ids)
    test_set = EMNISTSubsetDataset(args.data_root, train=False, download=args.download, class_ids=class_ids)
    train_loader = make_loader(train_set, args.batch_size, True, args.workers, device)
    test_loader = make_loader(test_set, args.batch_size * 2, False, args.workers, device)

    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"preset={args.classes} train={len(train_set)} test={len(test_set)} classes={len(train_set.classes)}", flush=True)
    print(
        f"save_score={args.save_score} sampler_power={args.sampler_power} "
        f"boosts={dict(zip(train_set.classes, [round(float(x), 3) for x in boosts.tolist()]))}",
        flush=True,
    )

    model = EMNISTCNN(num_classes=len(train_set.classes)).to(device)
    model.emnist_classes = train_set.classes
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    class_weights = make_class_weights(train_set, args.class_weight_power)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.20,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    history = []
    best_score = -1.0
    best_macro = 0.0
    best_min_class = 0.0
    best_epoch = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            amp_enabled,
            args.label_smoothing,
            class_weights,
            not args.no_augment,
        )
        test_result = evaluate_model(
            model,
            test_loader,
            device,
            num_classes=len(train_set.classes),
            class_names=train_set.classes,
        )
        macro = sum(float(x["accuracy"] or 0.0) for x in test_result.per_class) / len(test_result.per_class)
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_result.accuracy,
            "test_macro_acc": macro,
            "test_loss": test_result.loss,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: train_acc={train_acc:.4f}, "
            f"test_acc={test_result.accuracy:.4f}, macro={macro:.4f}, "
            f"min_class={min_class:.4f}, score={score:.4f}, "
            f"loss={test_result.loss:.4f}, time={elapsed:.1f}s",
            flush=True,
        )

        if score >= best_score:
            best_score = score
            best_macro = macro
            best_min_class = min_class
            best_epoch = epoch
            save_checkpoint(
                args.weights,
                model,
                {
                    "epoch": epoch,
                    "test_acc": test_result.accuracy,
                    "test_macro_acc": macro,
                    "test_loss": test_result.loss,
                    "class_preset": args.classes,
                },
            )

    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    best_model = EMNISTCNN(num_classes=int(checkpoint["num_classes"])).to(device)
    best_model.load_state_dict(checkpoint["state_dict"])
    best_model.emnist_classes = train_set.classes
    best_model.eval()
    final_test = evaluate_model(
        best_model,
        test_loader,
        device,
        num_classes=len(train_set.classes),
        class_names=train_set.classes,
    )
    final_macro = sum(float(x["accuracy"] or 0.0) for x in final_test.per_class) / len(final_test.per_class)
    metrics = {
        "model": "EMNISTCNN",
        "task": f"EMNIST ByClass subset {args.classes}",
        "classes": train_set.classes,
        "num_classes": len(train_set.classes),
        "train_count": len(train_set),
        "test_count": len(test_set),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "weights": str(args.weights),
        "best_epoch": best_epoch,
        "best_macro_acc": best_macro,
        "final_macro_acc": final_macro,
        "final_test": asdict(final_test),
        "history": history,
        "elapsed_sec": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_checkpoint(args.weights, best_model, metrics)
    print(f"saved weights: {args.weights}", flush=True)
    print(f"saved report : {args.report}", flush=True)
    print(f"best_epoch={best_epoch}, final_acc={final_test.accuracy:.4f}, final_macro={final_macro:.4f}", flush=True)


if __name__ == "__main__":
    main()



