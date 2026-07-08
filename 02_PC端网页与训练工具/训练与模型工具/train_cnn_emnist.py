#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a 62-class CNN on EMNIST ByClass and save a detailed report."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from emnist_common import (
    EMNISTCNN,
    EMNISTTensorDataset,
    NUM_EMNIST_CLASSES,
    ROOT,
    augment_batch,
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def make_loader(
    dataset: EMNISTTensorDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "sampler": sampler,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def make_balanced_sampler(dataset: EMNISTTensorDataset) -> WeightedRandomSampler:
    targets = dataset.targets.long()
    class_counts = torch.bincount(targets, minlength=NUM_EMNIST_CLASSES).float()
    sample_weights = (1.0 / class_counts.clamp_min(1.0))[targets]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=int(targets.numel()),
        replacement=True,
    )


def make_class_weights(dataset: EMNISTTensorDataset, power: float) -> torch.Tensor:
    targets = dataset.targets.long()
    class_counts = torch.bincount(targets, minlength=NUM_EMNIST_CLASSES).float()
    weights = class_counts.sum() / (NUM_EMNIST_CLASSES * class_counts.clamp_min(1.0))
    weights = weights.pow(power)
    weights = weights / weights.mean().clamp_min(1.0e-6)
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    use_augment: bool,
    label_smoothing: float,
    class_weights: torch.Tensor | None,
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

        total_loss += loss.item() * labels.size(0)
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total_count += int(labels.size(0))

    return total_loss / total_count, total_correct / total_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--target-test-acc", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "cnn_emnist_byclass.pth")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "emnist_cnn_metrics.json")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (not args.no_amp) and device.type == "cuda"

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    train_set = EMNISTTensorDataset(args.data_root, train=True, download=args.download, limit=args.limit_train)
    test_set = EMNISTTensorDataset(args.data_root, train=False, download=args.download, limit=args.limit_test)
    sampler = make_balanced_sampler(train_set) if args.balanced_sampler else None
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, workers=args.workers, device=device, sampler=sampler)
    test_loader = make_loader(test_set, args.batch_size * 2, shuffle=False, workers=args.workers, device=device)

    print(f"train={len(train_set)} test={len(test_set)} classes={NUM_EMNIST_CLASSES}")

    if args.eval_only:
        eval_weights = args.resume if args.resume is not None else args.weights
        model = load_checkpoint(eval_weights, device)
        final_test = evaluate_model(model, test_loader, device)
        final_metrics = {
            "model": "EMNISTCNN",
            "task": "EMNIST ByClass 62-class character recognition",
            "framework": f"PyTorch {torch.__version__}",
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "data_root": str(args.data_root),
            "classes": NUM_EMNIST_CLASSES,
            "train_count": len(train_set),
            "test_count": len(test_set),
            "weights": str(eval_weights),
            "final_test": asdict(final_test),
            "history": [],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"eval_accuracy={final_test.accuracy:.4f}")
        print(f"saved report : {args.report}")
        return

    model = EMNISTCNN().to(device)
    resume_best_test_acc = 0.0
    resume_best_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        metrics = checkpoint.get("metrics", {})
        if isinstance(metrics, dict):
            resume_best_test_acc = float(
                metrics.get("best_test_acc")
                or metrics.get("test_acc")
                or metrics.get("final_test", {}).get("accuracy", 0.0)
                or 0.0
            )
            resume_best_epoch = int(metrics.get("best_epoch") or metrics.get("epoch") or 0)
        print(f"resumed: {args.resume}")
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    class_weights = None
    if args.class_weight_power > 0.0:
        class_weights = make_class_weights(train_set, args.class_weight_power).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    history = []
    best_test_acc = resume_best_test_acc
    best_epoch = resume_best_epoch
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            use_augment=not args.no_augment,
            label_smoothing=args.label_smoothing,
            class_weights=class_weights,
        )
        test_result = evaluate_model(model, test_loader, device)
        elapsed = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss_step": train_loss,
            "train_acc_step": train_acc,
            "test_loss": test_result.loss,
            "test_acc": test_result.accuracy,
            "test_correct": test_result.correct,
            "test_total": test_result.total,
            "avg_infer_ms": test_result.avg_infer_ms,
            "elapsed_sec": elapsed,
        }
        history.append(row)

        if test_result.accuracy >= best_test_acc:
            best_test_acc = test_result.accuracy
            best_epoch = epoch
            save_checkpoint(
                args.weights,
                model,
                {
                    "epoch": epoch,
                    "test_acc": test_result.accuracy,
                    "test_loss": test_result.loss,
                    "train_acc_step": train_acc,
                },
            )

        print(
            f"epoch {epoch:02d}: "
            f"train_acc_step={train_acc:.4f}, test_acc={test_result.accuracy:.4f}, "
            f"test_loss={test_result.loss:.4f}, time={elapsed:.1f}s"
        )

        if args.target_test_acc > 0.0 and test_result.accuracy >= args.target_test_acc:
            break

    best_model = load_checkpoint(args.weights, device)
    final_test = evaluate_model(best_model, test_loader, device)
    elapsed_sec = time.perf_counter() - start_time

    final_metrics = {
        "model": "EMNISTCNN",
        "task": "EMNIST ByClass 62-class character recognition",
        "framework": f"PyTorch {torch.__version__}",
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "data_root": str(args.data_root),
        "classes": NUM_EMNIST_CLASSES,
        "train_count": len(train_set),
        "test_count": len(test_set),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "final_test": asdict(final_test),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "balanced_sampler": args.balanced_sampler,
        "class_weight_power": args.class_weight_power,
        "augment": not args.no_augment,
        "amp": amp_enabled,
        "elapsed_sec": elapsed_sec,
        "weights": str(args.weights),
        "history": history,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(final_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_checkpoint(args.weights, best_model, final_metrics)

    print(f"saved weights: {args.weights}")
    print(f"saved report : {args.report}")
    print(f"best_epoch={best_epoch}, best_test_acc={best_test_acc:.4f}")


if __name__ == "__main__":
    main()

