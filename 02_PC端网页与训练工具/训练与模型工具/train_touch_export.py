#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train an MNIST fully connected neural network and export FNN_Data.c/.h."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DEFAULT_MNIST_RAW = Path(__file__).resolve().parent / "data" / "MNIST" / "raw"


class FNNNet(nn.Module):
    """Two-layer fully connected network: 784 -> hidden -> 10."""

    def __init__(self, hidden_size: int = 24) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_idx_images(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fp:
        magic, count, rows, cols = struct.unpack(">IIII", fp.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError(f"Unsupported image file: {path}")
        data = np.frombuffer(fp.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fp:
        magic, count = struct.unpack(">II", fp.read(8))
        if magic != 2049:
            raise ValueError(f"Unsupported label file: {path}")
        data = np.frombuffer(fp.read(), dtype=np.uint8)
    if len(data) != count:
        raise ValueError(f"Label count mismatch: {path}")
    return data.astype(np.int64)


def find_idx(raw_dir: Path, stem: str) -> Path:
    for candidate in (raw_dir / stem, raw_dir / f"{stem}.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {stem} or {stem}.gz in {raw_dir}")


def load_mnist(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = read_idx_images(find_idx(raw_dir, "train-images-idx3-ubyte"))
    train_labels = read_idx_labels(find_idx(raw_dir, "train-labels-idx1-ubyte"))
    test_images = read_idx_images(find_idx(raw_dir, "t10k-images-idx3-ubyte"))
    test_labels = read_idx_labels(find_idx(raw_dir, "t10k-labels-idx1-ubyte"))
    return train_images, train_labels, test_images, test_labels


def touch_style_batch(images: torch.Tensor, train: bool) -> torch.Tensor:
    """Make MNIST tensors closer to the binary/thick strokes produced by the LCD app."""
    x = images.view(images.size(0), 1, 28, 28)
    if train:
        batch = x.size(0)
        angles = torch.empty(batch, device=x.device).uniform_(-12.0, 12.0) * 3.14159265 / 180.0
        scales = torch.empty(batch, device=x.device).uniform_(0.88, 1.18)
        tx = torch.empty(batch, device=x.device).uniform_(-2.5, 2.5) * (2.0 / 28.0)
        ty = torch.empty(batch, device=x.device).uniform_(-2.5, 2.5) * (2.0 / 28.0)
        cos_v = torch.cos(angles) / scales
        sin_v = torch.sin(angles) / scales
        theta = torch.zeros(batch, 2, 3, device=x.device)
        theta[:, 0, 0] = cos_v
        theta[:, 0, 1] = -sin_v
        theta[:, 1, 0] = sin_v
        theta[:, 1, 1] = cos_v
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        threshold = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.16, 0.30)
    else:
        threshold = torch.full((x.size(0), 1, 1, 1), 0.20, device=x.device)

    binary = (x > threshold).float()
    dilated = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)

    if train:
        soft_dilated = F.avg_pool2d(dilated, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)
        mode = torch.rand(x.size(0), 1, 1, 1, device=x.device)
        x = torch.where(mode < 0.15, binary, torch.where(mode < 0.85, dilated, soft_dilated))
    else:
        x = dilated

    return x.view(images.size(0), -1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    touch_style: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if touch_style:
                images = touch_style_batch(images, train=False)
            logits = model(images)
            loss = loss_func(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count


def quantize_weight(values: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(values)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    quantized = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return quantized, scale


def format_float_array(name: str, values: np.ndarray, per_line: int = 8) -> str:
    flat = values.astype(np.float32).reshape(-1)
    lines = [f"const float {name}[{flat.size}] = {{"]
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line]
        lines.append("    " + ", ".join(f"{float(v): .8e}f" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def format_int8_array(name: str, values: np.ndarray, per_line: int = 16) -> str:
    flat = values.astype(np.int8).reshape(-1)
    lines = [f"const int8_t {name}[{flat.size}] = {{"]
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line]
        lines.append("    " + ", ".join(f"{int(v):4d}" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def format_uint8_array(name: str, values: np.ndarray, per_line: int = 16) -> str:
    flat = values.reshape(-1)
    lines = [f"const uint8_t {name}[{flat.size}] = {{"]
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line]
        lines.append("    " + ", ".join(f"{int(v):3d}" for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def export_c_files(
    model: FNNNet,
    hidden_size: int,
    sample_image: np.ndarray,
    sample_label: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    fc1_weight = state["fc1.weight"].cpu().numpy()
    fc1_bias = state["fc1.bias"].cpu().numpy()
    fc2_weight = state["fc2.weight"].cpu().numpy()
    fc2_bias = state["fc2.bias"].cpu().numpy()
    fc1_weight_q, fc1_scale = quantize_weight(fc1_weight)
    fc2_weight_q, fc2_scale = quantize_weight(fc2_weight)

    header = f"""#ifndef FNN_DATA_H
#define FNN_DATA_H

#include <stdint.h>

#define FNN_INPUT_SIZE 784
#define FNN_HIDDEN_SIZE {hidden_size}
#define FNN_OUTPUT_SIZE 10

extern const int8_t g_fnn_fc1_weight[FNN_HIDDEN_SIZE * FNN_INPUT_SIZE];
extern const float g_fnn_fc1_bias[FNN_HIDDEN_SIZE];
extern const int8_t g_fnn_fc2_weight[FNN_OUTPUT_SIZE * FNN_HIDDEN_SIZE];
extern const float g_fnn_fc2_bias[FNN_OUTPUT_SIZE];
extern const float g_fnn_fc1_scale;
extern const float g_fnn_fc2_scale;
extern const uint8_t g_digit_test_image[FNN_INPUT_SIZE];
extern const uint8_t g_digit_test_label;

#endif
"""
    source = "\n\n".join(
        [
            '#include "FNN_Data.h"',
            format_int8_array("g_fnn_fc1_weight", fc1_weight_q),
            format_float_array("g_fnn_fc1_bias", fc1_bias),
            format_int8_array("g_fnn_fc2_weight", fc2_weight_q),
            format_float_array("g_fnn_fc2_bias", fc2_bias),
            f"const float g_fnn_fc1_scale = {fc1_scale:.8e}f;",
            f"const float g_fnn_fc2_scale = {fc2_scale:.8e}f;",
            format_uint8_array("g_digit_test_image", (sample_image * 255.0).round().astype(np.uint8)),
            f"const uint8_t g_digit_test_label = {int(sample_label)};",
        ]
    )
    (out_dir / "FNN_Data.h").write_text(header, encoding="utf-8")
    (out_dir / "FNN_Data.c").write_text(source + "\n", encoding="utf-8")


def choose_export_sample(
    model: FNNNet,
    test_x: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    touch_style: bool,
) -> tuple[np.ndarray, int]:
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_x), 1024):
            images = torch.from_numpy(test_x[start : start + 1024]).to(device)
            labels = torch.from_numpy(test_y[start : start + 1024]).to(device)
            if touch_style:
                images = touch_style_batch(images, train=False)
            preds = model(images).argmax(dim=1)
            correct = (preds == labels).nonzero(as_tuple=False)
            if correct.numel() > 0:
                local_index = int(correct[0].item())
                return images[local_index].cpu().numpy().reshape(-1), int(labels[local_index].item())

    sample_image = test_x[0]
    if touch_style:
        sample_tensor = torch.from_numpy(sample_image.reshape(1, -1)).to(device)
        sample_image = touch_style_batch(sample_tensor, train=False).cpu().numpy().reshape(-1)
    return sample_image, int(test_y[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_MNIST_RAW)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--touch-augment", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    train_x, train_y, test_x, test_y = load_mnist(args.raw_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNNNet(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    class_weights = None
    if args.touch_augment and not args.no_class_weights:
        class_weights = torch.tensor(
            [1.0, 1.05, 1.0, 1.15, 1.5, 1.8, 1.3, 1.0, 1.3, 1.4],
            device=device,
        )
    loss_func = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
        batch_size=1024,
        shuffle=False,
    )

    history = []
    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            if args.touch_augment:
                images = touch_style_batch(images, train=True)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

        clean_loss, clean_acc = evaluate(model, test_loader, device, touch_style=False)
        touch_loss, touch_acc = evaluate(model, test_loader, device, touch_style=True)
        row = {
            "epoch": epoch,
            "model": f"fnn_784_{args.hidden_size}_10",
            "touch_augment": args.touch_augment,
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
            "train_loss": total_loss / total_count,
            "train_acc": total_correct / total_count,
            "clean_test_loss": clean_loss,
            "clean_test_acc": clean_acc,
            "touch_test_loss": touch_loss,
            "touch_test_acc": touch_acc,
        }
        history.append(row)
        target_acc = touch_acc if args.touch_augment else clean_acc
        if target_acc >= best_acc:
            best_acc = target_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:02d}: train_acc={row['train_acc']:.4f}, "
            f"clean_acc={clean_acc:.4f}, touch_acc={touch_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    ROOT.joinpath("models").mkdir(exist_ok=True)
    ROOT.joinpath("reports").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / "fnn_touch_mnist.pth")
    sample_image, sample_label = choose_export_sample(model, test_x, test_y, device, args.touch_augment)
    export_c_files(model.to("cpu"), args.hidden_size, sample_image, sample_label, ROOT / "export")
    metrics = json.dumps(history, indent=2)
    (ROOT / "reports" / "fnn_metrics.json").write_text(metrics, encoding="utf-8")
    (ROOT / "reports" / "metrics.json").write_text(metrics, encoding="utf-8")
    print(f"model=fnn_784_{args.hidden_size}_10")
    print(f"best_selected_acc={best_acc:.4f}")
    print(f"exported: {ROOT / 'export'}")


if __name__ == "__main__":
    main()


