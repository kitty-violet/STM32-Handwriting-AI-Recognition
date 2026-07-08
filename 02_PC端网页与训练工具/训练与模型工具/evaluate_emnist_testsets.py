#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate a labeled EMNIST BMP test set with the trained CNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import torch

from emnist_common import ROOT, class_text, image_to_tensor, load_checkpoint


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def parse_label_file(path: Path) -> dict[str, tuple[int, str]]:
    labels: dict[str, tuple[int, str]] = {}
    if not path.exists():
        return labels
    for raw_line in path.read_text(encoding="ascii", errors="ignore").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 2:
            continue
        try:
            class_id = int(parts[1])
        except ValueError:
            continue
        label = parts[2] if len(parts) >= 3 else class_text(class_id)
        labels[parts[0].upper()] = (class_id, label)
    return labels


def label_from_name(path: Path) -> tuple[int, str] | None:
    match = re.search(r"_([0-9]{2})_([A-Za-z0-9]+)$", path.stem)
    if not match:
        return None
    class_id = int(match.group(1))
    return class_id, match.group(2)


def iter_labeled_images(dataset_dir: Path) -> list[tuple[Path, int, str]]:
    file_labels = parse_label_file(dataset_dir / "LABEL.TXT")
    rows: list[tuple[Path, int, str]] = []
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = file_labels.get(path.name.upper())
        if parsed is None:
            parsed = label_from_name(path)
        if parsed is not None:
            rows.append((path, parsed[0], parsed[1]))
    return rows


def evaluate_dataset(
    model: torch.nn.Module,
    dataset_dir: Path,
    device: torch.device,
    repeat: int,
) -> dict[str, object]:
    samples = iter_labeled_images(dataset_dir)
    rows = []
    correct = 0
    infer_seconds = 0.0

    with torch.no_grad():
        for path, class_id, label in samples:
            image = image_to_tensor(path).unsqueeze(0).to(device)
            pred = -1
            scores = None
            elapsed = 0.0
            for _ in range(repeat):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                logits = model(image)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                scores = logits.detach().cpu().squeeze(0)
                pred = int(scores.argmax().item())
            elapsed /= repeat
            ok = pred == class_id
            correct += int(ok)
            infer_seconds += elapsed
            top = torch.topk(scores, k=min(5, scores.numel()))
            rows.append(
                {
                    "file": path.name,
                    "label_id": class_id,
                    "label": label,
                    "pred_id": pred,
                    "pred": class_text(pred),
                    "ok": ok,
                    "infer_ms": elapsed * 1000.0,
                    "top5": [
                        {
                            "class_id": int(idx),
                            "label": class_text(int(idx)),
                            "score": round(float(score), 5),
                        }
                        for score, idx in zip(top.values.tolist(), top.indices.tolist())
                    ],
                }
            )

    count = len(samples)
    return {
        "path": str(dataset_dir),
        "count": count,
        "correct": correct,
        "accuracy": (correct / count) if count else None,
        "avg_infer_ms": (infer_seconds * 1000.0 / count) if count else None,
        "items": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "cnn_emnist_byclass.pth")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "testsets" / "EMNIST_BYCLASS")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "emnist_testset_eval.json")
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.weights, device)
    result = evaluate_dataset(model, args.dataset_dir, device, args.repeat)
    output = {
        "weights": str(args.weights),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "repeat": args.repeat,
        "dataset": result,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    acc = result["accuracy"]
    acc_text = "N/A" if acc is None else f"{acc * 100.0:.2f}%"
    infer = result["avg_infer_ms"]
    infer_text = "N/A" if infer is None else f"{infer:.4f} ms"
    print(
        f"count={result['count']}, correct={result['correct']}, "
        f"accuracy={acc_text}, avg_infer={infer_text}"
    )
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()

