#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch-evaluate labeled digit image test sets with exported FNN data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from digit_image import load_digit_image
from fnn_runtime import load_fnn_model


ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / "testsets"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def parse_label_file(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
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
            labels[parts[0].upper()] = int(parts[1])
        except ValueError:
            continue
    return labels


def label_from_name(path: Path) -> int | None:
    match = re.search(r"[_-]([0-9])$", path.stem)
    if match:
        return int(match.group(1))
    if path.stem and path.stem[-1].isdigit():
        return int(path.stem[-1])
    return None


def iter_labeled_images(dataset_dir: Path) -> list[tuple[Path, int]]:
    file_labels = parse_label_file(dataset_dir / "LABEL.TXT")
    result: list[tuple[Path, int]] = []
    if not dataset_dir.exists():
        return result

    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = file_labels.get(path.name.upper())
        if label is None:
            label = label_from_name(path)
        if label is not None and 0 <= label <= 9:
            result.append((path, label))
    return result


def evaluate_dataset(name: str, dataset_dir: Path, repeat: int, normalize_large: bool) -> dict[str, object]:
    model = load_fnn_model()
    samples = iter_labeled_images(dataset_dir)
    rows: list[dict[str, object]] = []
    correct = 0
    infer_seconds = 0.0
    total_seconds = 0.0

    for path, label in samples:
        total_start = time.perf_counter()
        image = load_digit_image(path, normalize_large=normalize_large)
        pred = -1
        logits = np.zeros((model.output_size,), dtype=np.float32)
        elapsed = 0.0
        for _ in range(repeat):
            infer_start = time.perf_counter()
            pred, logits = model.predict(image)
            elapsed += time.perf_counter() - infer_start
        elapsed /= repeat
        total_elapsed = time.perf_counter() - total_start
        infer_seconds += elapsed
        total_seconds += total_elapsed
        ok = pred == label
        correct += int(ok)
        rows.append(
            {
                "file": path.name,
                "label": label,
                "pred": pred,
                "ok": ok,
                "infer_ms": elapsed * 1000.0,
                "total_ms": total_elapsed * 1000.0,
                "logits": [round(float(x), 5) for x in logits.tolist()],
            }
        )

    count = len(samples)
    return {
        "name": name,
        "path": str(dataset_dir),
        "count": count,
        "correct": correct,
        "accuracy": (correct / count) if count else None,
        "avg_infer_ms": (infer_seconds * 1000.0 / count) if count else None,
        "avg_total_ms": (total_seconds * 1000.0 / count) if count else None,
        "items": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-normalize-large", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "test_eval.json")
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    results = [
        evaluate_dataset("MNIST", args.root / "MNIST", args.repeat, not args.no_normalize_large),
        evaluate_dataset("PERSONAL", args.root / "PERSONAL", args.repeat, not args.no_normalize_large),
    ]
    output = {
        "root": str(args.root),
        "repeat": args.repeat,
        "datasets": results,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    for item in results:
        acc = item["accuracy"]
        acc_text = "N/A" if acc is None else f"{acc * 100.0:.2f}%"
        infer = item["avg_infer_ms"]
        infer_text = "N/A" if infer is None else f"{infer:.4f} ms"
        print(
            f"{item['name']}: count={item['count']}, correct={item['correct']}, "
            f"accuracy={acc_text}, avg_infer={infer_text}"
        )
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()

