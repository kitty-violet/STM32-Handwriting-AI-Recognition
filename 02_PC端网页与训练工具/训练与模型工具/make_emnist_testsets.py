#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create labeled BMP test sets for EMNIST ByClass."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

from emnist_common import (
    EMNISTTensorDataset,
    NUM_EMNIST_CLASSES,
    ROOT,
    class_text,
    safe_class_text,
)


def reset_dir(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def create_testset(
    data_root: Path,
    out_dir: Path,
    per_class: int,
    seed: int,
    download: bool,
) -> dict[str, object]:
    dataset = EMNISTTensorDataset(data_root, train=False, download=download)
    reset_dir(out_dir)
    rng = random.Random(seed)
    rows: list[tuple[str, int, str]] = []

    by_class: list[list[int]] = [[] for _ in range(NUM_EMNIST_CLASSES)]
    for index, label in enumerate(dataset.targets.tolist()):
        by_class[int(label)].append(index)

    serial = 0
    for class_id in range(NUM_EMNIST_CLASSES):
        candidates = by_class[class_id]
        if len(candidates) < per_class:
            raise ValueError(f"class {class_id} has only {len(candidates)} test samples")
        selected = rng.sample(candidates, per_class)
        label_text = safe_class_text(class_id)
        for source_index in selected:
            image = dataset.data[source_index].numpy()
            name = f"E{serial:04d}_{class_id:02d}_{label_text}.BMP"
            Image.fromarray(image).save(out_dir / name, format="BMP")
            rows.append((name, class_id, class_text(class_id)))
            serial += 1

    label_lines = ["# filename class_id label"]
    label_lines.extend(f"{name} {class_id} {label}" for name, class_id, label in rows)
    (out_dir / "LABEL.TXT").write_text("\n".join(label_lines) + "\n", encoding="ascii")
    (out_dir / "README.TXT").write_text(
        "EMNIST ByClass 62-class test set.\n"
        "Labels are stored both in filenames and LABEL.TXT.\n",
        encoding="ascii",
    )

    summary = {
        "name": "EMNIST_BYCLASS",
        "path": str(out_dir),
        "classes": NUM_EMNIST_CLASSES,
        "per_class": per_class,
        "count": len(rows),
        "label_file": str(out_dir / "LABEL.TXT"),
    }
    (out_dir / "SUMMARY.JSON").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def create_random_testset(
    data_root: Path,
    out_dir: Path,
    count: int,
    seed: int,
    download: bool,
) -> dict[str, object]:
    dataset = EMNISTTensorDataset(data_root, train=False, download=download)
    reset_dir(out_dir)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    if count > len(indices):
        raise ValueError(f"--count exceeds test set size: {len(indices)}")
    selected = rng.sample(indices, count)
    rows: list[tuple[str, int, str]] = []

    for serial, source_index in enumerate(selected):
        class_id = int(dataset.targets[source_index].item())
        label_text = safe_class_text(class_id)
        image = dataset.data[source_index].numpy()
        name = f"R{serial:04d}_{class_id:02d}_{label_text}.BMP"
        Image.fromarray(image).save(out_dir / name, format="BMP")
        rows.append((name, class_id, class_text(class_id)))

    label_lines = ["# filename class_id label"]
    label_lines.extend(f"{name} {class_id} {label}" for name, class_id, label in rows)
    (out_dir / "LABEL.TXT").write_text("\n".join(label_lines) + "\n", encoding="ascii")
    (out_dir / "README.TXT").write_text(
        "Random EMNIST ByClass test set sampled from the official test split.\n"
        "Labels are stored both in filenames and LABEL.TXT.\n",
        encoding="ascii",
    )
    summary = {
        "name": "EMNIST_STANDARD_RANDOM",
        "path": str(out_dir),
        "classes": NUM_EMNIST_CLASSES,
        "count": len(rows),
        "label_file": str(out_dir / "LABEL.TXT"),
        "mode": "random",
    }
    (out_dir / "SUMMARY.JSON").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def sync_to_sd(local_dir: Path, sd_root: Path) -> Path:
    target = sd_root / "TEST" / "EMNIST_BYCLASS"
    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.is_file():
            child.unlink()
    for child in local_dir.iterdir():
        if child.is_file():
            shutil.copy2(child, target / child.name)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "testsets" / "EMNIST_BYCLASS")
    parser.add_argument("--mode", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--count", type=int, default=620)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sd-root", type=Path)
    args = parser.parse_args()

    if args.per_class < 1:
        raise ValueError("--per-class must be >= 1")
    if args.count < 1:
        raise ValueError("--count must be >= 1")

    if args.mode == "balanced":
        summary = create_testset(
            data_root=args.data_root,
            out_dir=args.out_dir,
            per_class=args.per_class,
            seed=args.seed,
            download=args.download,
        )
    else:
        summary = create_random_testset(
            data_root=args.data_root,
            out_dir=args.out_dir,
            count=args.count,
            seed=args.seed,
            download=args.download,
    )
    print(f"created: {summary['path']}")
    if args.mode == "balanced":
        print(f"count={summary['count']} classes={summary['classes']} per_class={summary['per_class']}")
    else:
        print(f"count={summary['count']} classes={summary['classes']} mode=random")

    if args.sd_root is not None:
        if not args.sd_root.exists():
            raise FileNotFoundError(f"SD root does not exist: {args.sd_root}")
        target = sync_to_sd(args.out_dir, args.sd_root)
        print(f"synced to SD: {target}")


if __name__ == "__main__":
    main()

