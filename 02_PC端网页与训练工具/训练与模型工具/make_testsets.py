#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create labeled BMP test sets for PC and microSD card testing."""

from __future__ import annotations

import argparse
import gzip
import random
import shutil
import struct
from pathlib import Path

import numpy as np

from digit_image import normalize_to_model_image, save_bmp_28
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "data" / "MNIST" / "raw"
DEFAULT_OUT_DIR = ROOT / "testsets"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}


def read_idx_images(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fp:
        magic, count, rows, cols = struct.unpack(">IIII", fp.read(16))
        if magic != 2051 or rows != 28 or cols != 28:
            raise ValueError(f"Unsupported image file: {path}")
        data = np.frombuffer(fp.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fp:
        magic, count = struct.unpack(">II", fp.read(8))
        if magic != 2049:
            raise ValueError(f"Unsupported label file: {path}")
        data = np.frombuffer(fp.read(), dtype=np.uint8)
    if len(data) != count:
        raise ValueError(f"Label count mismatch: {path}")
    return data.astype(np.uint8)


def find_idx(raw_dir: Path, stem: str) -> Path:
    for candidate in (raw_dir / stem, raw_dir / f"{stem}.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {stem} or {stem}.gz in {raw_dir}")


def load_mnist_test(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    images = read_idx_images(find_idx(raw_dir, "t10k-images-idx3-ubyte"))
    labels = read_idx_labels(find_idx(raw_dir, "t10k-labels-idx1-ubyte"))
    return images, labels


def reset_label_file(path: Path, rows: list[tuple[str, int]]) -> None:
    lines = ["# filename label"]
    lines.extend(f"{name} {label}" for name, label in rows)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def create_mnist_set(raw_dir: Path, out_dir: Path, per_digit: int) -> int:
    images, labels = load_mnist_test(raw_dir)
    target = out_dir / "MNIST"
    target.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int]] = []
    serial = 0

    for digit in range(10):
        indices = np.flatnonzero(labels == digit)[:per_digit]
        if len(indices) < per_digit:
            raise ValueError(f"MNIST digit {digit} has only {len(indices)} samples")
        for index in indices:
            pixels = normalize_to_model_image(images[int(index)])
            name = f"M{serial:03d}_{digit}.BMP"
            save_bmp_28(target / name, pixels)
            rows.append((name, digit))
            serial += 1

    reset_label_file(target / "LABEL.TXT", rows)
    (target / "README.TXT").write_text(
        "Standard MNIST test set. File name and LABEL.TXT both contain labels.\n",
        encoding="ascii",
    )
    return len(rows)


def create_personal_template(out_dir: Path) -> None:
    target = out_dir / "PERSONAL"
    target.mkdir(parents=True, exist_ok=True)
    label_path = target / "LABEL.TXT"
    if not label_path.exists():
        label_path.write_text(
            "# filename label\n"
            "# Put at least 10 own handwritten images here.\n"
            "# Example: P001_3.BMP 3\n",
            encoding="ascii",
        )
    (target / "README.TXT").write_text(
        "Put your own handwritten digit images in this folder.\n"
        "Use short names such as P001_3.BMP, or record labels in LABEL.TXT.\n",
        encoding="ascii",
    )


def _source_images_by_label(source_dir: Path) -> dict[int, list[Path]]:
    by_label: dict[int, list[Path]] = {digit: [] for digit in range(10)}
    for digit in range(10):
        label_dir = source_dir / str(digit)
        if not label_dir.exists():
            continue
        by_label[digit] = sorted(
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return by_label


def create_personal_from_source(
    source_dir: Path,
    out_dir: Path,
    per_digit: int,
    seed: int,
) -> int:
    target = out_dir / "PERSONAL"
    target.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int]] = []
    serial = 0
    rng = random.Random(seed)
    by_label = _source_images_by_label(source_dir)

    for digit in range(10):
        candidates = by_label[digit]
        if len(candidates) < per_digit:
            raise ValueError(f"{source_dir / str(digit)} has only {len(candidates)} images")
        selected = rng.sample(candidates, per_digit)
        for image_path in selected:
            pixels = normalize_to_model_image(np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8))
            name = f"P{serial:03d}_{digit}.BMP"
            save_bmp_28(target / name, pixels)
            rows.append((name, digit))
            serial += 1

    reset_label_file(target / "LABEL.TXT", rows)
    (target / "README.TXT").write_text(
        "Placeholder personal test set generated from a local labeled dataset.\n"
        "Use this only for debugging the batch-test workflow; replace with own handwriting for the final report.\n",
        encoding="ascii",
    )
    return len(rows)


def copy_tree_files(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
            count += 1
    return count


def sync_to_sd(local_root: Path, sd_root: Path) -> None:
    test_root = sd_root / "TEST"
    test_root.mkdir(parents=True, exist_ok=True)
    copy_tree_files(local_root / "MNIST", test_root / "MNIST")
    copy_tree_files(local_root / "PERSONAL", test_root / "PERSONAL")
    (test_root / "README.TXT").write_text(
        "Digit recognition test sets.\n"
        "MNIST: standard MNIST BMP files.\n"
        "PERSONAL: put own handwritten BMP/JPG/PNG files and labels here.\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--per-digit", type=int, default=10)
    parser.add_argument("--personal-source", type=Path)
    parser.add_argument("--personal-per-digit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--sd-root", type=Path)
    args = parser.parse_args()

    if args.per_digit < 1:
        raise ValueError("--per-digit must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    count = create_mnist_set(args.raw_dir, args.out_dir, args.per_digit)
    if args.personal_source is not None:
        personal_count = create_personal_from_source(
            args.personal_source,
            args.out_dir,
            args.personal_per_digit,
            args.seed,
        )
    else:
        create_personal_template(args.out_dir)
        personal_count = 0

    print(f"MNIST BMP count: {count}")
    print(f"PERSONAL BMP count: {personal_count}")
    print(f"local test root: {args.out_dir}")
    if args.sd_root is not None:
        if not args.sd_root.exists():
            raise FileNotFoundError(f"SD root does not exist: {args.sd_root}")
        sync_to_sd(args.out_dir, args.sd_root)
        print(f"synced to SD: {args.sd_root / 'TEST'}")


if __name__ == "__main__":
    main()


