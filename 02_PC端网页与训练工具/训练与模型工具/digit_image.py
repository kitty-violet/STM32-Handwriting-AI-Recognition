#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image conversion helpers for 28x28 digit inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


GRID_SIZE = 28
NORMAL_SIZE = 20


def normalize_to_model_image(gray: np.ndarray, normal_size: int = NORMAL_SIZE) -> np.ndarray:
    """Convert a grayscale digit image to the same 28x28 binary layout used by STM32."""
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    # Most photos/scans are dark strokes on light paper; MNIST is already bright on dark.
    work = arr
    if float(work.mean()) > 127.0:
        work = 255 - work

    if int(work.max()) > int(work.min()):
        work = np.asarray(ImageOps.autocontrast(Image.fromarray(work)), dtype=np.uint8)

    threshold = max(24, int(float(work.mean()) + 0.20 * float(work.std())))
    mask = work > threshold
    if not bool(mask.any()):
        return np.zeros((GRID_SIZE * GRID_SIZE,), dtype=np.uint8)

    ys, xs = np.nonzero(mask)
    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255
    height, width = cropped.shape
    long_side = max(width, height)
    scaled_w = max(1, min(normal_size, int((width * normal_size + long_side // 2) // long_side)))
    scaled_h = max(1, min(normal_size, int((height * normal_size + long_side // 2) // long_side)))

    resized = Image.fromarray(cropped).resize((scaled_w, scaled_h), Image.Resampling.NEAREST)
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    offset_x = (GRID_SIZE - scaled_w) // 2
    offset_y = (GRID_SIZE - scaled_h) // 2
    out[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = np.asarray(resized, dtype=np.uint8)
    return center_model_image(out).reshape(-1)


def center_model_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)
    ys, xs = np.nonzero(arr > 0)
    if xs.size == 0:
        return arr.copy()

    center_x = int((int(xs.sum()) + xs.size // 2) // xs.size)
    center_y = int((int(ys.sum()) + ys.size // 2) // ys.size)
    shift_x = GRID_SIZE // 2 - center_x
    shift_y = GRID_SIZE // 2 - center_y
    if shift_x == 0 and shift_y == 0:
        return arr.copy()

    out = np.zeros_like(arr)
    for y, x in zip(ys, xs):
        nx = int(x) + shift_x
        ny = int(y) + shift_y
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            out[ny, nx] = arr[y, x]
    return out


def save_bmp_28(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(pixels, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)
    Image.fromarray(arr).save(path, format="BMP")


def load_digit_image(path: Path, normalize_large: bool = True) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape == (GRID_SIZE, GRID_SIZE) and not normalize_large:
        return arr.reshape(-1)
    if arr.shape == (GRID_SIZE, GRID_SIZE):
        unique_count = np.unique(arr).size
        if unique_count <= 4:
            return arr.reshape(-1)
    if normalize_large:
        return normalize_to_model_image(arr)
    resized = img.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8).reshape(-1)

