#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime helpers for the exported STM32 FNN model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORT_DIR = ROOT / "export"
FLOAT_PATTERN = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?f?"


@dataclass(frozen=True)
class FNNModel:
    input_size: int
    hidden_size: int
    output_size: int
    fc1_weight: np.ndarray
    fc1_bias: np.ndarray
    fc2_weight: np.ndarray
    fc2_bias: np.ndarray
    fc1_scale: float
    fc2_scale: float

    def logits(self, image: np.ndarray) -> np.ndarray:
        x = np.asarray(image, dtype=np.float32).reshape(-1)
        if x.size != self.input_size:
            raise ValueError(f"expected {self.input_size} pixels, got {x.size}")

        hidden = (self.fc1_weight.astype(np.float32) @ x) * (self.fc1_scale / 255.0) + self.fc1_bias
        hidden = np.maximum(hidden, 0.0)
        return (self.fc2_weight.astype(np.float32) @ hidden) * self.fc2_scale + self.fc2_bias

    def predict(self, image: np.ndarray) -> tuple[int, np.ndarray]:
        scores = self.logits(image)
        return int(np.argmax(scores)), scores


def _read_define(text: str, name: str) -> int:
    match = re.search(rf"#define\s+{name}\s+(\d+)", text)
    if not match:
        raise ValueError(f"Cannot find {name}")
    return int(match.group(1))


def _read_array_body(text: str, c_type: str, name: str) -> str:
    pattern = rf"const\s+{re.escape(c_type)}\s+{name}\[\d+\]\s*=\s*\{{(.*?)\}};"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find {name}")
    return match.group(1)


def _read_float_array(text: str, name: str) -> np.ndarray:
    body = _read_array_body(text, "float", name)
    return np.array([float(x.rstrip("f")) for x in re.findall(FLOAT_PATTERN, body, flags=re.I)], dtype=np.float32)


def _read_int8_array(text: str, name: str) -> np.ndarray:
    body = _read_array_body(text, "int8_t", name)
    return np.array([int(x) for x in re.findall(r"-?\d+", body)], dtype=np.int8)


def _read_float_scalar(text: str, name: str) -> float:
    match = re.search(rf"const\s+float\s+{name}\s*=\s*({FLOAT_PATTERN});", text, flags=re.I)
    if not match:
        raise ValueError(f"Cannot find {name}")
    return float(match.group(1).rstrip("f"))


def load_fnn_model(export_dir: Path = DEFAULT_EXPORT_DIR) -> FNNModel:
    export_dir = Path(export_dir)
    source = (export_dir / "FNN_Data.c").read_text(encoding="utf-8")
    header = (export_dir / "FNN_Data.h").read_text(encoding="utf-8")

    input_size = _read_define(header, "FNN_INPUT_SIZE")
    hidden_size = _read_define(header, "FNN_HIDDEN_SIZE")
    output_size = _read_define(header, "FNN_OUTPUT_SIZE")

    fc1_weight = _read_int8_array(source, "g_fnn_fc1_weight").reshape(hidden_size, input_size)
    fc1_bias = _read_float_array(source, "g_fnn_fc1_bias")
    fc2_weight = _read_int8_array(source, "g_fnn_fc2_weight").reshape(output_size, hidden_size)
    fc2_bias = _read_float_array(source, "g_fnn_fc2_bias")

    return FNNModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
        fc1_scale=_read_float_scalar(source, "g_fnn_fc1_scale"),
        fc2_scale=_read_float_scalar(source, "g_fnn_fc2_scale"),
    )

