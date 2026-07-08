#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check exported FNN_Data.c arrays with the same math used by STM32."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "export" / "FNN_Data.c"
HEADER = ROOT / "export" / "FNN_Data.h"


def read_float_array(text: str, name: str) -> np.ndarray:
    pattern = rf"const float {name}\[\d+\] = \{{(.*?)\}};"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find {name}")
    values = [float(x.replace("f", "")) for x in re.findall(r"[-+ ]?\d+\.\d+e[-+]\d+f", match.group(1))]
    return np.array(values, dtype=np.float32)


def read_int8_array(text: str, name: str) -> np.ndarray:
    pattern = rf"const int8_t {name}\[\d+\] = \{{(.*?)\}};"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find {name}")
    return np.array([int(x) for x in re.findall(r"-?\d+", match.group(1))], dtype=np.int8)


def read_float_scalar(text: str, name: str) -> float:
    match = re.search(rf"const float {name} = ([-+0-9.e]+)f;", text)
    if not match:
        raise ValueError(f"Cannot find {name}")
    return float(match.group(1))


def read_uint8_array(text: str, name: str) -> np.ndarray:
    pattern = rf"const uint8_t {name}\[\d+\] = \{{(.*?)\}};"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find {name}")
    return np.array([int(x) for x in re.findall(r"\d+", match.group(1))], dtype=np.uint8)


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    input_size = int(re.search(r"#define FNN_INPUT_SIZE (\d+)", header).group(1))
    hidden_size = int(re.search(r"#define FNN_HIDDEN_SIZE (\d+)", header).group(1))
    output_size = int(re.search(r"#define FNN_OUTPUT_SIZE (\d+)", header).group(1))
    fc1_w = read_int8_array(text, "g_fnn_fc1_weight").reshape(hidden_size, input_size)
    fc1_b = read_float_array(text, "g_fnn_fc1_bias")
    fc2_w = read_int8_array(text, "g_fnn_fc2_weight").reshape(output_size, hidden_size)
    fc2_b = read_float_array(text, "g_fnn_fc2_bias")
    fc1_scale = read_float_scalar(text, "g_fnn_fc1_scale")
    fc2_scale = read_float_scalar(text, "g_fnn_fc2_scale")
    image = read_uint8_array(text, "g_digit_test_image").astype(np.float32)
    label = int(re.search(r"g_digit_test_label = (\d+);", text).group(1))

    hidden = np.maximum((fc1_w.astype(np.float32) @ image) * (fc1_scale / 255.0) + fc1_b, 0.0)
    logits = (fc2_w.astype(np.float32) @ hidden) * fc2_scale + fc2_b
    pred = int(np.argmax(logits))
    print(f"label={label}, pred={pred}, logits={np.round(logits, 4).tolist()}")
    raise SystemExit(0 if pred == label else 1)


if __name__ == "__main__":
    main()

