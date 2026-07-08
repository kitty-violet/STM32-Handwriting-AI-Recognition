"""Export the current FNN C arrays to a compact TF-card binary model."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from fnn_runtime import DEFAULT_EXPORT_DIR, load_fnn_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output", type=Path, default=Path("G:/MODEL/FNN.BIN"))
    args = parser.parse_args()

    model = load_fnn_model(args.export_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = bytearray()
    payload += b"FNN1"
    payload += struct.pack("<III", model.input_size, model.hidden_size, model.output_size)
    payload += model.fc1_weight.astype("<i1", copy=False).reshape(-1).tobytes()
    payload += model.fc1_bias.astype("<f4", copy=False).reshape(-1).tobytes()
    payload += model.fc2_weight.astype("<i1", copy=False).reshape(-1).tobytes()
    payload += model.fc2_bias.astype("<f4", copy=False).reshape(-1).tobytes()
    payload += struct.pack("<ff", float(model.fc1_scale), float(model.fc2_scale))

    args.output.write_bytes(payload)
    print(f"wrote {args.output} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()

