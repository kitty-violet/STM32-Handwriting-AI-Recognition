# EMNIST CNN Accuracy Report

## Task
EMNIST ByClass 62-class character recognition: 0-9, A-Z, a-z.

## Environment
- Framework: PyTorch 2.10.0+cu130
- CUDA: 13.0
- Device: NVIDIA GeForce RTX 4060 Laptop GPU

## Model
- Model: EMNISTCNN
- Weights: `<LOCAL_PATH>`
- Classes: 62
- Training samples: 697932
- Official test samples: 116323

## Accuracy
- Official full test accuracy: 102978/116323 = 88.53%
- Official full test macro average by class: 77.53%
- Random BMP test set accuracy: 552/620 = 89.03%
- Balanced BMP test set accuracy: 477/620 = 76.94%
- GPU average inference time on official test: 0.0414 ms/image

## Test Sets
- Random standard BMP set: `testsets/EMNIST_STANDARD_RANDOM`
- Balanced per-class BMP set: `testsets/EMNIST_BYCLASS`
- Each folder contains BMP images and `LABEL.TXT` with true labels.

## Main Confusions
- `O` -> `0`: 1727 samples
- `l` -> `1`: 1594 samples
- `0` -> `O`: 906 samples
- `I` -> `1`: 828 samples
- `c` -> `C`: 383 samples
- `s` -> `S`: 372 samples
- `u` -> `U`: 350 samples
- `m` -> `M`: 316 samples
- `1` -> `l`: 272 samples
- `l` -> `I`: 265 samples
- `f` -> `F`: 246 samples
- `o` -> `0`: 233 samples

## Notes
The 62-class ByClass task is harder than digit-only MNIST because many uppercase, lowercase, and digit glyphs are visually ambiguous, such as O/0/o and I/1/l. The random BMP set follows the official test distribution, while the balanced set gives equal weight to every class and is intentionally harder.


