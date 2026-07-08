# 训练与模型工具

本目录用于训练、评估和导出 MNIST/EMNIST 手写识别模型。

## 常用脚本

- `train_cnn_mnist.py`：训练 MNIST 数字 CNN。
- `train_cnn_emnist.py`：训练 EMNIST 数字、大小写字母 CNN。
- `train_cnn_emnist_subset.py`：训练 EMNIST 子集模型。
- `make_testsets.py`：生成 MNIST/个人测试集 BMP 文件。
- `make_emnist_testsets.py`：生成 EMNIST BMP 测试集。
- `evaluate_testsets.py`：评估 MNIST/个人测试集准确率。
- `evaluate_emnist_testsets.py`：评估 EMNIST 测试集准确率。
- `export_fnn_bin.py`：导出 FNN 二进制权重。

## 数据路径

原始 MNIST/EMNIST 数据不在本目录内完整发布。建议放在：

```text
02_PC端网页与训练工具/训练与模型工具/data/
```

也可以通过脚本参数传入自定义路径，例如：

```powershell
python train_cnn_mnist.py --raw-dir .\data\MNIST\raw --epochs 8 --batch-size 256
python make_testsets.py --raw-dir .\data\MNIST\raw --per-digit 10 --sd-root X:\
python evaluate_testsets.py --root X:\TEST --repeat 10 --report reports\test_eval_sd.json
```

`X:\` 表示你的 TF 卡盘符，请按实际情况替换。

## 模型权重

`models/` 中包含已训练模型权重，可用于 PC-CNN 协同识别或继续训练。发布到公开仓库前请说明数据集来源和课程用途。

