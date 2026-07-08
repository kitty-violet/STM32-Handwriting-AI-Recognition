# STM32 手写数字与字符识别 AI 开源参考工程

本项目用于课程设计复现和二次开发，包含 STM32 触摸屏手写采集、端侧 FNN/CNN 推理、TF 卡批量测试、WiFi 上报、PC 网页看板、英文单词翻译、PC 端 CNN 辅助识别和语音播报。

## 目录结构

```text
STM32手写识别AI开源参考工程/
  01_STM32固件源码/              STM32 固件源码、Keil 工程、端侧推理代码
  02_PC端网页与训练工具/
    网页接收端/                  TCP 接收、网页看板、翻译、TTS、PC-CNN 服务
    训练与模型工具/              MNIST/EMNIST 训练、评估、测试集制作和模型权重
    requirements.txt             Python 依赖清单
  03_TF卡示例文件/               复制到 microSD 根目录使用的模型、测试集、字库、网页文件
  04_项目说明文档/               功能介绍、使用说明、审核报告
  05_辅助工具/                   发布检查、TF 卡复制、压缩打包脚本
  06_演示视频/                   演示素材目录
```

## 快速开始

1. 将 `03_TF卡示例文件/` 内的内容复制到 TF 卡根目录。
2. 修改 `01_STM32固件源码/User/wifi_report.h` 中的 `YOUR_WIFI_SSID`、`YOUR_WIFI_PASSWORD`、`YOUR_PC_IP`。
3. 使用 Keil MDK 5 + ARM Compiler 6 打开 `01_STM32固件源码/Project/RVMDK（uv5）/BH-F103.uvprojx` 编译烧录。
4. 进入 `02_PC端网页与训练工具/`，执行 `pip install -r requirements.txt`。
5. 进入 `02_PC端网页与训练工具/网页接收端/`，运行 `启动接收端_可见窗口.bat` 或 `python web_receiver.py --deepseek`。
6. 浏览器打开 `http://127.0.0.1:8080/` 查看网页看板。

详细步骤见 `04_项目说明文档/使用说明.md`，功能说明见 `04_项目说明文档/功能介绍.md`。


## 示例 HEX

`01_STM32固件源码/prebuilt/Template_demo_placeholder_wifi.hex` 是已编译好的示例 HEX，方便没有 Keil 环境的同学先烧录体验。

注意：示例 HEX 使用占位 WiFi 配置，能用于体验本地触摸识别、TF 卡测试等主体功能；如果要让 WiFi 连接自己的电脑网页端，需要先修改 `01_STM32固件源码/User/wifi_report.h` 后重新编译。

## 辅助工具

`05_辅助工具/` 不是运行时必需目录，而是给 GitHub 发布和学弟学妹复现项目用的工具目录：

- `检查开源发布包.ps1`：检查关键文件、个人信息残留、编译中间产物和示例 HEX。
- `复制TF卡示例文件.ps1`：把 `03_TF卡示例文件/` 一键复制到 TF 卡。
- `生成发布压缩包.ps1`：生成便于上传 GitHub Release 或网盘分享的 zip 压缩包。


