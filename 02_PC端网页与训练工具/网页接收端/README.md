# PC 端网页接收端

本目录提供 STM32 WiFi 上报的 PC 端服务：

- TCP 接收 STM32 JSON/位图数据。
- 本地网页看板 `http://127.0.0.1:8080/`。
- 英文单词翻译、语音播报和结果回传。
- PC-CNN 协同识别。

## 使用方法

先修改 STM32 固件中的 WiFi 配置：

```c
#define WIFI_REPORT_SSID          "YOUR_WIFI_SSID"
#define WIFI_REPORT_PASSWORD      "YOUR_WIFI_PASSWORD"
#define WIFI_REPORT_SERVER_IP     "YOUR_PC_IP"
#define WIFI_REPORT_SERVER_PORT   "8000"
```

电脑端运行：

```powershell
python web_receiver.py
```

需要 DeepSeek 网页翻译时运行：

```powershell
python web_receiver.py --deepseek
```

也可以双击 `启动接收端_可见窗口.bat`。演示时只保留一个接收端进程，避免 STM32 连接到旧后台进程。

