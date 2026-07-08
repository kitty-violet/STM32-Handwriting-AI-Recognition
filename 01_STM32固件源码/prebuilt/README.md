# 预编译 HEX 说明

本目录提供一个可直接烧录的演示 HEX：`Template_demo_placeholder_wifi.hex`。

注意事项：
- 这个 HEX 使用的是开源发布包中的占位 WiFi 配置：`YOUR_WIFI_SSID`、`YOUR_WIFI_PASSWORD`、`YOUR_PC_IP`。
- 因此它适合用于验证程序主体和离线功能；若要连接自己的电脑网页端，必须先修改 `User/wifi_report.h` 中的 WiFi 名称、密码、电脑 IP 和端口，然后重新编译生成新的 HEX。
- 工程实测可由 Keil MDK 5.43 + Arm Compiler 6 编译生成，构建日志为 `0 Error(s), 16 Warning(s)`。这些警告来自原工程兼容性代码，不影响当前演示功能生成 HEX。
