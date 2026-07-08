# STM32 WEB 数据出口

当前 v5.0 固件会在表达式/字符串识别后写入：

- `0:/WEB/LAST.TXT`

字段格式：

```txt
mode=expr
text=12+3=
ok=1
value=15
```

电脑端演示方式：

1. 把 TF 卡插入电脑。
2. 进入 `<TF_CARD_ROOT>\\WEB`。
3. 推荐运行：`python -m http.server 8080`。
4. 浏览器打开 `http://127.0.0.1:8080/index.html`。

后续接 WiFi 模块时，可以让 ESP8266/实验箱读取同样的 `LAST.TXT` 协议，网页端无需改解析格式。



