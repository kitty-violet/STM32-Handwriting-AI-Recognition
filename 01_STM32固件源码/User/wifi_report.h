#ifndef WIFI_REPORT_H
#define WIFI_REPORT_H

#include "stm32f10x.h"
#include <stdint.h>

/*
 * ESP8266 wiring matches the experiment8 WiFi example:
 * USART3 TX/RX: PB10/PB11, CH_PD: PB8, RST: PB9.
 * Change the three macros below before compiling if your hotspot or PC IP changes.
 */
#define WIFI_REPORT_SSID          "YOUR_WIFI_SSID"
#define WIFI_REPORT_PASSWORD      "YOUR_WIFI_PASSWORD"
#define WIFI_REPORT_SERVER_IP     "YOUR_PC_IP"
#define WIFI_REPORT_SERVER_PORT   "8000"

uint8_t wifi_report_init(void);
void wifi_report_tick_1ms(void);
void wifi_report_poll(void);
void wifi_report_send_result(const char *mode, const char *model, const char *text, uint32_t infer_us);
void wifi_report_send_pc_cnn_bitmap(const uint8_t *bitmap, uint16_t width, uint16_t height);
void wifi_report_set_translation_callback(void (*callback)(const char *word, const char *translation));

#endif



