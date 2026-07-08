#include "wifi_report.h"

#include "misc.h"
#include "stm32f10x_gpio.h"
#include "stm32f10x_rcc.h"
#include "stm32f10x_usart.h"

#include <stdio.h>
#include <string.h>

#define WIFI_USART                 USART3
#define WIFI_USART_CLK             RCC_APB1Periph_USART3
#define WIFI_GPIO_CLK              (RCC_APB2Periph_GPIOB | RCC_APB2Periph_AFIO)
#define WIFI_TX_PORT               GPIOB
#define WIFI_TX_PIN                GPIO_Pin_10
#define WIFI_RX_PORT               GPIOB
#define WIFI_RX_PIN                GPIO_Pin_11
#define WIFI_CH_PD_PORT            GPIOB
#define WIFI_CH_PD_PIN             GPIO_Pin_8
#define WIFI_RST_PORT              GPIOB
#define WIFI_RST_PIN               GPIO_Pin_9
#define WIFI_BAUDRATE              115200
#define WIFI_RX_BUF_LEN            1024U
#define WIFI_PC_CNN_PAYLOAD_MAX    980U

static volatile uint32_t wifi_ms = 0;
static volatile uint16_t wifi_rx_len = 0;
static volatile char wifi_rx_buf[WIFI_RX_BUF_LEN];
static char wifi_pc_payload[WIFI_PC_CNN_PAYLOAD_MAX];
static uint8_t wifi_ready = 0;
static uint8_t wifi_hw_ready = 0;
static void (*wifi_translation_callback)(const char *word, const char *translation) = 0;

static void wifi_report_hw_init(void);
static void wifi_report_delay_ms(uint32_t ms);
static void wifi_report_rx_clear(void);
static uint8_t wifi_report_wait_for(const char *reply1, const char *reply2, uint32_t timeout_ms);
static void wifi_report_send_raw(const char *text);
static uint8_t wifi_report_cmd(const char *cmd, const char *reply1, const char *reply2, uint32_t timeout_ms);
static uint8_t wifi_report_connect_tcp(void);
static void wifi_report_process_rx(void);
static uint8_t wifi_report_process_json_response(const char *json);
static uint8_t wifi_report_extract_json_field(const char *json, const char *field, char *out, uint16_t out_len);

uint8_t wifi_report_init(void)
{
    wifi_ready = 0;

    wifi_report_hw_init();
    printf("[WIFI] GBK reply build\r\n");
    printf("[WIFI] ESP8266 USART3 PB10/PB11 init\r\n");

    GPIO_SetBits(WIFI_CH_PD_PORT, WIFI_CH_PD_PIN);
    GPIO_ResetBits(WIFI_RST_PORT, WIFI_RST_PIN);
    wifi_report_delay_ms(80);
    GPIO_SetBits(WIFI_RST_PORT, WIFI_RST_PIN);
    wifi_report_delay_ms(1200);

    if (!wifi_report_cmd("AT", "OK", 0, 800))
    {
        printf("[WIFI] AT failed, use serial JSON fallback\r\n");
        return 0;
    }

    (void)wifi_report_cmd("ATE0", "OK", 0, 500);
    if (!wifi_report_cmd("AT+CWMODE=1", "OK", "no change", 1200))
    {
        printf("[WIFI] CWMODE failed\r\n");
        return 0;
    }

    if (!wifi_report_cmd("AT+CIPMUX=0", "OK", 0, 800))
    {
        printf("[WIFI] CIPMUX failed\r\n");
        return 0;
    }

    {
        char cmd[120];
        sprintf(cmd, "AT+CWJAP=\"%s\",\"%s\"", WIFI_REPORT_SSID, WIFI_REPORT_PASSWORD);
        printf("[WIFI] Join AP: %s\r\n", WIFI_REPORT_SSID);
        if (!wifi_report_cmd(cmd, "OK", "WIFI GOT IP", 12000))
        {
            printf("[WIFI] Join AP failed\r\n");
            return 0;
        }
    }

    if (!wifi_report_connect_tcp())
    {
        printf("[WIFI] TCP connect failed: %s:%s\r\n", WIFI_REPORT_SERVER_IP, WIFI_REPORT_SERVER_PORT);
        return 0;
    }

    wifi_ready = 1;
    printf("[WIFI] TCP report ready: %s:%s\r\n", WIFI_REPORT_SERVER_IP, WIFI_REPORT_SERVER_PORT);
    wifi_report_send_result("boot", "SYS", "ready", 0);
    return 1;
}

void wifi_report_tick_1ms(void)
{
    wifi_ms++;
}

void wifi_report_poll(void)
{
    wifi_report_process_rx();
}

void wifi_report_set_translation_callback(void (*callback)(const char *word, const char *translation))
{
    wifi_translation_callback = callback;
}

void wifi_report_send_result(const char *mode, const char *model, const char *text, uint32_t infer_us)
{
    char payload[180];
    char cmd[32];
    uint16_t len;
    uint8_t attempt;
    uint8_t need_reply = 0U;

    if (mode == 0) mode = "-";
    if (model == 0) model = "-";
    if (text == 0) text = "-";

    if (strcmp(mode, "word") == 0 || strcmp(mode, "translate") == 0)
    {
        need_reply = 1U;
    }

    sprintf(payload,
            "{\"source\":\"stm32\",\"mode\":\"%s\",\"model\":\"%s\",\"text\":\"%s\",\"infer_us\":%lu}\n",
            mode,
            model,
            text,
            (unsigned long)infer_us);

    printf("[WEB] %s", payload);

    len = (uint16_t)strlen(payload);

    for (attempt = 0U; attempt < 2U; attempt++)
    {
        if (!wifi_ready)
        {
            printf("[WIFI] TCP reconnect...\r\n");
            wifi_ready = wifi_report_connect_tcp();
            if (!wifi_ready)
            {
                continue;
            }
        }

        sprintf(cmd, "AT+CIPSEND=%u", (unsigned int)len);

        if (!wifi_report_cmd(cmd, ">", 0, 1000))
        {
            wifi_ready = 0;
            printf("[WIFI] CIPSEND prompt failed\r\n");
            continue;
        }

        wifi_report_rx_clear();
        wifi_report_send_raw(payload);
        if (wifi_report_wait_for("SEND OK", 0, 1500))
        {
            if (need_reply)
            {
                printf("[WIFI] wait word reply...\r\n");
                if (wifi_report_wait_for("\"result\"", "\"zh_code\"", 30000))
                {
                    printf("[WIFI] word reply marker received\r\n");
                    wifi_report_delay_ms(120);
                    wifi_report_process_rx();
                }
                else
                {
                    printf("[WIFI] word reply timeout\r\n");
                }
            }
            return;
        }

        wifi_ready = 0;
        printf("[WIFI] SEND failed\r\n");
    }

    printf("[WIFI] report dropped\r\n");
}

void wifi_report_send_pc_cnn_bitmap(const uint8_t *bitmap, uint16_t width, uint16_t height)
{
    static const char hex[] = "0123456789ABCDEF";
    char cmd[32];
    uint32_t total_bits;
    uint32_t byte_count;
    uint32_t bit_index;
    uint16_t pos;
    uint8_t attempt;

    if (bitmap == 0 || width == 0U || height == 0U)
    {
        return;
    }

    total_bits = (uint32_t)width * (uint32_t)height;
    byte_count = (total_bits + 7UL) / 8UL;
    if ((byte_count * 2UL + 96UL) >= WIFI_PC_CNN_PAYLOAD_MAX)
    {
        printf("[WIFI] pc_cnn bitmap too large: %lu bits\r\n", (unsigned long)total_bits);
        return;
    }

    sprintf(wifi_pc_payload,
            "{\"source\":\"stm32\",\"mode\":\"pc_cnn\",\"model\":\"PC-CNN\",\"w\":%u,\"h\":%u,\"bits\":\"",
            (unsigned int)width,
            (unsigned int)height);
    pos = (uint16_t)strlen(wifi_pc_payload);

    for (bit_index = 0UL; bit_index < byte_count; bit_index++)
    {
        uint8_t value = 0U;
        uint8_t bit;

        for (bit = 0U; bit < 8U; bit++)
        {
            uint32_t pixel_index = bit_index * 8UL + bit;
            if (pixel_index < total_bits && bitmap[pixel_index] > 0U)
            {
                value |= (uint8_t)(1U << (7U - bit));
            }
        }

        wifi_pc_payload[pos++] = hex[(value >> 4) & 0x0F];
        wifi_pc_payload[pos++] = hex[value & 0x0F];
    }

    wifi_pc_payload[pos++] = '"';
    wifi_pc_payload[pos++] = '}';
    wifi_pc_payload[pos++] = '\n';
    wifi_pc_payload[pos] = '\0';

    printf("[WEB] pc_cnn bitmap %ux%u payload=%u\r\n",
           (unsigned int)width,
           (unsigned int)height,
           (unsigned int)pos);

    for (attempt = 0U; attempt < 2U; attempt++)
    {
        if (!wifi_ready)
        {
            printf("[WIFI] TCP reconnect...\r\n");
            wifi_ready = wifi_report_connect_tcp();
            if (!wifi_ready)
            {
                continue;
            }
        }

        sprintf(cmd, "AT+CIPSEND=%u", (unsigned int)pos);
        if (!wifi_report_cmd(cmd, ">", 0, 1000))
        {
            wifi_ready = 0;
            printf("[WIFI] CIPSEND prompt failed\r\n");
            continue;
        }

        wifi_report_rx_clear();
        wifi_report_send_raw(wifi_pc_payload);
        if (wifi_report_wait_for("SEND OK", 0, 1800))
        {
            if (wifi_report_wait_for("\"pc_cnn_result\"", 0, 3500))
            {
                wifi_report_process_rx();
            }
            return;
        }

        wifi_ready = 0;
        printf("[WIFI] pc_cnn SEND failed\r\n");
    }

    printf("[WIFI] pc_cnn bitmap dropped\r\n");
}

void USART3_IRQHandler(void)
{
    uint8_t ch;

    if (USART_GetITStatus(WIFI_USART, USART_IT_RXNE) != RESET)
    {
        ch = (uint8_t)USART_ReceiveData(WIFI_USART);
        if (wifi_rx_len < (WIFI_RX_BUF_LEN - 1U))
        {
            wifi_rx_buf[wifi_rx_len++] = (char)ch;
            wifi_rx_buf[wifi_rx_len] = '\0';
        }
    }

    if (USART_GetITStatus(WIFI_USART, USART_IT_IDLE) != RESET)
    {
        volatile uint32_t clear;
        clear = WIFI_USART->SR;
        clear = WIFI_USART->DR;
        (void)clear;
    }
}

static void wifi_report_hw_init(void)
{
    GPIO_InitTypeDef gpio;
    USART_InitTypeDef usart;
    NVIC_InitTypeDef nvic;

    if (wifi_hw_ready)
    {
        return;
    }

    RCC_APB2PeriphClockCmd(WIFI_GPIO_CLK, ENABLE);
    RCC_APB1PeriphClockCmd(WIFI_USART_CLK, ENABLE);

    gpio.GPIO_Pin = WIFI_CH_PD_PIN | WIFI_RST_PIN;
    gpio.GPIO_Mode = GPIO_Mode_Out_PP;
    gpio.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(WIFI_CH_PD_PORT, &gpio);

    gpio.GPIO_Pin = WIFI_TX_PIN;
    gpio.GPIO_Mode = GPIO_Mode_AF_PP;
    gpio.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(WIFI_TX_PORT, &gpio);

    gpio.GPIO_Pin = WIFI_RX_PIN;
    gpio.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(WIFI_RX_PORT, &gpio);

    usart.USART_BaudRate = WIFI_BAUDRATE;
    usart.USART_WordLength = USART_WordLength_8b;
    usart.USART_StopBits = USART_StopBits_1;
    usart.USART_Parity = USART_Parity_No;
    usart.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    usart.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
    USART_Init(WIFI_USART, &usart);
    USART_ITConfig(WIFI_USART, USART_IT_RXNE, ENABLE);
    USART_ITConfig(WIFI_USART, USART_IT_IDLE, ENABLE);

    nvic.NVIC_IRQChannel = USART3_IRQn;
    nvic.NVIC_IRQChannelPreemptionPriority = 1;
    nvic.NVIC_IRQChannelSubPriority = 1;
    nvic.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&nvic);

    USART_Cmd(WIFI_USART, ENABLE);
    wifi_hw_ready = 1;
}

static void wifi_report_delay_ms(uint32_t ms)
{
    uint32_t start = wifi_ms;
    while ((uint32_t)(wifi_ms - start) < ms)
    {
    }
}

static void wifi_report_rx_clear(void)
{
    __disable_irq();
    wifi_rx_len = 0;
    wifi_rx_buf[0] = '\0';
    __enable_irq();
}

static uint8_t wifi_report_wait_for(const char *reply1, const char *reply2, uint32_t timeout_ms)
{
    uint32_t start = wifi_ms;

    while ((uint32_t)(wifi_ms - start) < timeout_ms)
    {
        wifi_rx_buf[WIFI_RX_BUF_LEN - 1U] = '\0';
        if (reply1 != 0 && strstr((const char *)wifi_rx_buf, reply1) != 0)
        {
            return 1;
        }
        if (reply2 != 0 && strstr((const char *)wifi_rx_buf, reply2) != 0)
        {
            return 1;
        }
        if (strstr((const char *)wifi_rx_buf, "ERROR") != 0 ||
            strstr((const char *)wifi_rx_buf, "FAIL") != 0)
        {
            return 0;
        }
    }

    return 0;
}

static void wifi_report_send_raw(const char *text)
{
    while (*text)
    {
        USART_SendData(WIFI_USART, (uint8_t)*text++);
        while (USART_GetFlagStatus(WIFI_USART, USART_FLAG_TXE) == RESET)
        {
        }
    }
}

static uint8_t wifi_report_cmd(const char *cmd, const char *reply1, const char *reply2, uint32_t timeout_ms)
{
    wifi_report_rx_clear();
    wifi_report_send_raw(cmd);
    wifi_report_send_raw("\r\n");

    if (reply1 == 0 && reply2 == 0)
    {
        return 1;
    }

    return wifi_report_wait_for(reply1, reply2, timeout_ms);
}

static uint8_t wifi_report_connect_tcp(void)
{
    char cmd[96];

    sprintf(cmd,
            "AT+CIPSTART=\"TCP\",\"%s\",%s",
            WIFI_REPORT_SERVER_IP,
            WIFI_REPORT_SERVER_PORT);

    return wifi_report_cmd(cmd, "OK", "ALREADY CONNECT", 5000);
}

static void wifi_report_process_rx(void)
{
    char snapshot[WIFI_RX_BUF_LEN];
    const char *marker;
    const char *json;
    const char *json_end;
    uint32_t start;

    if (wifi_rx_len == 0U || wifi_translation_callback == 0)
    {
        return;
    }

    __disable_irq();
    strncpy(snapshot, (const char *)wifi_rx_buf, sizeof(snapshot) - 1U);
    snapshot[sizeof(snapshot) - 1U] = '\0';
    __enable_irq();

    marker = strstr(snapshot, "\"result\"");
    if (marker == 0)
    {
        marker = strstr(snapshot, "\"pc_cnn_result\"");
    }
    if (marker == 0)
    {
        marker = strstr(snapshot, "\"word_result\"");
    }
    if (marker == 0)
    {
        marker = strstr(snapshot, "\"translate_result\"");
    }
    if (marker == 0)
    {
        printf("[WIFI] RX marker missing: %.120s\r\n", snapshot);
        wifi_report_rx_clear();
        return;
    }

    json = marker;
    while (json > snapshot && *json != '{')
    {
        json--;
    }
    if (*json != '{')
    {
        printf("[WIFI] RX json start missing: %.120s\r\n", marker);
        wifi_report_rx_clear();
        return;
    }

    json_end = strchr(json, '}');
    start = wifi_ms;
    while (json_end == 0 && (uint32_t)(wifi_ms - start) < 350U)
    {
        __disable_irq();
        strncpy(snapshot, (const char *)wifi_rx_buf, sizeof(snapshot) - 1U);
        snapshot[sizeof(snapshot) - 1U] = '\0';
        __enable_irq();

        marker = strstr(snapshot, "\"result\"");
        if (marker == 0)
        {
            marker = strstr(snapshot, "\"pc_cnn_result\"");
        }
        if (marker == 0)
        {
            marker = strstr(snapshot, "\"word_result\"");
        }
        if (marker == 0)
        {
            marker = strstr(snapshot, "\"translate_result\"");
        }
        if (marker != 0)
        {
            json = marker;
            while (json > snapshot && *json != '{')
            {
                json--;
            }
            if (*json == '{')
            {
                json_end = strchr(json, '}');
            }
        }
    }
    if (json_end == 0)
    {
        printf("[WIFI] RX json end missing: %.120s\r\n", json);
        if (wifi_rx_len >= (WIFI_RX_BUF_LEN - 1U))
        {
            wifi_report_rx_clear();
        }
        return;
    }

    if (wifi_report_process_json_response(json))
    {
        wifi_report_rx_clear();
    }
    else
    {
        printf("[WIFI] RX json parse fail: %.120s\r\n", json);
        wifi_report_rx_clear();
    }
}

static uint8_t wifi_report_process_json_response(const char *json)
{
    char word[32];
    char translation[80];
    char zh_code[24];
    char ascii_text[32];

    if (!wifi_report_extract_json_field(json, "word", word, sizeof(word)) &&
        !wifi_report_extract_json_field(json, "result", word, sizeof(word)))
    {
        word[0] = '\0';
    }

    if (wifi_report_extract_json_field(json, "zh_code", zh_code, sizeof(zh_code)) &&
        zh_code[0] != '\0')
    {
        if (!wifi_report_extract_json_field(json, "ascii", ascii_text, sizeof(ascii_text)) &&
            !wifi_report_extract_json_field(json, "pinyin", ascii_text, sizeof(ascii_text)))
        {
            ascii_text[0] = '\0';
        }
        sprintf(translation, "ZHCODE:%s;PY:%s", zh_code, ascii_text);
    }
    else if (wifi_report_extract_json_field(json, "result", translation, sizeof(translation)))
    {
    }
    else if (!wifi_report_extract_json_field(json, "ascii", translation, sizeof(translation)) &&
        !wifi_report_extract_json_field(json, "pinyin", translation, sizeof(translation)) &&
        !wifi_report_extract_json_field(json, "translation", translation, sizeof(translation)) &&
        !wifi_report_extract_json_field(json, "text", translation, sizeof(translation)))
    {
        printf("[WIFI] RX json no result field\r\n");
        return 0U;
    }

    printf("[WIFI] RX result word=%s text=%s\r\n", word, translation);
    wifi_translation_callback(word, translation);
    return 1U;
}

static uint8_t wifi_report_extract_json_field(const char *json, const char *field, char *out, uint16_t out_len)
{
    char key[32];
    const char *p;
    const char *end;
    uint16_t n = 0U;

    if (json == 0 || field == 0 || out == 0 || out_len == 0U)
    {
        return 0;
    }

    sprintf(key, "\"%s\"", field);
    p = strstr(json, key);
    if (p == 0)
    {
        return 0;
    }

    p += strlen(key);
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }
    if (*p != ':')
    {
        return 0;
    }
    p++;
    while (*p == ' ' || *p == '\t')
    {
        p++;
    }
    if (*p != '"')
    {
        return 0;
    }
    p++;

    end = strchr(p, '"');
    if (end == 0)
    {
        return 0;
    }

    while (p < end && n < (out_len - 1U))
    {
        if (*p == '\\')
        {
            p++;
            if (p >= end)
            {
                break;
            }
        }
        out[n++] = *p++;
    }
    out[n] = '\0';
    return 1;
}


