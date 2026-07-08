/**
  ******************************************************************************
  * @file    main.c
  * @brief   鍩轰簬绁炵粡缃戠粶鐨勬墜鍐欐暟瀛楄瘑鍒郴缁?
  * @author  閭濇稕
  * @class   鑷姩鍖?408
  ******************************************************************************
  */

#include "stm32f10x.h"
#include "./usart/bsp_usart.h"
#include "./led/bsp_led.h"
#ifdef SD_BATCH_TEST_ONLY
#include "sd_batch_test.h"
#ifdef SD_BATCH_TEST_LCD
#include "./lcd/bsp_ili9341_lcd.h"
#endif
#else
#include "./lcd/bsp_ili9341_lcd.h"
#include "./lcd/bsp_xpt2046_lcd.h"
#include "digit_app.h"
#include "emnist_cnn.h"
#include "fnn_model.h"
#include "sd_batch_test.h"
#include "wifi_report.h"
#include "./FATFS/ff.h"
#include "./key/bsp_key.h"
#endif

#ifndef SD_BATCH_TEST_ONLY
static FATFS g_app_fs;
typedef enum
{
    APP_MODE_SINGLE = 0,
    APP_MODE_SD_BATCH = 1,
    APP_MODE_STRING = 2,
    APP_MODE_TRANSLATE = 3,
    APP_MODE_PC_CNN = 4
} AppMode;

static AppMode g_app_mode = APP_MODE_SINGLE;
static uint8_t app_load_models(void);
static void app_show_boot_text(const char *line, uint16_t color);
static void app_draw_status_bar(void);
static void app_enter_touch_mode(DigitInputMode input_mode);
static void app_enter_current_touch_mode(void);
static void app_toggle_model(void);
static void app_switch_mode(void);
static void app_run_sd_batch_mode(void);
static uint8_t app_sd_batch_abort_requested(void);
static void app_translation_received(const char *word, const char *translation);
#endif

int main(void)
{
#ifdef SD_BATCH_TEST_ONLY
    USART_Config();

#ifdef SD_BATCH_TEST_LCD
    ILI9341_Init();
    ILI9341_GramScan(3);
#endif

#ifndef SD_BATCH_TEST_LCD
    printf("\r\n ********** SD batch digit test *********** \r\n");
#endif

    sd_batch_test_run_once(0);

    while (1)
    {
    }
#else
    ILI9341_Init();
    XPT2046_Init();
    USART_Config();
    LED_GPIO_Config();
    Key_GPIO_Config();
    SysTick_Config(SystemCoreClock / 1000);

    printf("\r\n ********** Handwritten digit recognition *********** \r\n");
    printf("Project: Neural-network handwritten digit recognition\r\n");
    printf("Author : Open Course Project");
    printf("Class  : Embedded AI Demo");

    ILI9341_GramScan(3);
    if (!app_load_models())
    {
        while (1)
        {
        }
    }

    app_show_boot_text("WIFI INIT", CYAN);
    wifi_report_set_translation_callback(app_translation_received);
    if (wifi_report_init())
    {
        app_show_boot_text("WIFI READY", GREEN);
        printf("[APP] WiFi report ready\r\n");
    }
    else
    {
        app_show_boot_text("WIFI SERIAL", YELLOW);
        printf("[APP] WiFi report unavailable, serial JSON still enabled\r\n");
    }
    digit_app_set_model(DIGIT_MODEL_FNN);
    g_app_mode = APP_MODE_SINGLE;
    app_enter_current_touch_mode();

    while (1)
    {
        if (Key_PressedEvent(KEY1_GPIO_PORT, KEY1_GPIO_PIN))
        {
            app_switch_mode();
        }
        if (Key_PressedEvent(KEY2_GPIO_PORT, KEY2_GPIO_PIN))
        {
            app_toggle_model();
        }
        XPT2046_TouchEvenHandler();
        digit_app_process();
        wifi_report_poll();
    }
#endif
}

#ifndef SD_BATCH_TEST_ONLY
static uint8_t app_load_models(void)
{
    FRESULT result;
    uint8_t fnn_loaded;
    uint8_t cnn_loaded;

    app_show_boot_text("LOAD FNN", CYAN);
    printf("[APP] Mount TF card for FNN model...\r\n");

    result = f_mount(&g_app_fs, "0:", 1);
    if (result != FR_OK)
    {
        app_show_boot_text("TF MOUNT FAIL", RED);
        printf("[APP] TF mount failed: %d\r\n", result);
        return 0;
    }

    fnn_loaded = fnn_model_load(FNN_MODEL_PATH);
    cnn_loaded = emnist_cnn_model_load(0);

    if (!fnn_loaded)
    {
        app_show_boot_text("FNN FAIL", RED);
        printf("[APP] Model load failed: %s\r\n", FNN_MODEL_PATH);
        return 0;
    }

    if (!cnn_loaded)
    {
        app_show_boot_text("CNN FAIL", RED);
        printf("[APP] CNN int8 Flash model unavailable\r\n");
        return 0;
    }

    app_show_boot_text("MODELS OK", GREEN);
    printf("[APP] FNN model loaded from TF: %s\r\n", FNN_MODEL_PATH);
    printf("[APP] CNN model uses int8 Flash weights\r\n");
    return 1;
}

static void app_show_boot_text(const char *line, uint16_t color)
{
    LCD_SetBackColor(BLACK);
    ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH);
    digit_app_draw_ascii(20, 70, "HAND DIGIT", WHITE, 2);
    digit_app_draw_ascii(20, 110, line, color, 2);
}

static void app_draw_status_bar(void)
{
    LCD_SetColors(BLACK, BLACK);
    ILI9341_DrawRectangle(0, 230, LCD_X_LENGTH, 10, 1);
    digit_app_draw_ascii(8, 232, "K1 MODE", CYAN, 1);
    if (g_app_mode == APP_MODE_PC_CNN)
    {
        digit_app_draw_ascii(88, 232, "PC CNN", YELLOW, 1);
    }
    else if (digit_app_get_model() == DIGIT_MODEL_FNN)
    {
        digit_app_draw_ascii(88, 232, "K2 FNN", GREEN, 1);
    }
    else
    {
        digit_app_draw_ascii(88, 232, "K2 CNN", YELLOW, 1);
    }

    if (g_app_mode == APP_MODE_STRING)
    {
        digit_app_draw_ascii(168, 232, "STRING", YELLOW, 1);
    }
    else if (g_app_mode == APP_MODE_TRANSLATE)
    {
        digit_app_draw_ascii(168, 232, "TRANS", YELLOW, 1);
    }
    else if (g_app_mode == APP_MODE_PC_CNN)
    {
        digit_app_draw_ascii(168, 232, "REMOTE", YELLOW, 1);
    }
    else
    {
        digit_app_draw_ascii(168, 232, "SINGLE", WHITE, 1);
    }
}

static void app_enter_touch_mode(DigitInputMode input_mode)
{
    digit_app_set_input_mode(input_mode);
    digit_app_init();
    app_draw_status_bar();
    Key_ResetAllEvents();
}

static void app_enter_current_touch_mode(void)
{
    if (g_app_mode == APP_MODE_STRING)
    {
        app_enter_touch_mode(DIGIT_INPUT_STRING);
        printf("[APP] Enter string mode\r\n");
    }
    else if (g_app_mode == APP_MODE_TRANSLATE)
    {
        app_enter_touch_mode(DIGIT_INPUT_TRANSLATE);
        printf("[APP] Enter translate mode\r\n");
    }
    else if (g_app_mode == APP_MODE_PC_CNN)
    {
        app_enter_touch_mode(DIGIT_INPUT_PC_CNN);
        printf("[APP] Enter PC-CNN mode\r\n");
    }
    else
    {
        g_app_mode = APP_MODE_SINGLE;
        app_enter_touch_mode(DIGIT_INPUT_SINGLE);
        printf("[APP] Enter single-char mode\r\n");
    }
}

static void app_toggle_model(void)
{
    if (g_app_mode == APP_MODE_PC_CNN)
    {
        printf("[APP] KEY2 ignored in PC-CNN mode\r\n");
        return;
    }

    if (digit_app_get_model() == DIGIT_MODEL_FNN)
    {
        if (!emnist_cnn_model_is_loaded())
        {
            printf("[APP] CNN model is not loaded\r\n");
            return;
        }
        digit_app_set_model(DIGIT_MODEL_CNN);
        printf("[APP] Model switched to CNN\r\n");
    }
    else
    {
        digit_app_set_model(DIGIT_MODEL_FNN);
        printf("[APP] Model switched to FNN\r\n");
    }

    app_enter_current_touch_mode();
}

static void app_switch_mode(void)
{
    if (g_app_mode == APP_MODE_SINGLE)
    {
        g_app_mode = APP_MODE_SD_BATCH;
        app_run_sd_batch_mode();
        g_app_mode = APP_MODE_STRING;
        app_enter_current_touch_mode();
    }
    else if (g_app_mode == APP_MODE_STRING)
    {
        g_app_mode = APP_MODE_TRANSLATE;
        app_enter_current_touch_mode();
    }
    else if (g_app_mode == APP_MODE_TRANSLATE)
    {
        g_app_mode = APP_MODE_PC_CNN;
        app_enter_current_touch_mode();
    }
    else if (g_app_mode == APP_MODE_PC_CNN)
    {
        g_app_mode = APP_MODE_SINGLE;
        app_enter_current_touch_mode();
    }
}

static void app_run_sd_batch_mode(void)
{
    uint8_t completed;

    printf("[APP] Enter SD batch mode\r\n");
    completed = sd_batch_test_run_once(app_sd_batch_abort_requested);
    if (!completed)
    {
        printf("[APP] SD batch interrupted by KEY1\r\n");
        Key_ResetAllEvents();
        return;
    }

    digit_app_draw_ascii(8, 216, "K1 BACK", CYAN, 1);
    Key_ResetAllEvents();

    while (!Key_PressedEvent(KEY1_GPIO_PORT, KEY1_GPIO_PIN))
    {
        wifi_report_poll();
    }

    printf("[APP] Leave SD batch mode\r\n");
}

static uint8_t app_sd_batch_abort_requested(void)
{
    wifi_report_poll();
    return Key_PressedEvent(KEY1_GPIO_PORT, KEY1_GPIO_PIN);
}

static void app_translation_received(const char *word, const char *translation)
{
    digit_app_show_translation(word, translation);
}
#endif


