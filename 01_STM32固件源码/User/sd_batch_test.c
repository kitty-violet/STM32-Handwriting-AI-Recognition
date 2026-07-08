#include "sd_batch_test.h"

#include "FNN_Data.h"
#ifndef SD_BATCH_TEST_ONLY
#include "digit_app.h"
#include "emnist_cnn.h"
#include "wifi_report.h"
#endif
#include "fnn.h"
#include "./FATFS/ff.h"
#include "stm32f10x.h"
#ifdef SD_BATCH_TEST_LCD
#include "./lcd/bsp_ili9341_lcd.h"
#include "./usart/bsp_usart.h"
#endif

#include <stdio.h>
#include <string.h>

#define SD_TEST_REPEAT 1U
#define SD_TEST_MAX_PATH 64U
#define SD_TEST_BMP_HEADER_SIZE 54U
#define SD_TEST_BMP_ROW_SIZE 84U

typedef struct
{
    uint16_t total;
    uint16_t correct;
    uint32_t infer_cycles;
} SdBatchStats;

static FATFS g_sd_fs;
static uint8_t g_sd_image[FNN_INPUT_SIZE];
static uint8_t g_sd_bmp_header[SD_TEST_BMP_HEADER_SIZE];
static uint8_t g_sd_bmp_row[SD_TEST_BMP_ROW_SIZE];
static SdBatchAbortCallback g_sd_abort_callback = 0;
static uint8_t g_sd_abort_requested = 0;

#ifdef SD_BATCH_TEST_LCD
static void sd_batch_test_lcd_init(void);
static void sd_batch_test_lcd_line(uint16_t y, const char *text);
static void sd_batch_test_lcd_summary(uint16_t y, const char *name, const SdBatchStats *stats);
static char *sd_batch_test_lcd_append_u32(char *dst, uint32_t value);
static void sd_batch_test_lcd_text(uint16_t x, uint16_t y, const char *text, uint16_t color);
static void sd_batch_test_lcd_char(uint16_t x, uint16_t y, char ch);
static const uint8_t *sd_batch_test_lcd_glyph(char ch);
static void sd_batch_test_serial_summary(const char *name, const SdBatchStats *stats);
static void sd_batch_test_serial_puts(const char *text);
static void sd_batch_test_serial_u32(uint32_t value);
#endif
static uint8_t sd_batch_test_should_abort(void);
static uint8_t sd_batch_test_run_mnist(void);
static uint8_t sd_batch_test_run_personal(void);
static void sd_batch_test_run_file(const char *dataset, const char *path, uint8_t label, SdBatchStats *stats);
static uint8_t sd_batch_test_read_bmp28(const char *path, uint8_t image[FNN_INPUT_SIZE]);
static void sd_batch_test_build_mnist_path(char *path, uint16_t serial, uint8_t label);
static void sd_batch_test_build_personal_path(char *path, uint16_t serial, uint8_t label);
static void sd_batch_test_write_u3(char *dst, uint16_t value);
static uint32_t sd_batch_test_cycles_now(void);
static void sd_batch_test_cycles_init(void);
#ifndef SD_BATCH_TEST_LCD
static void sd_batch_test_print_result(const char *name, const SdBatchStats *stats);
#endif
#ifndef SD_BATCH_TEST_ONLY
static const char *sd_batch_test_model_name(void);
static void sd_batch_test_web_summary(const char *name, const SdBatchStats *stats);
#endif

uint8_t sd_batch_test_run_once(SdBatchAbortCallback abort_callback)
{
    FRESULT result;

    g_sd_abort_callback = abort_callback;
    g_sd_abort_requested = 0U;

#ifdef SD_BATCH_TEST_LCD
    sd_batch_test_lcd_init();
    sd_batch_test_lcd_line(42, "MOUNT TF CARD");
#endif
#ifndef SD_BATCH_TEST_LCD
    printf("\r\n[SDTEST] Mount TF card...\r\n");
#endif
    result = f_mount(&g_sd_fs, "0:", 1);
    if (result != FR_OK)
    {
#ifndef SD_BATCH_TEST_LCD
        printf("[SDTEST] mount failed: %d\r\n", result);
#endif
#ifdef SD_BATCH_TEST_LCD
        sd_batch_test_lcd_line(66, "MOUNT FAILED");
        sd_batch_test_serial_puts("[SDTEST] MOUNT FAILED\r\n");
#endif
        g_sd_abort_callback = 0;
        return 0U;
    }

    sd_batch_test_cycles_init();
    if (sd_batch_test_run_mnist())
    {
        sd_batch_test_run_personal();
    }

    if (g_sd_abort_requested)
    {
#ifndef SD_BATCH_TEST_LCD
        printf("[SDTEST] aborted by KEY1\r\n");
#endif
#ifdef SD_BATCH_TEST_LCD
        sd_batch_test_lcd_line(198, "ABORTED BY KEY1");
        sd_batch_test_serial_puts("[SDTEST] ABORTED BY KEY1\r\n");
#endif
    }

    g_sd_abort_callback = 0;
    return g_sd_abort_requested ? 0U : 1U;
}

static uint8_t sd_batch_test_should_abort(void)
{
    if (!g_sd_abort_requested && g_sd_abort_callback != 0 && g_sd_abort_callback())
    {
        g_sd_abort_requested = 1U;
    }

    return g_sd_abort_requested;
}

static uint8_t sd_batch_test_run_mnist(void)
{
    SdBatchStats stats = {0, 0, 0};
    uint8_t digit;
    uint8_t sample;

#ifndef SD_BATCH_TEST_LCD
    printf("[SDTEST] Dataset MNIST\r\n");
#endif
#ifdef SD_BATCH_TEST_LCD
    sd_batch_test_lcd_line(42, "TESTING MNIST");
#endif
    for (digit = 0; digit < 10; digit++)
    {
        for (sample = 0; sample < 10; sample++)
        {
            char path[SD_TEST_MAX_PATH];
            uint16_t serial = (uint16_t)digit * 10U + sample;
            if (sd_batch_test_should_abort())
            {
                goto mnist_done;
            }
            sd_batch_test_build_mnist_path(path, serial, digit);
            sd_batch_test_run_file("MNIST", path, digit, &stats);
            if (sd_batch_test_should_abort())
            {
                goto mnist_done;
            }
        }
    }
mnist_done:
#ifndef SD_BATCH_TEST_LCD
    sd_batch_test_print_result("MNIST", &stats);
#endif
#ifdef SD_BATCH_TEST_LCD
    sd_batch_test_lcd_summary(72, "MNIST", &stats);
    sd_batch_test_serial_summary("MNIST", &stats);
#ifndef SD_BATCH_TEST_ONLY
    sd_batch_test_web_summary("MNIST", &stats);
#endif
#endif
    return g_sd_abort_requested ? 0U : 1U;
}

static uint8_t sd_batch_test_run_personal(void)
{
    SdBatchStats stats = {0, 0, 0};
    uint8_t digit;

#ifndef SD_BATCH_TEST_LCD
    printf("[SDTEST] Dataset PERSONAL\r\n");
#endif
#ifdef SD_BATCH_TEST_LCD
    sd_batch_test_lcd_line(42, "TESTING PERSONAL");
#endif
    for (digit = 0; digit < 10; digit++)
    {
        char path[SD_TEST_MAX_PATH];
        if (sd_batch_test_should_abort())
        {
            goto personal_done;
        }
        sd_batch_test_build_personal_path(path, digit, digit);
        sd_batch_test_run_file("PERSONAL", path, digit, &stats);
        if (sd_batch_test_should_abort())
        {
            goto personal_done;
        }
    }
personal_done:
#ifndef SD_BATCH_TEST_LCD
    sd_batch_test_print_result("PERSONAL", &stats);
#endif
#ifdef SD_BATCH_TEST_LCD
    sd_batch_test_lcd_summary(126, "PERSONAL", &stats);
    sd_batch_test_serial_summary("PERSONAL", &stats);
#ifndef SD_BATCH_TEST_ONLY
    sd_batch_test_web_summary("PERSONAL", &stats);
#endif
    sd_batch_test_lcd_line(198, "DONE");
#endif
    return g_sd_abort_requested ? 0U : 1U;
}

static void sd_batch_test_run_file(const char *dataset, const char *path, uint8_t label, SdBatchStats *stats)
{
    float fnn_logits[FNN_OUTPUT_SIZE];
#ifndef SD_BATCH_TEST_ONLY
    float cnn_logits[EMNIST_CNN_OUTPUT_SIZE];
#endif
    uint8_t pred = 0;
    uint8_t repeat;
    uint32_t start_cycles;
    uint32_t elapsed_cycles = 0;

#ifdef SD_BATCH_TEST_LCD
    (void)dataset;
#endif

    if (!sd_batch_test_read_bmp28(path, g_sd_image))
    {
#ifndef SD_BATCH_TEST_LCD
        printf("[SDTEST] %s skip: %s\r\n", dataset, path);
#endif
        return;
    }

    for (repeat = 0; repeat < SD_TEST_REPEAT; repeat++)
    {
        if (sd_batch_test_should_abort())
        {
            return;
        }
        start_cycles = sd_batch_test_cycles_now();
#ifndef SD_BATCH_TEST_ONLY
        if (digit_app_get_model() == DIGIT_MODEL_CNN)
        {
            pred = emnist_cnn_predict(g_sd_image, cnn_logits);
        }
        else
#endif
        {
            pred = fnn_predict(g_sd_image, fnn_logits);
        }
        elapsed_cycles += sd_batch_test_cycles_now() - start_cycles;
    }
    elapsed_cycles /= SD_TEST_REPEAT;

    stats->total++;
    stats->infer_cycles += elapsed_cycles;
    if (pred == label)
    {
        stats->correct++;
    }

#ifndef SD_BATCH_TEST_LCD
    printf("[SDTEST] %s label=%d pred=%d cycles=%lu %s\r\n",
           path,
           label,
           pred,
           elapsed_cycles,
           pred == label ? "OK" : "ERR");
#endif
}

static uint8_t sd_batch_test_read_bmp28(const char *path, uint8_t image[FNN_INPUT_SIZE])
{
    FIL file;
    FRESULT result;
    UINT read_count;
    uint32_t offset;
    int32_t width;
    int32_t height;
    uint16_t bit_count;
    uint16_t row_size;
    uint8_t top_down = 0;
    uint8_t row;
    uint32_t pos;

    memset(image, 0, FNN_INPUT_SIZE);

    result = f_open(&file, path, FA_READ);
    if (result != FR_OK)
    {
        return 0;
    }

    result = f_read(&file, g_sd_bmp_header, SD_TEST_BMP_HEADER_SIZE, &read_count);
    if (result != FR_OK || read_count != SD_TEST_BMP_HEADER_SIZE)
    {
        f_close(&file);
        return 0;
    }

    if (g_sd_bmp_header[0] != 'B' || g_sd_bmp_header[1] != 'M')
    {
        f_close(&file);
        return 0;
    }

    offset = (uint32_t)g_sd_bmp_header[10] |
             ((uint32_t)g_sd_bmp_header[11] << 8) |
             ((uint32_t)g_sd_bmp_header[12] << 16) |
             ((uint32_t)g_sd_bmp_header[13] << 24);
    width = (int32_t)g_sd_bmp_header[18] |
            ((int32_t)g_sd_bmp_header[19] << 8) |
            ((int32_t)g_sd_bmp_header[20] << 16) |
            ((int32_t)g_sd_bmp_header[21] << 24);
    height = (int32_t)g_sd_bmp_header[22] |
             ((int32_t)g_sd_bmp_header[23] << 8) |
             ((int32_t)g_sd_bmp_header[24] << 16) |
             ((int32_t)g_sd_bmp_header[25] << 24);
    bit_count = (uint16_t)g_sd_bmp_header[28] |
                ((uint16_t)g_sd_bmp_header[29] << 8);

    if (height < 0)
    {
        height = -height;
        top_down = 1;
    }

    if (width != 28 || height != 28 || (bit_count != 8 && bit_count != 24))
    {
        f_close(&file);
        return 0;
    }

    row_size = (uint16_t)((((uint32_t)width * bit_count + 31U) / 32U) * 4U);
    if (row_size > SD_TEST_BMP_ROW_SIZE)
    {
        f_close(&file);
        return 0;
    }

    pos = SD_TEST_BMP_HEADER_SIZE;
    while (pos < offset)
    {
        UINT chunk = (UINT)(offset - pos);
        if (chunk > sizeof(g_sd_bmp_row))
        {
            chunk = sizeof(g_sd_bmp_row);
        }
        result = f_read(&file, g_sd_bmp_row, chunk, &read_count);
        if (result != FR_OK || read_count != chunk)
        {
            f_close(&file);
            return 0;
        }
        pos += chunk;
    }

    for (row = 0; row < 28; row++)
    {
        uint8_t col;
        uint8_t dst_y = top_down ? row : (uint8_t)(27 - row);

        result = f_read(&file, g_sd_bmp_row, row_size, &read_count);
        if (result != FR_OK || read_count != row_size)
        {
            f_close(&file);
            return 0;
        }

        for (col = 0; col < 28; col++)
        {
            if (bit_count == 8)
            {
                image[(uint16_t)dst_y * 28U + col] = g_sd_bmp_row[col];
            }
            else
            {
                uint8_t blue = g_sd_bmp_row[col * 3U + 0U];
                uint8_t green = g_sd_bmp_row[col * 3U + 1U];
                uint8_t red = g_sd_bmp_row[col * 3U + 2U];
                uint16_t gray = (uint16_t)red + (uint16_t)green + (uint16_t)blue;
                image[(uint16_t)dst_y * 28U + col] = (uint8_t)(gray / 3U);
            }
        }
    }

    f_close(&file);
    return 1;
}

static void sd_batch_test_build_mnist_path(char *path, uint16_t serial, uint8_t label)
{
    strcpy(path, "0:/TEST/MNIST/M000_0.BMP");
    sd_batch_test_write_u3(&path[15], serial);
    path[19] = (char)('0' + label);
}

static void sd_batch_test_build_personal_path(char *path, uint16_t serial, uint8_t label)
{
    strcpy(path, "0:/TEST/PERSONAL/P000_0.BMP");
    sd_batch_test_write_u3(&path[18], serial);
    path[22] = (char)('0' + label);
}

static void sd_batch_test_write_u3(char *dst, uint16_t value)
{
    dst[0] = (char)('0' + ((value / 100U) % 10U));
    dst[1] = (char)('0' + ((value / 10U) % 10U));
    dst[2] = (char)('0' + (value % 10U));
}

static void sd_batch_test_cycles_init(void)
{
    volatile uint32_t *demcr = (uint32_t *)0xE000EDFCUL;
    volatile uint32_t *dwt_ctrl = (uint32_t *)0xE0001000UL;
    volatile uint32_t *dwt_cyccnt = (uint32_t *)0xE0001004UL;

    *demcr |= (1UL << 24);
    *dwt_cyccnt = 0;
    *dwt_ctrl |= 1UL;
}

static uint32_t sd_batch_test_cycles_now(void)
{
    volatile uint32_t *dwt_cyccnt = (uint32_t *)0xE0001004UL;
    return *dwt_cyccnt;
}

#ifndef SD_BATCH_TEST_LCD
static void sd_batch_test_print_result(const char *name, const SdBatchStats *stats)
{
    uint32_t acc_x100 = 0;
    uint32_t avg_cycles = 0;
    uint32_t avg_us = 0;

    if (stats->total > 0)
    {
        acc_x100 = ((uint32_t)stats->correct * 10000UL + stats->total / 2U) / stats->total;
        avg_cycles = stats->infer_cycles / stats->total;
        avg_us = (avg_cycles + (SystemCoreClock / 2000000UL)) / (SystemCoreClock / 1000000UL);
    }

    printf("[SDTEST] %s summary: total=%u correct=%u acc=%lu.%02lu%% avg_cycles=%lu avg_us=%lu\r\n",
           name,
           stats->total,
           stats->correct,
           acc_x100 / 100UL,
           acc_x100 % 100UL,
           avg_cycles,
           avg_us);
#ifndef SD_BATCH_TEST_ONLY
    sd_batch_test_web_summary(name, stats);
#endif
}
#endif

#ifndef SD_BATCH_TEST_ONLY
static const char *sd_batch_test_model_name(void)
{
    return (digit_app_get_model() == DIGIT_MODEL_CNN) ? "CNN" : "FNN";
}

static void sd_batch_test_web_summary(const char *name, const SdBatchStats *stats)
{
    char text[56];
    uint32_t acc_x100 = 0;
    uint32_t avg_cycles = 0;
    uint32_t avg_us = 0;

    if (stats->total > 0)
    {
        acc_x100 = ((uint32_t)stats->correct * 10000UL + stats->total / 2U) / stats->total;
        avg_cycles = stats->infer_cycles / stats->total;
        avg_us = (avg_cycles + (SystemCoreClock / 2000000UL)) / (SystemCoreClock / 1000000UL);
    }

    sprintf(text,
            "%s %u/%u %lu.%02lu%%",
            name,
            stats->correct,
            stats->total,
            acc_x100 / 100UL,
            acc_x100 % 100UL);
    wifi_report_send_result("sdtest", sd_batch_test_model_name(), text, avg_us);
}
#endif

#ifdef SD_BATCH_TEST_LCD
typedef struct
{
    char ch;
    uint8_t rows[7];
} SdBatchGlyph;

#define SD_LCD_SCALE 2U
#define SD_LCD_CHAR_W 5U
#define SD_LCD_CHAR_H 7U
#define SD_LCD_STEP_X 12U

static const SdBatchGlyph g_sd_batch_lcd_font[] = {
    {' ', {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
    {'%', {0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13}},
    {'.', {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C}},
    {'/', {0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10}},
    {':', {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}},
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}},
    {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}},
    {'3', {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}},
    {'5', {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E}},
    {'6', {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E}},
    {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}},
    {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E}},
    {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}},
    {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}},
    {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}},
    {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}},
    {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}},
    {'G', {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F}},
    {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}},
    {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}},
    {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}},
    {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
    {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
    {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}},
    {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
    {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'V', {0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04}},
    {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
};

static void sd_batch_test_lcd_init(void)
{
    LCD_SetBackColor(BLACK);
    LCD_SetTextColor(WHITE);
    ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH);

    sd_batch_test_lcd_text(8, 10, "TF CARD BATCH TEST", CYAN);
}

static void sd_batch_test_lcd_line(uint16_t y, const char *text)
{
    LCD_SetBackColor(BLACK);
    ILI9341_Clear(0, y, LCD_X_LENGTH, 20);
    sd_batch_test_lcd_text(8, y, text, WHITE);
}

static void sd_batch_test_lcd_summary(uint16_t y, const char *name, const SdBatchStats *stats)
{
    char line[40];
    uint32_t acc_x100 = 0;
    uint32_t avg_cycles = 0;
    uint32_t avg_us = 0;

    if (stats->total > 0)
    {
        acc_x100 = ((uint32_t)stats->correct * 10000UL + stats->total / 2U) / stats->total;
        avg_cycles = stats->infer_cycles / stats->total;
        avg_us = (avg_cycles + (SystemCoreClock / 2000000UL)) / (SystemCoreClock / 1000000UL);
    }

    LCD_SetBackColor(BLACK);
    sd_batch_test_lcd_text(8, y, name, YELLOW);

    strcpy(line, "CORRECT: ");
    sd_batch_test_lcd_append_u32(line + strlen(line), stats->correct);
    strcat(line, "/");
    sd_batch_test_lcd_append_u32(line + strlen(line), stats->total);
    sd_batch_test_lcd_text(8, y + 18U, line, WHITE);

    strcpy(line, "ACC: ");
    sd_batch_test_lcd_append_u32(line + strlen(line), acc_x100 / 100UL);
    strcat(line, ".");
    if ((acc_x100 % 100UL) < 10UL)
    {
        strcat(line, "0");
    }
    sd_batch_test_lcd_append_u32(line + strlen(line), acc_x100 % 100UL);
    strcat(line, "%");
    sd_batch_test_lcd_text(8, y + 36U, line, WHITE);

    strcpy(line, "AVG: ");
    sd_batch_test_lcd_append_u32(line + strlen(line), avg_us);
    strcat(line, " US");
    sd_batch_test_lcd_text(8, y + 54U, line, WHITE);
}

static char *sd_batch_test_lcd_append_u32(char *dst, uint32_t value)
{
    char tmp[10];
    uint8_t len = 0;

    if (value == 0U)
    {
        *dst++ = '0';
        *dst = '\0';
        return dst;
    }

    while (value > 0U)
    {
        tmp[len++] = (char)('0' + (value % 10U));
        value /= 10U;
    }

    while (len > 0U)
    {
        *dst++ = tmp[--len];
    }
    *dst = '\0';
    return dst;
}

static void sd_batch_test_lcd_text(uint16_t x, uint16_t y, const char *text, uint16_t color)
{
    LCD_SetTextColor(color);
    while (*text != '\0')
    {
        if ((x + (SD_LCD_CHAR_W * SD_LCD_SCALE)) >= LCD_X_LENGTH)
        {
            break;
        }
        sd_batch_test_lcd_char(x, y, *text);
        x += SD_LCD_STEP_X;
        text++;
    }
}

static void sd_batch_test_lcd_char(uint16_t x, uint16_t y, char ch)
{
    const uint8_t *glyph = sd_batch_test_lcd_glyph(ch);
    uint8_t row;
    uint8_t col;
    uint8_t sx;
    uint8_t sy;

    for (row = 0U; row < SD_LCD_CHAR_H; row++)
    {
        for (col = 0U; col < SD_LCD_CHAR_W; col++)
        {
            if ((glyph[row] & (uint8_t)(1U << (SD_LCD_CHAR_W - 1U - col))) != 0U)
            {
                for (sy = 0U; sy < SD_LCD_SCALE; sy++)
                {
                    for (sx = 0U; sx < SD_LCD_SCALE; sx++)
                    {
                        ILI9341_SetPointPixel(x + (uint16_t)col * SD_LCD_SCALE + sx,
                                              y + (uint16_t)row * SD_LCD_SCALE + sy);
                    }
                }
            }
        }
    }
}

static const uint8_t *sd_batch_test_lcd_glyph(char ch)
{
    uint16_t index;

    if (ch >= 'a' && ch <= 'z')
    {
        ch = (char)(ch - 'a' + 'A');
    }

    for (index = 0U; index < (sizeof(g_sd_batch_lcd_font) / sizeof(g_sd_batch_lcd_font[0])); index++)
    {
        if (g_sd_batch_lcd_font[index].ch == ch)
        {
            return g_sd_batch_lcd_font[index].rows;
        }
    }

    return g_sd_batch_lcd_font[0].rows;
}

static void sd_batch_test_serial_summary(const char *name, const SdBatchStats *stats)
{
    uint32_t acc_x100 = 0;
    uint32_t avg_cycles = 0;
    uint32_t avg_us = 0;

    if (stats->total > 0)
    {
        acc_x100 = ((uint32_t)stats->correct * 10000UL + stats->total / 2U) / stats->total;
        avg_cycles = stats->infer_cycles / stats->total;
        avg_us = (avg_cycles + (SystemCoreClock / 2000000UL)) / (SystemCoreClock / 1000000UL);
    }

    sd_batch_test_serial_puts("[SDTEST] ");
    sd_batch_test_serial_puts(name);
    sd_batch_test_serial_puts(" total=");
    sd_batch_test_serial_u32(stats->total);
    sd_batch_test_serial_puts(" correct=");
    sd_batch_test_serial_u32(stats->correct);
    sd_batch_test_serial_puts(" acc=");
    sd_batch_test_serial_u32(acc_x100 / 100UL);
    sd_batch_test_serial_puts(".");
    if ((acc_x100 % 100UL) < 10UL)
    {
        sd_batch_test_serial_puts("0");
    }
    sd_batch_test_serial_u32(acc_x100 % 100UL);
    sd_batch_test_serial_puts("% avg_us=");
    sd_batch_test_serial_u32(avg_us);
    sd_batch_test_serial_puts("\r\n");
}

static void sd_batch_test_serial_puts(const char *text)
{
    while (*text != '\0')
    {
        USART_SendData(DEBUG_USARTx, (uint8_t)*text);
        while (USART_GetFlagStatus(DEBUG_USARTx, USART_FLAG_TXE) == RESET)
        {
        }
        text++;
    }
}

static void sd_batch_test_serial_u32(uint32_t value)
{
    char tmp[10];
    uint8_t len = 0;

    if (value == 0U)
    {
        sd_batch_test_serial_puts("0");
        return;
    }

    while (value > 0U)
    {
        tmp[len++] = (char)('0' + (value % 10U));
        value /= 10U;
    }

    while (len > 0U)
    {
        char out[2];
        out[0] = tmp[--len];
        out[1] = '\0';
        sd_batch_test_serial_puts(out);
    }
}
#endif

