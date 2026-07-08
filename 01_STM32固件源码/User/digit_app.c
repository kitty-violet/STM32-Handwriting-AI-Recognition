#include "digit_app.h"

#include "FNN_Data.h"
#include "emnist_cnn.h"
#include "fnn.h"
#include "wifi_report.h"
#include "./lcd/bsp_ili9341_lcd.h"
#include "./font/fonts.h"
#include "stm32f10x.h"

#include <stdio.h>
#include <string.h>

/* Course project: 鍩轰簬绁炵粡缃戠粶鐨勬墜鍐欐暟瀛楄瘑鍒郴缁? * Author: 閭濇稕
 * Class: 鑷姩鍖?408
 */
#define DIGIT_AREA_X 10
#define DIGIT_AREA_Y 30
#define DIGIT_GRID_SIZE 28
#define DIGIT_CAPTURE_GRID_SIZE 56
#define DIGIT_AREA_SIZE 200
#define DIGIT_NORMAL_SIZE 20
#define DIGIT_RESULT_X 232
#define DIGIT_RESULT_Y 36
#define DIGIT_RESULT_W 86
#define DIGIT_RESULT_H 76
#define DIGIT_RESULT_CLEAR_H 136
#define DIGIT_PREVIEW_CELL_SIZE 2
#define DIGIT_PREVIEW_X (DIGIT_RESULT_X + 15)
#define DIGIT_PREVIEW_Y (DIGIT_RESULT_Y + 10)
#define DIGIT_PREVIEW_PAD 5
#define DIGIT_STROKE_HISTORY_MAX 256
#define DIGIT_STROKE_BREAK 0xFFFFU
#define DIGIT_RESULT_TEXT_Y (DIGIT_RESULT_Y + DIGIT_RESULT_H + 10)
#define DIGIT_WAIT_X 234
#define DIGIT_WAIT_Y 176
#define DIGIT_WAIT_W 78
#define DIGIT_WAIT_H 16
#define DIGIT_IDLE_RECOGNIZE_MS 350
#define DIGIT_STRING_IDLE_RECOGNIZE_MS 1800
#define DIGIT_STRING_MAX 26U
#define DIGIT_STRING_LINE_CHARS 13U
#define DIGIT_STRING_MAX_SEGMENTS 8U
#define DIGIT_STRING_MIN_SEGMENT_W 2U
#define DIGIT_SEND_X 232
#define DIGIT_SEND_Y 210
#define DIGIT_SEND_W 86
#define DIGIT_SEND_H 18
#define DIGIT_DEL_X 232
#define DIGIT_DEL_Y 188
#define DIGIT_DEL_W 86
#define DIGIT_DEL_H 18
#define DIGIT_TRANSLATION_Y 156
#define DIGIT_TRANSLATION_LINE_CHARS 13U
#define DIGIT_PC_CNN_TIMEOUT_MS 10000U

static uint8_t raw_image[DIGIT_CAPTURE_GRID_SIZE * DIGIT_CAPTURE_GRID_SIZE];
static uint8_t model_image[DIGIT_GRID_SIZE * DIGIT_GRID_SIZE];
static uint8_t model_tmp_image[DIGIT_GRID_SIZE * DIGIT_GRID_SIZE];
static uint8_t has_stroke = 0;
static uint8_t min_grid_x = DIGIT_CAPTURE_GRID_SIZE;
static uint8_t min_grid_y = DIGIT_CAPTURE_GRID_SIZE;
static uint8_t max_grid_x = 0;
static uint8_t max_grid_y = 0;
static uint8_t pen_is_down = 0;
static DigitModelType current_model = DIGIT_MODEL_FNN;
static DigitInputMode current_input_mode = DIGIT_INPUT_SINGLE;
static volatile uint8_t pending_recognition = 0;
static volatile uint8_t recognition_ready = 0;
static volatile uint16_t idle_countdown_ms = 0;
static uint8_t need_clear_pad_on_next_touch = 0;
static uint32_t last_infer_us = 0;
static uint16_t stroke_x[DIGIT_STROKE_HISTORY_MAX];
static uint16_t stroke_y[DIGIT_STROKE_HISTORY_MAX];
static uint16_t stroke_count = 0;
static char string_result[DIGIT_STRING_MAX + 1U];
static uint8_t string_length = 0;
static char translate_text[DIGIT_STRING_MAX + 1U];
static uint16_t translate_zh_codes[9];
static volatile uint8_t pc_cnn_waiting = 0;
static volatile uint16_t pc_cnn_wait_ms = 0;

typedef struct
{
    uint16_t code;
    uint16_t rows[12];
} DigitAppGlyph12;

static const DigitAppGlyph12 digit_app_chinese12[] = {
    {0x795E, {0x218, 0x07E, 0xFDA, 0x2D2, 0x6FE, 0x752, 0xFDA, 0x6FE, 0x652, 0x610, 0x618, 0x000}},
    {0x7ECF, {0x000, 0x27C, 0x67C, 0x588, 0xD38, 0xF66, 0x6C2, 0xD7E, 0xF10, 0x010, 0xF10, 0x8FE}},
    {0x7F51, {0xFFE, 0x802, 0xECE, 0xAEA, 0x9BA, 0x992, 0x9BA, 0xBEA, 0xE6A, 0x802, 0x80E, 0x000}},
    {0x7EDC, {0x660, 0x47E, 0xD64, 0xBB8, 0xE18, 0x67E, 0xDFE, 0xF7C, 0x044, 0xF7C, 0x87C, 0x040}},
    {0x624B, {0x1FC, 0x7E0, 0x040, 0x7FC, 0x060, 0x040, 0xFFF, 0x040, 0x040, 0x040, 0x0C0, 0x000}},
    {0x5199, {0xFFE, 0x802, 0xB02, 0x3F8, 0x300, 0x300, 0x3FC, 0x004, 0xFF4, 0x004, 0x01C, 0x000}},
    {0x6570, {0xFD0, 0x790, 0xFFE, 0x7A4, 0xFE4, 0x274, 0xFD4, 0x49C, 0x798, 0x3FC, 0xE66, 0x000}},
    {0x5B57, {0x0C0, 0xFFE, 0xC02, 0x3F8, 0x038, 0x060, 0xFFE, 0xFFE, 0x040, 0x040, 0x1C0, 0x000}},
    {0x8BC6, {0x6FC, 0x2C4, 0x0C4, 0xCC4, 0x4C4, 0x4FC, 0x400, 0x748, 0x74C, 0x686, 0x582, 0x000}},
    {0x522B, {0xFC2, 0xCD2, 0xCD2, 0xFD2, 0x312, 0xB12, 0xFD2, 0x252, 0x252, 0x442, 0xCCE, 0x000}},
    {0x7CFB, {0x7FC, 0x0C0, 0x198, 0x3F0, 0x0C8, 0x38C, 0x7FE, 0x068, 0x66C, 0xC66, 0x8C0, 0x000}},
    {0x7EDF, {0x610, 0x4FE, 0xD30, 0xF20, 0x644, 0x4FE, 0xF28, 0xC28, 0x168, 0xF4A, 0x8CE, 0x000}},
    {0x4F5C, {0x260, 0x67E, 0x6FE, 0xEA0, 0xEBE, 0xE3C, 0x620, 0x63E, 0x63E, 0x620, 0x620, 0x000}},
    {0x8005, {0x0C0, 0x7F4, 0x0C8, 0xFFE, 0xFFE, 0x0C0, 0x3FC, 0xFFC, 0x3FC, 0x3FC, 0x3FC, 0x000}},
    {0x909D, {0x11E, 0x116, 0xFD4, 0xC14, 0xC14, 0xC12, 0xC12, 0xC16, 0xC14, 0x810, 0x810, 0x000}},
    {0x6D9B, {0xC20, 0x7FE, 0x060, 0x9FE, 0xC60, 0x1FE, 0x6CC, 0x4FE, 0xDCC, 0xB2C, 0x818, 0x000}},
    {0x73ED, {0x020, 0xF3E, 0x42C, 0x4AC, 0x6AC, 0xFFE, 0x46C, 0x44C, 0x74C, 0xECC, 0x99E, 0x000}},
    {0x7EA7, {0x6FC, 0x46C, 0xD4C, 0xF48, 0xE6E, 0x466, 0xF74, 0xC5C, 0x798, 0xD9C, 0x166, 0x000}},
    {0x81EA, {0x080, 0x7FC, 0x604, 0x404, 0x7FC, 0x404, 0x7FC, 0x604, 0x404, 0x7FC, 0x404, 0x000}},
    {0x52A8, {0x018, 0xF98, 0x07E, 0x01A, 0xF92, 0x612, 0x592, 0xC92, 0xFF2, 0x862, 0x04E, 0x000}},
    {0x5316, {0x320, 0x222, 0x626, 0x62C, 0xE38, 0x270, 0x2E0, 0x3A0, 0x222, 0x222, 0x23E, 0x000}},
    {0x4F60, {0x140, 0x140, 0x27E, 0x242, 0x694, 0xA10, 0x254, 0x252, 0x252, 0x292, 0x210, 0x230}},
    {0x597D, {0x200, 0x27C, 0x204, 0xF88, 0x490, 0x490, 0x4FE, 0x910, 0x510, 0x210, 0x510, 0x8B0}},
    {0x9605, {0x9FE, 0x402, 0x112, 0x8A2, 0xBFA, 0xA0A, 0xA0A, 0xBFA, 0x8A2, 0x92A, 0xA1A, 0x806}},
    {0x8BFB, {0x820, 0x4FC, 0x020, 0x1FE, 0xC02, 0x450, 0x530, 0x490, 0x5FE, 0x628, 0x444, 0x182}},
    {0x4E66, {0x088, 0x084, 0x7F8, 0x088, 0x088, 0x088, 0xFFE, 0x082, 0x082, 0x082, 0x08C, 0x080}},
    {0x8DD1, {0x020, 0xF20, 0x97E, 0x982, 0xF7A, 0x24A, 0x24A, 0xB7A, 0xA46, 0xA40, 0xB42, 0xC3E}},
    {0x6B65, {0x040, 0x240, 0x27C, 0x240, 0x240, 0xFFE, 0x000, 0x144, 0x248, 0x470, 0x0C0, 0x700}}
};

static const uint16_t digit_app_title_zh[] = {0x795E, 0x7ECF, 0x7F51, 0x7EDC, 0x624B, 0x5199, 0x6570, 0x5B57, 0x8BC6, 0x522B, 0};
static const uint16_t digit_app_author_label_zh[] = {0x4F5C, 0x8005, 0};
static const uint16_t digit_app_author_name_zh[] = {0x909D, 0x6D9B, 0};
static const uint16_t digit_app_class_label_zh[] = {0x73ED, 0x7EA7, 0};
static const uint16_t digit_app_auto_zh[] = {0x81EA, 0x52A8, 0x5316, 0};

static void digit_app_draw_frame(void);
static void digit_app_clear_pad(void);
static void digit_app_clear_result(void);
static void digit_app_draw_waiting_mark(void);
static void digit_app_clear_waiting_mark(void);
static void digit_app_run_recognition(void);
static void digit_app_run_string_recognition(void);
static void digit_app_run_pc_cnn_upload(void);
static char digit_app_infer_current_image(void);
static const char *digit_app_model_name(void);
static void digit_app_clear_image(void);
static void digit_app_prepare_model_image(void);
static void digit_app_prepare_segment_image(uint8_t x0, uint8_t x1, uint8_t y0, uint8_t y1);
static void digit_app_center_model_image(void);
static void digit_app_add_point(int16_t x, int16_t y);
static void digit_app_add_line(int16_t x0, int16_t y0, int16_t x1, int16_t y1);
static void digit_app_draw_thick_line(int16_t x0, int16_t y0, int16_t x1, int16_t y1);
static void digit_app_record_stroke_point(uint16_t x, uint16_t y);
static void digit_app_record_stroke_break(void);
static void digit_app_draw_preview_strokes(uint16_t color);
static uint16_t digit_app_preview_x(uint16_t x);
static uint16_t digit_app_preview_y(uint16_t y);
static uint8_t digit_app_is_inside(int16_t x, int16_t y);
static int32_t digit_app_abs(int32_t value);
static void digit_app_draw_result_char(char ch, uint16_t color);
static void digit_app_append_result_char(char ch);
static void digit_app_draw_string_result(void);
static void digit_app_run_translate_letter_recognition(void);
static void digit_app_draw_translate_result(void);
static void digit_app_draw_send_button(uint16_t color);
static void digit_app_draw_del_button(uint16_t color);
static uint8_t digit_app_is_send_button(int16_t x, int16_t y);
static uint8_t digit_app_is_del_button(int16_t x, int16_t y);
static void digit_app_send_translate_word(void);
static void digit_app_delete_translate_char(void);
static void digit_app_clear_string_result(void);
static void digit_app_draw_metadata(void);
static void digit_app_draw_text5x7(uint16_t x, uint16_t y, const char *text, uint16_t color, uint8_t scale);
static void digit_app_draw_text12(uint16_t x, uint16_t y, const uint16_t *text, uint16_t color);
static void digit_app_draw_text16_gbk(uint16_t x, uint16_t y, const uint16_t *text, uint16_t color);
static void digit_app_copy_ascii_translation(const char *translation);
static uint8_t digit_app_parse_zh_code(const char *text);
static uint8_t digit_app_hex_value(char ch);
static void digit_app_cycles_init(void);
static uint32_t digit_app_cycles_now(void);
static uint8_t digit_app_translate_button_ready(void);
static void digit_app_translate_button_arm(void);

#define DIGIT_TRANSLATE_BUTTON_COOLDOWN_MS 350U
static volatile uint16_t translate_button_cooldown_ms = 0U;

void digit_app_init(void)
{
    digit_app_clear_image();
    digit_app_draw_frame();
}

void digit_app_touch_down(strType_XPT2046_Coordinate *touch)
{
    if (current_input_mode == DIGIT_INPUT_TRANSLATE &&
        digit_app_is_send_button(touch->x, touch->y))
    {
        if (!digit_app_translate_button_ready())
        {
            return;
        }
        digit_app_translate_button_arm();
        pending_recognition = 0;
        recognition_ready = 0;
        idle_countdown_ms = 0;
        digit_app_send_translate_word();
        return;
    }

    if (current_input_mode == DIGIT_INPUT_TRANSLATE &&
        digit_app_is_del_button(touch->x, touch->y))
    {
        if (!digit_app_translate_button_ready())
        {
            return;
        }
        digit_app_translate_button_arm();
        pending_recognition = 0;
        recognition_ready = 0;
        idle_countdown_ms = 0;
        digit_app_delete_translate_char();
        return;
    }

    if (!digit_app_is_inside(touch->x, touch->y))
    {
        return;
    }

    if (need_clear_pad_on_next_touch)
    {
        digit_app_clear_pad();
        digit_app_clear_image();
        need_clear_pad_on_next_touch = 0;
    }

    pending_recognition = 0;
    recognition_ready = 0;
    idle_countdown_ms = 0;
    pen_is_down = 1;
    digit_app_clear_waiting_mark();

    if (digit_app_is_inside(touch->pre_x, touch->pre_y))
    {
        digit_app_add_line(touch->pre_x, touch->pre_y, touch->x, touch->y);
        digit_app_draw_thick_line(touch->pre_x, touch->pre_y, touch->x, touch->y);
    }
    else
    {
        digit_app_add_point(touch->x, touch->y);
        LCD_SetColors(GREEN, BLACK);
        ILI9341_DrawCircle(touch->x, touch->y, 3, 1);
    }

    digit_app_record_stroke_point(touch->x, touch->y);
    has_stroke = 1;
}

void digit_app_touch_up(strType_XPT2046_Coordinate *touch)
{
    if (!pen_is_down)
    {
        return;
    }

    pen_is_down = 0;

    if (!has_stroke)
    {
        return;
    }

    pending_recognition = 1;
    recognition_ready = 0;
    idle_countdown_ms = (current_input_mode == DIGIT_INPUT_STRING ||
                         current_input_mode == DIGIT_INPUT_PC_CNN) ?
                        DIGIT_STRING_IDLE_RECOGNIZE_MS :
                        DIGIT_IDLE_RECOGNIZE_MS;
    digit_app_draw_waiting_mark();
    digit_app_record_stroke_break();
    (void)touch;
}

void digit_app_tick_1ms(void)
{
    if (translate_button_cooldown_ms > 0U)
    {
        translate_button_cooldown_ms--;
    }

    if (pending_recognition && idle_countdown_ms > 0)
    {
        idle_countdown_ms--;
        if (idle_countdown_ms == 0)
        {
            pending_recognition = 0;
            recognition_ready = 1;
        }
    }

    if (pc_cnn_waiting && pc_cnn_wait_ms > 0U)
    {
        pc_cnn_wait_ms--;
    }
}

void digit_app_process(void)
{
    if (recognition_ready)
    {
        recognition_ready = 0;
        digit_app_run_recognition();
    }

    if (pc_cnn_waiting && pc_cnn_wait_ms == 0U)
    {
        pc_cnn_waiting = 0U;
        digit_app_clear_result();
        LCD_SetColors(WHITE, BLACK);
        ILI9341_DrawRectangle(DIGIT_RESULT_X, DIGIT_RESULT_Y, DIGIT_RESULT_W, DIGIT_RESULT_H, 0);
        digit_app_draw_text5x7(DIGIT_RESULT_X, DIGIT_RESULT_TEXT_Y, "PC-CNN", YELLOW, 1);
        digit_app_draw_text5x7(DIGIT_RESULT_X, (uint16_t)(DIGIT_RESULT_TEXT_Y + 14U), "TIMEOUT", RED, 1);
        printf("[APP] PC-CNN result timeout\r\n");
    }
}

void digit_app_draw_ascii(uint16_t x, uint16_t y, const char *text, uint16_t color, uint8_t scale)
{
    digit_app_draw_text5x7(x, y, text, color, scale);
}

void digit_app_set_model(DigitModelType model)
{
    current_model = model;
}

DigitModelType digit_app_get_model(void)
{
    return current_model;
}

void digit_app_set_input_mode(DigitInputMode mode)
{
    current_input_mode = mode;
    pc_cnn_waiting = 0U;
    pc_cnn_wait_ms = 0U;
    digit_app_clear_string_result();
}

DigitInputMode digit_app_get_input_mode(void)
{
    return current_input_mode;
}

static void digit_app_run_recognition(void)
{
    char label;
    char text[2];

    if (current_input_mode == DIGIT_INPUT_STRING)
    {
        digit_app_run_string_recognition();
        return;
    }
    if (current_input_mode == DIGIT_INPUT_PC_CNN)
    {
        digit_app_run_pc_cnn_upload();
        return;
    }
    if (current_input_mode == DIGIT_INPUT_TRANSLATE)
    {
        digit_app_run_translate_letter_recognition();
        return;
    }

    if (!has_stroke)
    {
        return;
    }

    digit_app_prepare_model_image();

    digit_app_clear_result();
    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_RESULT_X, DIGIT_RESULT_Y, DIGIT_RESULT_W, DIGIT_RESULT_H, 0);

    label = digit_app_infer_current_image();

    digit_app_clear_waiting_mark();
    digit_app_draw_preview_strokes(GREEN);
    digit_app_draw_result_char(label, YELLOW);
    text[0] = label;
    text[1] = '\0';
    wifi_report_send_result("single", digit_app_model_name(), text, last_infer_us);
    digit_app_clear_image();
    need_clear_pad_on_next_touch = 1;
}

static char digit_app_infer_current_image(void)
{
    float fnn_logits[FNN_OUTPUT_SIZE];
    float cnn_logits[EMNIST_CNN_OUTPUT_SIZE];
    uint8_t result = 0;
    char label = '?';
    uint32_t start_cycles;
    uint32_t elapsed_cycles;
    uint32_t infer_us;

    if (current_model == DIGIT_MODEL_CNN)
    {
        digit_app_cycles_init();
        start_cycles = digit_app_cycles_now();
        result = emnist_cnn_predict(model_image, cnn_logits);
        elapsed_cycles = digit_app_cycles_now() - start_cycles;
        infer_us = (elapsed_cycles + (SystemCoreClock / 2000000UL)) / (SystemCoreClock / 1000000UL);
        last_infer_us = infer_us;
        label = emnist_cnn_label(result);

        printf("\r\nCNN predict: class=%d label=%c cycles=%u us=%u\r\n",
               result,
               label,
               elapsed_cycles,
               infer_us);
    }
    else
    {
        digit_app_cycles_init();
        start_cycles = digit_app_cycles_now();
        result = fnn_predict(model_image, fnn_logits);
        elapsed_cycles = digit_app_cycles_now() - start_cycles;
        infer_us = (elapsed_cycles + (SystemCoreClock / 2000000UL)) / (SystemCoreClock / 1000000UL);
        last_infer_us = infer_us;
        label = (char)('0' + result);

        printf("\r\nFNN digit predict: %d cycles=%u us=%u\r\n",
               result,
               elapsed_cycles,
               infer_us);
    }

    return label;
}

static const char *digit_app_model_name(void)
{
    return (current_model == DIGIT_MODEL_CNN) ? "CNN" : "FNN";
}

static void digit_app_run_string_recognition(void)
{
    uint8_t segment_start[DIGIT_STRING_MAX_SEGMENTS];
    uint8_t segment_end[DIGIT_STRING_MAX_SEGMENTS];
    uint8_t segment_count = 0U;
    uint8_t in_segment = 0U;
    uint8_t start_x = 0U;
    uint8_t x;
    uint8_t y;

    if (!has_stroke)
    {
        return;
    }

    digit_app_clear_string_result();

    for (x = 0U; x < DIGIT_CAPTURE_GRID_SIZE; x++)
    {
        uint8_t has_column = 0U;

        for (y = 0U; y < DIGIT_CAPTURE_GRID_SIZE; y++)
        {
            if (raw_image[y * DIGIT_CAPTURE_GRID_SIZE + x] > 0U)
            {
                has_column = 1U;
                break;
            }
        }

        if (has_column && !in_segment)
        {
            start_x = x;
            in_segment = 1U;
        }
        else if (!has_column && in_segment)
        {
            uint8_t end_x = (uint8_t)(x - 1U);
            if ((uint8_t)(end_x - start_x + 1U) >= DIGIT_STRING_MIN_SEGMENT_W &&
                segment_count < DIGIT_STRING_MAX_SEGMENTS)
            {
                segment_start[segment_count] = start_x;
                segment_end[segment_count] = end_x;
                segment_count++;
            }
            in_segment = 0U;
        }
    }

    if (in_segment && segment_count < DIGIT_STRING_MAX_SEGMENTS)
    {
        segment_start[segment_count] = start_x;
        segment_end[segment_count] = (uint8_t)(DIGIT_CAPTURE_GRID_SIZE - 1U);
        segment_count++;
    }

    digit_app_clear_result();
    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_RESULT_X, DIGIT_RESULT_Y, DIGIT_RESULT_W, DIGIT_RESULT_H, 0);

    for (x = 0U; x < segment_count; x++)
    {
        uint8_t min_y = DIGIT_CAPTURE_GRID_SIZE;
        uint8_t max_y = 0U;
        uint8_t sx;

        for (sx = segment_start[x]; sx <= segment_end[x]; sx++)
        {
            for (y = 0U; y < DIGIT_CAPTURE_GRID_SIZE; y++)
            {
                if (raw_image[y * DIGIT_CAPTURE_GRID_SIZE + sx] > 0U)
                {
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }

        if (min_y <= max_y)
        {
            char label;
            digit_app_prepare_segment_image(segment_start[x], segment_end[x], min_y, max_y);
            label = digit_app_infer_current_image();
            digit_app_append_result_char(label);
        }
    }

    if (segment_count == 0U)
    {
        digit_app_prepare_model_image();
        digit_app_append_result_char(digit_app_infer_current_image());
    }

    digit_app_clear_waiting_mark();
    digit_app_draw_preview_strokes(GREEN);
    digit_app_draw_string_result();
    printf("[APP] String segments=%d result: %s\r\n", segment_count, string_result);
    wifi_report_send_result("string", digit_app_model_name(), string_result, 0);
    digit_app_clear_image();
    need_clear_pad_on_next_touch = 1;
}

static void digit_app_run_pc_cnn_upload(void)
{
    if (!has_stroke)
    {
        return;
    }

    digit_app_clear_string_result();
    digit_app_clear_result();
    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_RESULT_X, DIGIT_RESULT_Y, DIGIT_RESULT_W, DIGIT_RESULT_H, 0);
    digit_app_draw_text5x7(DIGIT_RESULT_X, DIGIT_RESULT_TEXT_Y, "PC-CNN", YELLOW, 1);
    digit_app_draw_text5x7(DIGIT_RESULT_X, (uint16_t)(DIGIT_RESULT_TEXT_Y + 14U), "SENDING", GREEN, 1);

    digit_app_clear_waiting_mark();
    digit_app_draw_preview_strokes(GREEN);
    printf("[APP] PC-CNN upload 56x56 bitmap\r\n");
    pc_cnn_waiting = 1U;
    pc_cnn_wait_ms = DIGIT_PC_CNN_TIMEOUT_MS;
    wifi_report_send_pc_cnn_bitmap(raw_image, DIGIT_CAPTURE_GRID_SIZE, DIGIT_CAPTURE_GRID_SIZE);
    digit_app_clear_image();
    need_clear_pad_on_next_touch = 1;
}

static void digit_app_run_translate_letter_recognition(void)
{
    char label;

    if (!has_stroke)
    {
        return;
    }

    if (current_model != DIGIT_MODEL_CNN)
    {
        digit_app_clear_result();
        digit_app_draw_send_button(CYAN);
        digit_app_draw_del_button(CYAN);
        digit_app_draw_text5x7(DIGIT_RESULT_X, DIGIT_RESULT_TEXT_Y, "USE CNN", RED, 1);
        digit_app_clear_waiting_mark();
        digit_app_clear_image();
        need_clear_pad_on_next_touch = 1;
        printf("[APP] Translate mode requires CNN model\r\n");
        return;
    }

    digit_app_prepare_model_image();
    label = digit_app_infer_current_image();
    digit_app_append_result_char(label);
    translate_text[0] = '\0';

    digit_app_clear_result();
    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_RESULT_X, DIGIT_RESULT_Y, DIGIT_RESULT_W, DIGIT_RESULT_H, 0);
    digit_app_clear_waiting_mark();
    digit_app_draw_preview_strokes(GREEN);
    digit_app_draw_send_button(CYAN);
    digit_app_draw_del_button(CYAN);
    digit_app_draw_translate_result();

    printf("[APP] Translate letter=%c word=%s\r\n", label, string_result);
    digit_app_clear_image();
    need_clear_pad_on_next_touch = 1;
}

static void digit_app_cycles_init(void)
{
    volatile uint32_t *demcr = (uint32_t *)0xE000EDFCUL;
    volatile uint32_t *dwt_ctrl = (uint32_t *)0xE0001000UL;
    volatile uint32_t *dwt_cyccnt = (uint32_t *)0xE0001004UL;

    *demcr |= (1UL << 24);
    *dwt_cyccnt = 0;
    *dwt_ctrl |= 1UL;
}

static uint32_t digit_app_cycles_now(void)
{
    volatile uint32_t *dwt_cyccnt = (uint32_t *)0xE0001004UL;
    return *dwt_cyccnt;
}

static uint8_t digit_app_translate_button_ready(void)
{
    return (translate_button_cooldown_ms == 0U);
}

static void digit_app_translate_button_arm(void)
{
    translate_button_cooldown_ms = DIGIT_TRANSLATE_BUTTON_COOLDOWN_MS;
}

static void digit_app_draw_frame(void)
{
    LCD_SetBackColor(BLACK);
    ILI9341_Clear(0, 0, LCD_X_LENGTH, LCD_Y_LENGTH);

    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_AREA_X, 6, DIGIT_AREA_SIZE, 20, 0);
    ILI9341_DrawRectangle(DIGIT_AREA_X + 4, 9, DIGIT_AREA_SIZE - 8, 14, 1);
    digit_app_clear_waiting_mark();
    digit_app_clear_result();

    digit_app_clear_pad();
    if (current_input_mode != DIGIT_INPUT_TRANSLATE &&
        current_input_mode != DIGIT_INPUT_PC_CNN)
    {
        digit_app_draw_metadata();
    }
    if (current_input_mode == DIGIT_INPUT_TRANSLATE)
    {
        digit_app_draw_send_button(CYAN);
        digit_app_draw_del_button(CYAN);
        digit_app_draw_translate_result();
    }
    if (current_input_mode == DIGIT_INPUT_PC_CNN)
    {
        digit_app_draw_translate_result();
    }
}

static void digit_app_clear_pad(void)
{
    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_AREA_X + 1, DIGIT_AREA_Y + 1, DIGIT_AREA_SIZE - 2, DIGIT_AREA_SIZE - 2, 1);
    ILI9341_DrawRectangle(DIGIT_AREA_X, DIGIT_AREA_Y, DIGIT_AREA_SIZE, DIGIT_AREA_SIZE, 0);
    LCD_SetColors(GREY, BLACK);
    ILI9341_DrawLine(DIGIT_AREA_X + DIGIT_AREA_SIZE / 2, DIGIT_AREA_Y,
                     DIGIT_AREA_X + DIGIT_AREA_SIZE / 2, DIGIT_AREA_Y + DIGIT_AREA_SIZE);
    ILI9341_DrawLine(DIGIT_AREA_X, DIGIT_AREA_Y + DIGIT_AREA_SIZE / 2,
                     DIGIT_AREA_X + DIGIT_AREA_SIZE, DIGIT_AREA_Y + DIGIT_AREA_SIZE / 2);

    LCD_SetColors(GREEN, BLACK);
}

static void digit_app_draw_metadata(void)
{
    digit_app_draw_text12(DIGIT_AREA_X + 40, 10, digit_app_title_zh, BLACK);
    digit_app_draw_text12(DIGIT_RESULT_X, 190, digit_app_author_label_zh, CYAN);
    digit_app_draw_text5x7(DIGIT_RESULT_X + 25, 193, ":", CYAN, 1);
    digit_app_draw_text12(DIGIT_RESULT_X + 33, 190, digit_app_author_name_zh, CYAN);
    digit_app_draw_text12(DIGIT_RESULT_X, 206, digit_app_class_label_zh, CYAN);
    digit_app_draw_text5x7(DIGIT_RESULT_X + 25, 209, ":", CYAN, 1);
    digit_app_draw_text12(DIGIT_RESULT_X + 33, 206, digit_app_auto_zh, CYAN);
    digit_app_draw_text5x7(DIGIT_RESULT_X + 33, 222, "2408", CYAN, 1);
}

static void digit_app_clear_result(void)
{
    LCD_SetColors(BLACK, BLACK);
    ILI9341_DrawRectangle(DIGIT_RESULT_X, DIGIT_RESULT_Y, DIGIT_RESULT_W, LCD_Y_LENGTH - DIGIT_RESULT_Y, 1);
    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_RESULT_X, DIGIT_RESULT_Y, DIGIT_RESULT_W, DIGIT_RESULT_H, 0);
}

static void digit_app_draw_waiting_mark(void)
{
    LCD_SetColors(GREEN, BLACK);
    ILI9341_DrawRectangle(DIGIT_WAIT_X, DIGIT_WAIT_Y, DIGIT_WAIT_W, DIGIT_WAIT_H, 1);
}

static void digit_app_clear_waiting_mark(void)
{
    LCD_SetColors(BLACK, BLACK);
    ILI9341_DrawRectangle(DIGIT_WAIT_X, DIGIT_WAIT_Y, DIGIT_WAIT_W, DIGIT_WAIT_H, 1);
    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_WAIT_X, DIGIT_WAIT_Y, DIGIT_WAIT_W, DIGIT_WAIT_H, 0);
}

static void digit_app_clear_image(void)
{
    memset(raw_image, 0, sizeof(raw_image));
    memset(model_image, 0, sizeof(model_image));
    memset(model_tmp_image, 0, sizeof(model_tmp_image));
    has_stroke = 0;
    pen_is_down = 0;
    min_grid_x = DIGIT_CAPTURE_GRID_SIZE;
    min_grid_y = DIGIT_CAPTURE_GRID_SIZE;
    max_grid_x = 0;
    max_grid_y = 0;
    pending_recognition = 0;
    recognition_ready = 0;
    idle_countdown_ms = 0;
    need_clear_pad_on_next_touch = 0;
    stroke_count = 0;
}

static void digit_app_prepare_model_image(void)
{
    uint8_t width;
    uint8_t height;
    uint8_t long_side;
    uint8_t scaled_w;
    uint8_t scaled_h;
    uint8_t offset_x;
    uint8_t offset_y;
    uint8_t x;
    uint8_t y;

    memset(model_image, 0, sizeof(model_image));

    if (!has_stroke || min_grid_x > max_grid_x || min_grid_y > max_grid_y)
    {
        return;
    }

    width = (uint8_t)(max_grid_x - min_grid_x + 1);
    height = (uint8_t)(max_grid_y - min_grid_y + 1);
    long_side = width > height ? width : height;
    if (long_side == 0)
    {
        return;
    }

    scaled_w = (uint8_t)(((uint16_t)width * DIGIT_NORMAL_SIZE + long_side / 2) / long_side);
    scaled_h = (uint8_t)(((uint16_t)height * DIGIT_NORMAL_SIZE + long_side / 2) / long_side);
    if (scaled_w == 0) scaled_w = 1;
    if (scaled_h == 0) scaled_h = 1;
    if (scaled_w > DIGIT_NORMAL_SIZE) scaled_w = DIGIT_NORMAL_SIZE;
    if (scaled_h > DIGIT_NORMAL_SIZE) scaled_h = DIGIT_NORMAL_SIZE;

    offset_x = (uint8_t)((DIGIT_GRID_SIZE - scaled_w) / 2);
    offset_y = (uint8_t)((DIGIT_GRID_SIZE - scaled_h) / 2);

    for (y = min_grid_y; y <= max_grid_y; y++)
    {
        for (x = min_grid_x; x <= max_grid_x; x++)
        {
            if (raw_image[y * DIGIT_CAPTURE_GRID_SIZE + x] > 0)
            {
                uint8_t tx = (uint8_t)(offset_x + (((uint16_t)(x - min_grid_x) * scaled_w) / width));
                uint8_t ty = (uint8_t)(offset_y + (((uint16_t)(y - min_grid_y) * scaled_h) / height));

                if (tx >= DIGIT_GRID_SIZE) tx = DIGIT_GRID_SIZE - 1;
                if (ty >= DIGIT_GRID_SIZE) ty = DIGIT_GRID_SIZE - 1;

                model_image[ty * DIGIT_GRID_SIZE + tx] = 255;
            }
        }
    }

    digit_app_center_model_image();
}

static void digit_app_prepare_segment_image(uint8_t x0, uint8_t x1, uint8_t y0, uint8_t y1)
{
    uint8_t width;
    uint8_t height;
    uint8_t long_side;
    uint8_t scaled_w;
    uint8_t scaled_h;
    uint8_t offset_x;
    uint8_t offset_y;
    uint8_t x;
    uint8_t y;

    memset(model_image, 0, sizeof(model_image));

    if (x0 > x1 || y0 > y1)
    {
        return;
    }

    width = (uint8_t)(x1 - x0 + 1U);
    height = (uint8_t)(y1 - y0 + 1U);
    long_side = width > height ? width : height;
    if (long_side == 0U)
    {
        return;
    }

    scaled_w = (uint8_t)(((uint16_t)width * DIGIT_NORMAL_SIZE + long_side / 2U) / long_side);
    scaled_h = (uint8_t)(((uint16_t)height * DIGIT_NORMAL_SIZE + long_side / 2U) / long_side);
    if (scaled_w == 0U) scaled_w = 1U;
    if (scaled_h == 0U) scaled_h = 1U;
    if (scaled_w > DIGIT_NORMAL_SIZE) scaled_w = DIGIT_NORMAL_SIZE;
    if (scaled_h > DIGIT_NORMAL_SIZE) scaled_h = DIGIT_NORMAL_SIZE;

    offset_x = (uint8_t)((DIGIT_GRID_SIZE - scaled_w) / 2U);
    offset_y = (uint8_t)((DIGIT_GRID_SIZE - scaled_h) / 2U);

    for (y = y0; y <= y1; y++)
    {
        for (x = x0; x <= x1; x++)
        {
            if (raw_image[y * DIGIT_CAPTURE_GRID_SIZE + x] > 0U)
            {
                uint8_t tx = (uint8_t)(offset_x + (((uint16_t)(x - x0) * scaled_w) / width));
                uint8_t ty = (uint8_t)(offset_y + (((uint16_t)(y - y0) * scaled_h) / height));

                if (tx >= DIGIT_GRID_SIZE) tx = DIGIT_GRID_SIZE - 1U;
                if (ty >= DIGIT_GRID_SIZE) ty = DIGIT_GRID_SIZE - 1U;

                model_image[ty * DIGIT_GRID_SIZE + tx] = 255U;
            }
        }
    }

    digit_app_center_model_image();
}

static void digit_app_center_model_image(void)
{
    uint32_t sum_x = 0;
    uint32_t sum_y = 0;
    uint16_t count = 0;
    uint8_t x;
    uint8_t y;
    int8_t shift_x;
    int8_t shift_y;

    for (y = 0; y < DIGIT_GRID_SIZE; y++)
    {
        for (x = 0; x < DIGIT_GRID_SIZE; x++)
        {
            if (model_image[y * DIGIT_GRID_SIZE + x] > 0)
            {
                sum_x += x;
                sum_y += y;
                count++;
            }
        }
    }

    if (count == 0)
    {
        return;
    }

    shift_x = (int8_t)((DIGIT_GRID_SIZE / 2) - ((sum_x + count / 2) / count));
    shift_y = (int8_t)((DIGIT_GRID_SIZE / 2) - ((sum_y + count / 2) / count));

    if (shift_x == 0 && shift_y == 0)
    {
        return;
    }

    memset(model_tmp_image, 0, sizeof(model_tmp_image));
    for (y = 0; y < DIGIT_GRID_SIZE; y++)
    {
        for (x = 0; x < DIGIT_GRID_SIZE; x++)
        {
            if (model_image[y * DIGIT_GRID_SIZE + x] > 0)
            {
                int16_t nx = (int16_t)x + shift_x;
                int16_t ny = (int16_t)y + shift_y;
                if (nx >= 0 && nx < DIGIT_GRID_SIZE && ny >= 0 && ny < DIGIT_GRID_SIZE)
                {
                    model_tmp_image[ny * DIGIT_GRID_SIZE + nx] = model_image[y * DIGIT_GRID_SIZE + x];
                }
            }
        }
    }
    memcpy(model_image, model_tmp_image, sizeof(model_image));
}

static void digit_app_add_point(int16_t x, int16_t y)
{
    int16_t grid_x;
    int16_t grid_y;
    int16_t dx;
    int16_t dy;

    if (!digit_app_is_inside(x, y))
    {
        return;
    }

    grid_x = (int16_t)(((int32_t)(x - DIGIT_AREA_X) * DIGIT_CAPTURE_GRID_SIZE) / DIGIT_AREA_SIZE);
    grid_y = (int16_t)(((int32_t)(y - DIGIT_AREA_Y) * DIGIT_CAPTURE_GRID_SIZE) / DIGIT_AREA_SIZE);

    for (dy = -1; dy <= 1; dy++)
    {
        for (dx = -1; dx <= 1; dx++)
        {
            int16_t px = grid_x + dx;
            int16_t py = grid_y + dy;
            if (px >= 0 && px < DIGIT_CAPTURE_GRID_SIZE && py >= 0 && py < DIGIT_CAPTURE_GRID_SIZE)
            {
                raw_image[py * DIGIT_CAPTURE_GRID_SIZE + px] = 255;
                if ((uint8_t)px < min_grid_x) min_grid_x = (uint8_t)px;
                if ((uint8_t)py < min_grid_y) min_grid_y = (uint8_t)py;
                if ((uint8_t)px > max_grid_x) max_grid_x = (uint8_t)px;
                if ((uint8_t)py > max_grid_y) max_grid_y = (uint8_t)py;
            }
        }
    }
}

static void digit_app_add_line(int16_t x0, int16_t y0, int16_t x1, int16_t y1)
{
    int32_t dx = digit_app_abs(x1 - x0);
    int32_t dy = digit_app_abs(y1 - y0);
    int32_t sx = x0 < x1 ? 1 : -1;
    int32_t sy = y0 < y1 ? 1 : -1;
    int32_t err = dx - dy;

    while (1)
    {
        int32_t e2;
        digit_app_add_point(x0, y0);
        if (x0 == x1 && y0 == y1)
        {
            break;
        }
        e2 = 2 * err;
        if (e2 > -dy)
        {
            err -= dy;
            x0 = (int16_t)(x0 + sx);
        }
        if (e2 < dx)
        {
            err += dx;
            y0 = (int16_t)(y0 + sy);
        }
    }
}

static void digit_app_draw_thick_line(int16_t x0, int16_t y0, int16_t x1, int16_t y1)
{
    int32_t dx = digit_app_abs(x1 - x0);
    int32_t dy = digit_app_abs(y1 - y0);
    int32_t sx = x0 < x1 ? 1 : -1;
    int32_t sy = y0 < y1 ? 1 : -1;
    int32_t err = dx - dy;

    LCD_SetColors(GREEN, BLACK);
    while (1)
    {
        int32_t e2;
        ILI9341_DrawCircle((uint16_t)x0, (uint16_t)y0, 3, 1);
        if (x0 == x1 && y0 == y1)
        {
            break;
        }
        e2 = 2 * err;
        if (e2 > -dy)
        {
            err -= dy;
            x0 = (int16_t)(x0 + sx);
        }
        if (e2 < dx)
        {
            err += dx;
            y0 = (int16_t)(y0 + sy);
        }
    }
}

static void digit_app_record_stroke_point(uint16_t x, uint16_t y)
{
    if (stroke_count >= DIGIT_STROKE_HISTORY_MAX)
    {
        return;
    }

    if (stroke_count > 0 &&
        stroke_x[stroke_count - 1U] == x &&
        stroke_y[stroke_count - 1U] == y)
    {
        return;
    }

    stroke_x[stroke_count] = x;
    stroke_y[stroke_count] = y;
    stroke_count++;
}

static void digit_app_record_stroke_break(void)
{
    if (stroke_count == 0 || stroke_count >= DIGIT_STROKE_HISTORY_MAX)
    {
        return;
    }

    if (stroke_x[stroke_count - 1U] == DIGIT_STROKE_BREAK)
    {
        return;
    }

    stroke_x[stroke_count] = DIGIT_STROKE_BREAK;
    stroke_y[stroke_count] = DIGIT_STROKE_BREAK;
    stroke_count++;
}

static uint16_t digit_app_preview_x(uint16_t x)
{
    uint16_t inner = (uint16_t)(DIGIT_RESULT_H - 2U * DIGIT_PREVIEW_PAD - 1U);

    if (x < DIGIT_AREA_X)
    {
        x = DIGIT_AREA_X;
    }
    if (x >= (DIGIT_AREA_X + DIGIT_AREA_SIZE))
    {
        x = (uint16_t)(DIGIT_AREA_X + DIGIT_AREA_SIZE - 1U);
    }

    return (uint16_t)(DIGIT_RESULT_X + DIGIT_PREVIEW_PAD +
                      (((uint32_t)(x - DIGIT_AREA_X) * inner + DIGIT_AREA_SIZE / 2U) / DIGIT_AREA_SIZE));
}

static uint16_t digit_app_preview_y(uint16_t y)
{
    uint16_t inner = (uint16_t)(DIGIT_RESULT_H - 2U * DIGIT_PREVIEW_PAD - 1U);

    if (y < DIGIT_AREA_Y)
    {
        y = DIGIT_AREA_Y;
    }
    if (y >= (DIGIT_AREA_Y + DIGIT_AREA_SIZE))
    {
        y = (uint16_t)(DIGIT_AREA_Y + DIGIT_AREA_SIZE - 1U);
    }

    return (uint16_t)(DIGIT_RESULT_Y + DIGIT_PREVIEW_PAD +
                      (((uint32_t)(y - DIGIT_AREA_Y) * inner + DIGIT_AREA_SIZE / 2U) / DIGIT_AREA_SIZE));
}

static void digit_app_draw_preview_strokes(uint16_t color)
{
    uint16_t index;
    uint8_t has_previous = 0;
    uint16_t prev_x = 0;
    uint16_t prev_y = 0;

    LCD_SetColors(color, BLACK);
    for (index = 0; index < stroke_count; index++)
    {
        uint16_t x = stroke_x[index];
        uint16_t y = stroke_y[index];

        if (x == DIGIT_STROKE_BREAK)
        {
            has_previous = 0;
            continue;
        }

        x = digit_app_preview_x(x);
        y = digit_app_preview_y(y);

        if (has_previous)
        {
            ILI9341_DrawLine(prev_x, prev_y, x, y);
        }
        else
        {
            ILI9341_SetPointPixel(x, y);
        }

        prev_x = x;
        prev_y = y;
        has_previous = 1;
    }
}

static uint8_t digit_app_is_inside(int16_t x, int16_t y)
{
    return (x >= DIGIT_AREA_X &&
            x < (DIGIT_AREA_X + DIGIT_AREA_SIZE) &&
            y >= DIGIT_AREA_Y &&
            y < (DIGIT_AREA_Y + DIGIT_AREA_SIZE));
}

static int32_t digit_app_abs(int32_t value)
{
    return value >= 0 ? value : -value;
}

static void digit_app_draw_result_char(char ch, uint16_t color)
{
    char text[9] = "RESULT:";

    text[7] = ch;
    text[8] = '\0';
    digit_app_draw_text5x7(DIGIT_RESULT_X, DIGIT_RESULT_TEXT_Y, text, color, 1);
}

static void digit_app_append_result_char(char ch)
{
    if (string_length >= DIGIT_STRING_MAX)
    {
        memmove(string_result, string_result + 1, DIGIT_STRING_MAX - 1U);
        string_result[DIGIT_STRING_MAX - 1U] = ch;
        string_result[DIGIT_STRING_MAX] = '\0';
        return;
    }

    string_result[string_length] = ch;
    string_length++;
    string_result[string_length] = '\0';
}

static void digit_app_draw_string_result(void)
{
    char line[14];
    uint8_t i;
    uint8_t base;

    digit_app_draw_text5x7(DIGIT_RESULT_X, DIGIT_RESULT_TEXT_Y, "STR:", YELLOW, 1);

    for (i = 0U; i < DIGIT_STRING_LINE_CHARS; i++)
    {
        line[i] = string_result[i];
        if (line[i] == '\0')
        {
            break;
        }
    }
    line[i] = '\0';
    digit_app_draw_text5x7(DIGIT_RESULT_X, (uint16_t)(DIGIT_RESULT_TEXT_Y + 14U), line, WHITE, 1);

    base = DIGIT_STRING_LINE_CHARS;
    for (i = 0U; i < DIGIT_STRING_LINE_CHARS; i++)
    {
        line[i] = string_result[base + i];
        if (line[i] == '\0')
        {
            break;
        }
    }
    line[i] = '\0';
    digit_app_draw_text5x7(DIGIT_RESULT_X, (uint16_t)(DIGIT_RESULT_TEXT_Y + 26U), line, WHITE, 1);
}

static void digit_app_draw_translate_result(void)
{
    char line[14];
    uint8_t i;
    uint8_t base;

    digit_app_draw_text5x7(DIGIT_RESULT_X, DIGIT_RESULT_TEXT_Y, "WORD:", YELLOW, 1);

    for (i = 0U; i < DIGIT_TRANSLATION_LINE_CHARS; i++)
    {
        line[i] = string_result[i];
        if (line[i] == '\0')
        {
            break;
        }
    }
    line[i] = '\0';
    digit_app_draw_text5x7(DIGIT_RESULT_X, (uint16_t)(DIGIT_RESULT_TEXT_Y + 14U), line, WHITE, 1);

    base = DIGIT_TRANSLATION_LINE_CHARS;
    for (i = 0U; i < DIGIT_TRANSLATION_LINE_CHARS; i++)
    {
        line[i] = string_result[base + i];
        if (line[i] == '\0')
        {
            break;
        }
    }
    line[i] = '\0';
    digit_app_draw_text5x7(DIGIT_RESULT_X, (uint16_t)(DIGIT_RESULT_TEXT_Y + 26U), line, WHITE, 1);

    for (i = 0U; i < DIGIT_TRANSLATION_LINE_CHARS; i++)
    {
        line[i] = translate_text[i];
        if (line[i] == '\0')
        {
            break;
        }
    }
    line[i] = '\0';
    if (translate_zh_codes[0] != 0U)
    {
        digit_app_draw_text16_gbk(DIGIT_RESULT_X, DIGIT_TRANSLATION_Y, translate_zh_codes, GREEN);
        digit_app_draw_text5x7(DIGIT_RESULT_X, (uint16_t)(DIGIT_TRANSLATION_Y + 18U), line, GREEN, 1);
        return;
    }
    digit_app_draw_text5x7(DIGIT_RESULT_X, DIGIT_TRANSLATION_Y, line, GREEN, 1);

    base = DIGIT_TRANSLATION_LINE_CHARS;
    for (i = 0U; i < DIGIT_TRANSLATION_LINE_CHARS; i++)
    {
        line[i] = translate_text[base + i];
        if (line[i] == '\0')
        {
            break;
        }
    }
    line[i] = '\0';
    digit_app_draw_text5x7(DIGIT_RESULT_X, (uint16_t)(DIGIT_TRANSLATION_Y + 12U), line, GREEN, 1);
}

static void digit_app_draw_send_button(uint16_t color)
{
    LCD_SetColors(color, BLACK);
    ILI9341_DrawRectangle(DIGIT_SEND_X, DIGIT_SEND_Y, DIGIT_SEND_W, DIGIT_SEND_H, 0);
    digit_app_draw_text5x7((uint16_t)(DIGIT_SEND_X + 26U),
                           (uint16_t)(DIGIT_SEND_Y + 5U),
                           "SEND",
                           color,
                           1);
}

static void digit_app_draw_del_button(uint16_t color)
{
    LCD_SetColors(color, BLACK);
    ILI9341_DrawRectangle(DIGIT_DEL_X, DIGIT_DEL_Y, DIGIT_DEL_W, DIGIT_DEL_H, 0);
    digit_app_draw_text5x7((uint16_t)(DIGIT_DEL_X + 29U),
                           (uint16_t)(DIGIT_DEL_Y + 5U),
                           "DEL",
                           color,
                           1);
}

static uint8_t digit_app_is_send_button(int16_t x, int16_t y)
{
    return (x >= DIGIT_SEND_X &&
            x < (DIGIT_SEND_X + DIGIT_SEND_W) &&
            y >= DIGIT_SEND_Y &&
            y < (DIGIT_SEND_Y + DIGIT_SEND_H));
}

static uint8_t digit_app_is_del_button(int16_t x, int16_t y)
{
    return (x >= DIGIT_DEL_X &&
            x < (DIGIT_DEL_X + DIGIT_DEL_W) &&
            y >= DIGIT_DEL_Y &&
            y < (DIGIT_DEL_Y + DIGIT_DEL_H));
}

static void digit_app_send_translate_word(void)
{
    if (string_length == 0U)
    {
        strcpy(translate_text, "NO WORD");
        digit_app_clear_result();
        digit_app_draw_send_button(CYAN);
        digit_app_draw_del_button(CYAN);
        digit_app_draw_translate_result();
        printf("[APP] Translate send skipped: empty word\r\n");
        return;
    }

    strcpy(translate_text, "SENDING");
    digit_app_clear_result();
    digit_app_draw_send_button(YELLOW);
    digit_app_draw_del_button(CYAN);
    digit_app_draw_translate_result();

    printf("[APP] Translate send: %s\r\n", string_result);
    wifi_report_send_result("word", digit_app_model_name(), string_result, 0);
}

static void digit_app_delete_translate_char(void)
{
    translate_text[0] = '\0';

    if (string_length > 0U)
    {
        string_length--;
        string_result[string_length] = '\0';
    }
    else
    {
        strcpy(translate_text, "EMPTY");
    }

    digit_app_clear_result();
    LCD_SetColors(WHITE, BLACK);
    ILI9341_DrawRectangle(DIGIT_RESULT_X, DIGIT_RESULT_Y, DIGIT_RESULT_W, DIGIT_RESULT_H, 0);
    digit_app_draw_send_button(CYAN);
    digit_app_draw_del_button(CYAN);
    digit_app_draw_translate_result();
    printf("[APP] Translate delete, word=%s\r\n", string_result);
}

void digit_app_show_translation(const char *word, const char *translation)
{
    uint8_t n = 0U;
    const char *py;

    pc_cnn_waiting = 0U;
    pc_cnn_wait_ms = 0U;

    if (word != 0 && word[0] != '\0')
    {
        digit_app_clear_string_result();
        while (word[n] != '\0' && n < DIGIT_STRING_MAX)
        {
            string_result[n] = word[n];
            n++;
        }
        string_result[n] = '\0';
        string_length = n;
    }

    memset(translate_text, 0, sizeof(translate_text));
    memset(translate_zh_codes, 0, sizeof(translate_zh_codes));
    if (translation != 0)
    {
        if (strncmp(translation, "ZHCODE:", 7) == 0)
        {
            py = strstr(translation, ";PY:");
            if (py != 0)
            {
                digit_app_copy_ascii_translation(py + 4);
            }
            else
            {
                digit_app_copy_ascii_translation("");
            }

            if (!digit_app_parse_zh_code(translation + 7) &&
                translate_text[0] == '\0')
            {
                digit_app_copy_ascii_translation("NO FONT");
            }
        }
        else
        {
            digit_app_copy_ascii_translation(translation);
        }
    }

    digit_app_clear_result();
    if (current_input_mode == DIGIT_INPUT_TRANSLATE)
    {
        digit_app_draw_send_button(CYAN);
        digit_app_draw_del_button(CYAN);
    }
    digit_app_draw_translate_result();
    printf("[APP] Translation received: %s -> %s\r\n", string_result, translate_text);
}

static void digit_app_clear_string_result(void)
{
    memset(string_result, 0, sizeof(string_result));
    string_length = 0;
    translate_text[0] = '\0';
    memset(translate_zh_codes, 0, sizeof(translate_zh_codes));
}

static void digit_app_copy_ascii_translation(const char *translation)
{
    uint8_t i;

    memset(translate_text, 0, sizeof(translate_text));
    if (translation == 0)
    {
        return;
    }

    for (i = 0U; i < DIGIT_STRING_MAX && translation[i] != '\0'; i++)
    {
        char ch = translation[i];
        if (ch >= 32 && ch <= 126)
        {
            translate_text[i] = ch;
        }
        else
        {
            translate_text[i] = '?';
        }
    }
    translate_text[i] = '\0';
}

static uint8_t digit_app_parse_zh_code(const char *text)
{
    uint8_t count = 0U;

    memset(translate_zh_codes, 0, sizeof(translate_zh_codes));
    while (text != 0 && *text != '\0' && *text != ';' && count < 8U)
    {
        uint8_t i;
        uint16_t code = 0U;

        while (*text == ',' || *text == ' ')
        {
            text++;
        }

        if (*text == '\0' || *text == ';')
        {
            break;
        }

        for (i = 0U; i < 4U; i++)
        {
            uint8_t value = digit_app_hex_value(*text++);
            if (value > 15U)
            {
                memset(translate_zh_codes, 0, sizeof(translate_zh_codes));
                return 0;
            }
            code = (uint16_t)((code << 4) | value);
        }

        translate_zh_codes[count++] = code;
    }

    translate_zh_codes[count] = 0U;
    return (count > 0U);
}

static uint8_t digit_app_hex_value(char ch)
{
    if (ch >= '0' && ch <= '9')
    {
        return (uint8_t)(ch - '0');
    }
    if (ch >= 'A' && ch <= 'F')
    {
        return (uint8_t)(ch - 'A' + 10);
    }
    if (ch >= 'a' && ch <= 'f')
    {
        return (uint8_t)(ch - 'a' + 10);
    }
    return 0xFFU;
}

typedef struct
{
    char ch;
    uint8_t rows[7];
} DigitAppGlyph;

static const DigitAppGlyph digit_app_font5x7[] = {
    {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}},
    {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}},
    {'3', {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E}},
    {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}},
    {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
    {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}},
    {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
    {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}},
    {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
    {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}},
    {'C', {0x0F, 0x10, 0x10, 0x10, 0x10, 0x10, 0x0F}},
    {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}},
    {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}},
    {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}},
    {'G', {0x0F, 0x10, 0x10, 0x13, 0x11, 0x11, 0x0F}},
    {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
    {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'J', {0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E}},
    {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}},
    {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}},
    {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}},
    {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}},
    {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
    {'Q', {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}},
    {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
    {'S', {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E}},
    {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
    {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
    {'V', {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04}},
    {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A}},
    {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}},
    {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
    {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
    {'a', {0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F}},
    {'b', {0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E}},
    {'c', {0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E}},
    {'d', {0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F}},
    {'e', {0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E}},
    {'f', {0x03, 0x04, 0x04, 0x0E, 0x04, 0x04, 0x04}},
    {'g', {0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E}},
    {'h', {0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11}},
    {'i', {0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E}},
    {'j', {0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C}},
    {'k', {0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12}},
    {'l', {0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}},
    {'m', {0x00, 0x00, 0x1A, 0x15, 0x15, 0x15, 0x15}},
    {'n', {0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11}},
    {'o', {0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E}},
    {'p', {0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10}},
    {'q', {0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x01}},
    {'r', {0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10}},
    {'s', {0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E}},
    {'t', {0x04, 0x04, 0x0E, 0x04, 0x04, 0x05, 0x02}},
    {'u', {0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D}},
    {'v', {0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04}},
    {'w', {0x00, 0x00, 0x11, 0x15, 0x15, 0x15, 0x0A}},
    {'x', {0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11}},
    {'y', {0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E}},
    {'z', {0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F}},
    {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}},
    {':', {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00}}
};

static const uint8_t *digit_app_get_glyph(char ch)
{
    uint8_t index;

    for (index = 0; index < (sizeof(digit_app_font5x7) / sizeof(digit_app_font5x7[0])); index++)
    {
        if (digit_app_font5x7[index].ch == ch)
        {
            return digit_app_font5x7[index].rows;
        }
    }

    return 0;
}

static const uint16_t *digit_app_get_chinese_glyph(uint16_t code)
{
    uint8_t index;

    for (index = 0; index < (sizeof(digit_app_chinese12) / sizeof(digit_app_chinese12[0])); index++)
    {
        if (digit_app_chinese12[index].code == code)
        {
            return digit_app_chinese12[index].rows;
        }
    }

    return 0;
}

static void digit_app_draw_text12(uint16_t x, uint16_t y, const uint16_t *text, uint16_t color)
{
    uint16_t cursor_x = x;

    LCD_SetColors(color, BLACK);
    while (*text)
    {
        const uint16_t *glyph = digit_app_get_chinese_glyph(*text);
        uint8_t row;
        uint8_t col;

        if (glyph != 0)
        {
            for (row = 0; row < 12; row++)
            {
                for (col = 0; col < 12; col++)
                {
                    if (glyph[row] & (uint16_t)(1U << (11 - col)))
                    {
                        ILI9341_SetPointPixel((uint16_t)(cursor_x + col), (uint16_t)(y + row));
                    }
                }
            }
        }

        cursor_x = (uint16_t)(cursor_x + 12);
        text++;
    }
}

static void digit_app_draw_text16_gbk(uint16_t x, uint16_t y, const uint16_t *text, uint16_t color)
{
    uint16_t cursor_x = x;
    uint8_t glyph[WIDTH_CH_CHAR * HEIGHT_CH_CHAR / 8];

    LCD_SetColors(color, BLACK);
    while (*text && cursor_x < (LCD_X_LENGTH - WIDTH_CH_CHAR))
    {
        uint8_t row;
        uint8_t col;

        if (GetGBKCode(glyph, *text) == 0)
        {
            for (row = 0U; row < HEIGHT_CH_CHAR; row++)
            {
                for (col = 0U; col < WIDTH_CH_CHAR; col++)
                {
                    if (glyph[row * 2U + col / 8U] & (uint8_t)(0x80U >> (col % 8U)))
                    {
                        ILI9341_SetPointPixel((uint16_t)(cursor_x + col), (uint16_t)(y + row));
                    }
                }
            }
        }
        else
        {
            const uint16_t *fallback = digit_app_get_chinese_glyph(*text);
            if (fallback != 0)
            {
                for (row = 0U; row < 12U; row++)
                {
                    for (col = 0U; col < 12U; col++)
                    {
                        if (fallback[row] & (uint16_t)(1U << (11U - col)))
                        {
                            ILI9341_SetPointPixel((uint16_t)(cursor_x + col), (uint16_t)(y + row));
                        }
                    }
                }
            }
        }

        cursor_x = (uint16_t)(cursor_x + WIDTH_CH_CHAR);
        text++;
    }
}

static void digit_app_draw_text5x7(uint16_t x, uint16_t y, const char *text, uint16_t color, uint8_t scale)
{
    uint16_t cursor_x = x;
    uint8_t s = scale ? scale : 1;

    LCD_SetColors(color, BLACK);
    while (*text)
    {
        const uint8_t *glyph;
        uint8_t row;
        uint8_t col;

        if (*text == ' ')
        {
            cursor_x = (uint16_t)(cursor_x + 4 * s);
            text++;
            continue;
        }

        glyph = digit_app_get_glyph(*text);
        if (glyph != 0)
        {
            for (row = 0; row < 7; row++)
            {
                for (col = 0; col < 5; col++)
                {
                    if (glyph[row] & (uint8_t)(1U << (4 - col)))
                    {
                        ILI9341_DrawRectangle((uint16_t)(cursor_x + col * s),
                                              (uint16_t)(y + row * s),
                                              s,
                                              s,
                                              1);
                    }
                }
            }
        }
        cursor_x = (uint16_t)(cursor_x + 6 * s);
        text++;
    }
}

