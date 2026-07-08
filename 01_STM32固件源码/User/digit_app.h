#ifndef DIGIT_APP_H
#define DIGIT_APP_H

#include "stm32f10x.h"
#include "./lcd/bsp_xpt2046_lcd.h"

typedef enum
{
    DIGIT_MODEL_FNN = 0,
    DIGIT_MODEL_CNN = 1
} DigitModelType;

typedef enum
{
    DIGIT_INPUT_SINGLE = 0,
    DIGIT_INPUT_STRING = 1,
    DIGIT_INPUT_TRANSLATE = 2,
    DIGIT_INPUT_PC_CNN = 3
} DigitInputMode;

void digit_app_init(void);
void digit_app_touch_down(strType_XPT2046_Coordinate *touch);
void digit_app_touch_up(strType_XPT2046_Coordinate *touch);
void digit_app_tick_1ms(void);
void digit_app_process(void);
void digit_app_draw_ascii(uint16_t x, uint16_t y, const char *text, uint16_t color, uint8_t scale);
void digit_app_set_model(DigitModelType model);
DigitModelType digit_app_get_model(void);
void digit_app_set_input_mode(DigitInputMode mode);
DigitInputMode digit_app_get_input_mode(void);
void digit_app_show_translation(const char *word, const char *translation);

#endif

