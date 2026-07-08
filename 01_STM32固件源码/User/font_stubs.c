#include "./font/fonts.h"
#include "./FATFS/ff.h"

#include <stdio.h>
#include <string.h>

static const uint8_t dummy_font_table[1] = {0};

sFONT Font24x32 = {dummy_font_table, 24, 32};
sFONT Font16x24 = {dummy_font_table, 16, 24};
sFONT Font8x16 = {dummy_font_table, 8, 16};

#define GB2312_FONT_FILE "0:/FONT/GB2312.FON"

int GetGBKCode_from_EXFlash(uint8_t *pBuffer, uint16_t c)
{
    FIL font_file;
    FRESULT res;
    UINT br = 0;
    uint8_t high;
    uint8_t low;
    uint32_t pos;

    if (pBuffer == 0)
    {
        return -1;
    }

    memset(pBuffer, 0, WIDTH_CH_CHAR * HEIGHT_CH_CHAR / 8);
    high = (uint8_t)(c >> 8);
    low = (uint8_t)(c & 0xFFU);

    if (high < 0xA1U || low < 0xA1U)
    {
        return -1;
    }

    pos = ((uint32_t)(high - 0xA1U) * 94U + (uint32_t)(low - 0xA1U)) *
          (WIDTH_CH_CHAR * HEIGHT_CH_CHAR / 8U);

    res = f_open(&font_file, GB2312_FONT_FILE, FA_OPEN_EXISTING | FA_READ);
    if (res != FR_OK)
    {
        printf("[FONT] open failed %s res=%d\r\n", GB2312_FONT_FILE, res);
        return -1;
    }

    res = f_lseek(&font_file, pos);
    if (res == FR_OK)
    {
        res = f_read(&font_file, pBuffer, WIDTH_CH_CHAR * HEIGHT_CH_CHAR / 8U, &br);
    }
    f_close(&font_file);

    if (res != FR_OK || br != (WIDTH_CH_CHAR * HEIGHT_CH_CHAR / 8U))
    {
        printf("[FONT] read failed code=%04X res=%d br=%u\r\n", c, res, br);
        memset(pBuffer, 0, WIDTH_CH_CHAR * HEIGHT_CH_CHAR / 8);
        return -1;
    }

    return 0;
}

