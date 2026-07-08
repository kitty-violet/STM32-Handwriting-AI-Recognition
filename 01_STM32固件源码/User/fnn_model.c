#include "fnn_model.h"

#ifdef FNN_MODEL_FROM_SD

#include "FNN_Data.h"
#include "./FATFS/ff.h"

#define FNN_MODEL_HEADER_SIZE 16U

int8_t g_fnn_fc1_weight[FNN_HIDDEN_SIZE * FNN_INPUT_SIZE];
float g_fnn_fc1_bias[FNN_HIDDEN_SIZE];
int8_t g_fnn_fc2_weight[FNN_OUTPUT_SIZE * FNN_HIDDEN_SIZE];
float g_fnn_fc2_bias[FNN_OUTPUT_SIZE];
float g_fnn_fc1_scale;
float g_fnn_fc2_scale;

static uint8_t g_fnn_model_loaded = 0;

static uint32_t fnn_model_read_u32_le(const uint8_t *data)
{
    return (uint32_t)data[0] |
           ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) |
           ((uint32_t)data[3] << 24);
}

static uint8_t fnn_model_read_exact(FIL *file, void *buffer, UINT bytes)
{
    UINT read_count = 0;
    FRESULT result = f_read(file, buffer, bytes, &read_count);

    return (result == FR_OK && read_count == bytes) ? 1U : 0U;
}

uint8_t fnn_model_load(const char *path)
{
    FIL file;
    uint8_t header[FNN_MODEL_HEADER_SIZE];
    FRESULT result;

    g_fnn_model_loaded = 0;

    result = f_open(&file, path, FA_READ);
    if (result != FR_OK)
    {
        return 0;
    }

    if (!fnn_model_read_exact(&file, header, sizeof(header)))
    {
        f_close(&file);
        return 0;
    }

    if (header[0] != 'F' || header[1] != 'N' ||
        header[2] != 'N' || header[3] != '1')
    {
        f_close(&file);
        return 0;
    }

    if (fnn_model_read_u32_le(&header[4]) != FNN_INPUT_SIZE ||
        fnn_model_read_u32_le(&header[8]) != FNN_HIDDEN_SIZE ||
        fnn_model_read_u32_le(&header[12]) != FNN_OUTPUT_SIZE)
    {
        f_close(&file);
        return 0;
    }

    if (!fnn_model_read_exact(&file, g_fnn_fc1_weight, sizeof(g_fnn_fc1_weight)) ||
        !fnn_model_read_exact(&file, g_fnn_fc1_bias, sizeof(g_fnn_fc1_bias)) ||
        !fnn_model_read_exact(&file, g_fnn_fc2_weight, sizeof(g_fnn_fc2_weight)) ||
        !fnn_model_read_exact(&file, g_fnn_fc2_bias, sizeof(g_fnn_fc2_bias)) ||
        !fnn_model_read_exact(&file, &g_fnn_fc1_scale, sizeof(g_fnn_fc1_scale)) ||
        !fnn_model_read_exact(&file, &g_fnn_fc2_scale, sizeof(g_fnn_fc2_scale)))
    {
        f_close(&file);
        return 0;
    }

    f_close(&file);
    g_fnn_model_loaded = 1;
    return 1;
}

uint8_t fnn_model_is_loaded(void)
{
    return g_fnn_model_loaded;
}

#else

uint8_t fnn_model_load(const char *path)
{
    (void)path;
    return 1;
}

uint8_t fnn_model_is_loaded(void)
{
    return 1;
}

#endif

