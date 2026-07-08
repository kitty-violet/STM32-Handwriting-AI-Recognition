#include "emnist_cnn.h"

#include <string.h>

#ifdef EMNIST_CNN_MODEL_FROM_SD
#include "./FATFS/ff.h"
#endif

#define EMNIST_CNN_MODEL_HEADER_SIZE 64U
#define EMNIST_CNN_FC_CHUNK_SIZE 64U

#ifndef EMNIST_CNN_MODEL_FROM_SD
#define EMNIST_CNN_ACT_SHIFT 8
#define EMNIST_CNN_MULT_SHIFT 16
static int16_t g_cnn_pool1[EMNIST_CNN_CONV1_OUT * EMNIST_CNN_POOL1_SIZE * EMNIST_CNN_POOL1_SIZE];
static int16_t g_cnn_pool2[EMNIST_CNN_FC_INPUT_SIZE];
static uint8_t g_cnn_input_pad[30 * 30];
static int16_t g_cnn_pool1_pad[EMNIST_CNN_CONV1_OUT * 16 * 16];
static int32_t g_cnn_conv1_bias_q[EMNIST_CNN_CONV1_OUT];
static int32_t g_cnn_conv2_bias_q[EMNIST_CNN_CONV2_OUT];
static int32_t g_cnn_fc_bias_q[EMNIST_CNN_OUTPUT_SIZE];
static int32_t g_cnn_conv1_input_mult = 0;
static int32_t g_cnn_conv2_mult = 0;
static int32_t g_cnn_fc_mult = 0;
static uint8_t g_cnn_fixed_ready = 0;
#else
static float g_cnn_pool1[EMNIST_CNN_CONV1_OUT * EMNIST_CNN_POOL1_SIZE * EMNIST_CNN_POOL1_SIZE];
static float g_cnn_pool2[EMNIST_CNN_FC_INPUT_SIZE];
#endif

#ifdef EMNIST_CNN_MODEL_FROM_SD
static float g_emnist_cnn_conv1_weight[EMNIST_CNN_CONV1_OUT * 3 * 3];
static float g_emnist_cnn_conv1_bias[EMNIST_CNN_CONV1_OUT];
static float g_emnist_cnn_conv2_bias[EMNIST_CNN_CONV2_OUT];
static float g_emnist_cnn_fc_bias[EMNIST_CNN_OUTPUT_SIZE];
static char g_emnist_cnn_labels[EMNIST_CNN_OUTPUT_SIZE + 1];
static float g_cnn_conv2_weight_channel[EMNIST_CNN_CONV1_OUT * 3 * 3];
static float g_cnn_fc_weight_chunk[EMNIST_CNN_FC_CHUNK_SIZE];
static uint32_t g_cnn_conv2_weight_offset = 0;
static uint32_t g_cnn_fc_weight_offset = 0;
static uint8_t g_cnn_model_loaded = 0;
static FIL g_cnn_file;

static uint32_t emnist_cnn_read_u32_le(const uint8_t *data);
static uint8_t emnist_cnn_read_exact(FIL *file, void *buffer, UINT bytes);
static uint8_t emnist_cnn_read_at(FIL *file, uint32_t offset, void *buffer, UINT bytes);
static uint8_t emnist_cnn_load_conv2_channel(uint8_t out_ch);
#endif

#ifdef EMNIST_CNN_MODEL_FROM_SD
static float emnist_cnn_relu(float value)
{
    return value > 0.0f ? value : 0.0f;
}

static float emnist_cnn_input_at(const uint8_t image[EMNIST_CNN_INPUT_SIZE],
                                 int16_t y,
                                 int16_t x)
{
    if (x < 0 || x >= 28 || y < 0 || y >= 28)
    {
        return 0.0f;
    }

    return (float)image[(uint16_t)y * 28U + (uint16_t)x] / 255.0f;
}

static float emnist_cnn_pool1_at(uint8_t channel, int16_t y, int16_t x)
{
    if (x < 0 || x >= EMNIST_CNN_POOL1_SIZE || y < 0 || y >= EMNIST_CNN_POOL1_SIZE)
    {
        return 0.0f;
    }

    return g_cnn_pool1[((uint32_t)channel * EMNIST_CNN_POOL1_SIZE + (uint16_t)y) *
                       EMNIST_CNN_POOL1_SIZE + (uint16_t)x];
}

static void emnist_cnn_conv1_pool(const uint8_t image[EMNIST_CNN_INPUT_SIZE])
{
    uint8_t out_ch;

    for (out_ch = 0; out_ch < EMNIST_CNN_CONV1_OUT; out_ch++)
    {
        uint8_t py;
        for (py = 0; py < EMNIST_CNN_POOL1_SIZE; py++)
        {
            uint8_t px;
            for (px = 0; px < EMNIST_CNN_POOL1_SIZE; px++)
            {
                float max_value = 0.0f;
                uint8_t dy;
                for (dy = 0; dy < 2; dy++)
                {
                    uint8_t dx;
                    for (dx = 0; dx < 2; dx++)
                    {
                        int16_t oy = (int16_t)py * 2 + dy;
                        int16_t ox = (int16_t)px * 2 + dx;
                        float acc = 0.0f;
                        float sum;
                        int8_t ky;
                        for (ky = -1; ky <= 1; ky++)
                        {
                            int8_t kx;
                            for (kx = -1; kx <= 1; kx++)
                            {
                                uint16_t weight_index = ((uint16_t)out_ch * 3U + (uint8_t)(ky + 1)) * 3U +
                                                        (uint8_t)(kx + 1);
                                acc += emnist_cnn_input_at(image, (int16_t)(oy + ky), (int16_t)(ox + kx)) *
                                       (float)g_emnist_cnn_conv1_weight_q7[weight_index];
                            }
                        }
                        sum = g_emnist_cnn_conv1_bias[out_ch] + acc * g_emnist_cnn_conv1_weight_scale;
                        sum = emnist_cnn_relu(sum);
                        if (dy == 0 && dx == 0)
                        {
                            max_value = sum;
                        }
                        else if (sum > max_value)
                        {
                            max_value = sum;
                        }
                    }
                }
                g_cnn_pool1[((uint32_t)out_ch * EMNIST_CNN_POOL1_SIZE + py) *
                            EMNIST_CNN_POOL1_SIZE + px] = max_value;
            }
        }
    }
}

static float emnist_cnn_conv2_point(uint8_t out_ch, uint8_t y, uint8_t x)
{
    float acc = 0.0f;
    float sum;
    uint8_t in_ch;

    for (in_ch = 0; in_ch < EMNIST_CNN_CONV1_OUT; in_ch++)
    {
        int8_t ky;
        for (ky = -1; ky <= 1; ky++)
        {
            int8_t kx;
            for (kx = -1; kx <= 1; kx++)
            {
                uint16_t weight_index = ((uint16_t)in_ch * 3U +
                                         (uint8_t)(ky + 1)) * 3U + (uint8_t)(kx + 1);
                float weight = g_cnn_conv2_weight_channel[weight_index];
                acc += emnist_cnn_pool1_at(in_ch, (int16_t)(y + ky), (int16_t)(x + kx)) * weight;
            }
        }
    }

    sum = g_emnist_cnn_conv2_bias[out_ch] + acc * g_emnist_cnn_conv2_weight_scale;
    return emnist_cnn_relu(sum);
}

static uint8_t emnist_cnn_conv2_pool(void)
{
    uint8_t out_ch;

    for (out_ch = 0; out_ch < EMNIST_CNN_CONV2_OUT; out_ch++)
    {
        if (!emnist_cnn_load_conv2_channel(out_ch))
        {
            return 0;
        }
        {
            uint8_t py;
            for (py = 0; py < EMNIST_CNN_POOL2_SIZE; py++)
            {
                uint8_t px;
                for (px = 0; px < EMNIST_CNN_POOL2_SIZE; px++)
                {
                    float max_value = 0.0f;
                    uint8_t dy;
                    for (dy = 0; dy < 2; dy++)
                    {
                        uint8_t dx;
                        for (dx = 0; dx < 2; dx++)
                        {
                            float value = emnist_cnn_conv2_point(out_ch,
                                                                 (uint8_t)(py * 2U + dy),
                                                                 (uint8_t)(px * 2U + dx));
                            if (dy == 0 && dx == 0)
                            {
                                max_value = value;
                            }
                            else if (value > max_value)
                            {
                                max_value = value;
                            }
                        }
                    }
                    g_cnn_pool2[((uint32_t)out_ch * EMNIST_CNN_POOL2_SIZE + py) *
                                EMNIST_CNN_POOL2_SIZE + px] = max_value;
                }
            }
        }
    }

    return 1;
}

static uint8_t emnist_cnn_fc_score(uint8_t out, float *score)
{
    uint16_t index = 0;

    *score = g_emnist_cnn_fc_bias[out];
    while (index < EMNIST_CNN_FC_INPUT_SIZE)
    {
        uint16_t chunk = (uint16_t)(EMNIST_CNN_FC_INPUT_SIZE - index);
        uint16_t i;
        uint32_t offset;

        if (chunk > EMNIST_CNN_FC_CHUNK_SIZE)
        {
            chunk = EMNIST_CNN_FC_CHUNK_SIZE;
        }

        offset = g_cnn_fc_weight_offset +
                 (((uint32_t)out * EMNIST_CNN_FC_INPUT_SIZE + index) * sizeof(float));
        if (!emnist_cnn_read_at(&g_cnn_file,
                                offset,
                                g_cnn_fc_weight_chunk,
                                (UINT)(chunk * sizeof(float))))
        {
            return 0;
        }

        for (i = 0; i < chunk; i++)
        {
            *score += g_cnn_pool2[index + i] * g_cnn_fc_weight_chunk[i];
        }

        index = (uint16_t)(index + chunk);
    }

    return 1;
}
#else
static int32_t emnist_cnn_float_to_q(float value, uint8_t shift)
{
    float scaled = value * (float)(1UL << shift);

    if (scaled >= 0.0f)
    {
        return (int32_t)(scaled + 0.5f);
    }
    return (int32_t)(scaled - 0.5f);
}

static int32_t emnist_cnn_float_to_mult(float value, uint8_t shift)
{
    float scaled = value * (float)(1UL << shift);

    if (scaled >= 0.0f)
    {
        return (int32_t)(scaled + 0.5f);
    }
    return (int32_t)(scaled - 0.5f);
}

static int32_t emnist_cnn_apply_mult(int32_t value, int32_t mult)
{
    return (int32_t)(((int64_t)value * (int64_t)mult) >> EMNIST_CNN_MULT_SHIFT);
}

static int16_t emnist_cnn_relu_sat16(int32_t value)
{
    if (value <= 0)
    {
        return 0;
    }
    if (value > 32767)
    {
        return 32767;
    }
    return (int16_t)value;
}

static void emnist_cnn_prepare_input_pad(const uint8_t image[EMNIST_CNN_INPUT_SIZE])
{
    uint8_t y;

    memset(g_cnn_input_pad, 0, sizeof(g_cnn_input_pad));
    for (y = 0; y < 28; y++)
    {
        memcpy(&g_cnn_input_pad[(uint16_t)(y + 1U) * 30U + 1U],
               &image[(uint16_t)y * 28U],
               28U);
    }
}

static void emnist_cnn_prepare_pool1_pad(void)
{
    uint8_t ch;

    memset(g_cnn_pool1_pad, 0, sizeof(g_cnn_pool1_pad));
    for (ch = 0; ch < EMNIST_CNN_CONV1_OUT; ch++)
    {
        uint8_t y;
        for (y = 0; y < EMNIST_CNN_POOL1_SIZE; y++)
        {
            memcpy(&g_cnn_pool1_pad[((uint32_t)ch * 16U + (uint16_t)(y + 1U)) * 16U + 1U],
                   &g_cnn_pool1[((uint32_t)ch * EMNIST_CNN_POOL1_SIZE + y) * EMNIST_CNN_POOL1_SIZE],
                   EMNIST_CNN_POOL1_SIZE * sizeof(g_cnn_pool1[0]));
        }
    }
}

static void emnist_cnn_prepare_fixed(void)
{
    uint8_t i;

    if (g_cnn_fixed_ready)
    {
        return;
    }

    for (i = 0; i < EMNIST_CNN_CONV1_OUT; i++)
    {
        g_cnn_conv1_bias_q[i] = emnist_cnn_float_to_q(g_emnist_cnn_conv1_bias[i],
                                                      EMNIST_CNN_ACT_SHIFT);
    }

    for (i = 0; i < EMNIST_CNN_CONV2_OUT; i++)
    {
        g_cnn_conv2_bias_q[i] = emnist_cnn_float_to_q(g_emnist_cnn_conv2_bias[i],
                                                      EMNIST_CNN_ACT_SHIFT);
    }

    for (i = 0; i < EMNIST_CNN_OUTPUT_SIZE; i++)
    {
        g_cnn_fc_bias_q[i] = emnist_cnn_float_to_q(g_emnist_cnn_fc_bias[i],
                                                   EMNIST_CNN_ACT_SHIFT);
    }

    g_cnn_conv1_input_mult = emnist_cnn_float_to_mult(g_emnist_cnn_conv1_weight_scale *
                                                      (float)(1UL << EMNIST_CNN_ACT_SHIFT) / 255.0f,
                                                      EMNIST_CNN_MULT_SHIFT);
    g_cnn_conv2_mult = emnist_cnn_float_to_mult(g_emnist_cnn_conv2_weight_scale,
                                                EMNIST_CNN_MULT_SHIFT);
    g_cnn_fc_mult = emnist_cnn_float_to_mult(g_emnist_cnn_fc_weight_scale,
                                             EMNIST_CNN_MULT_SHIFT);

    g_cnn_fixed_ready = 1;
}

static void emnist_cnn_conv1_pool_fixed(const uint8_t image[EMNIST_CNN_INPUT_SIZE])
{
    uint8_t out_ch;

    emnist_cnn_prepare_input_pad(image);

    for (out_ch = 0; out_ch < EMNIST_CNN_CONV1_OUT; out_ch++)
    {
        const int8_t *weight_base = &g_emnist_cnn_conv1_weight_q7[(uint16_t)out_ch * 9U];
        uint8_t py;
        for (py = 0; py < EMNIST_CNN_POOL1_SIZE; py++)
        {
            uint8_t px;
            for (px = 0; px < EMNIST_CNN_POOL1_SIZE; px++)
            {
                int16_t max_value = 0;
                uint8_t dy;
                for (dy = 0; dy < 2; dy++)
                {
                    uint8_t dx;
                    for (dx = 0; dx < 2; dx++)
                    {
                        uint16_t input_index = (uint16_t)(py * 2U + dy) * 30U +
                                               (uint16_t)(px * 2U + dx);
                        int32_t acc = 0;

                        acc += (int32_t)g_cnn_input_pad[input_index] * weight_base[0];
                        acc += (int32_t)g_cnn_input_pad[input_index + 1U] * weight_base[1];
                        acc += (int32_t)g_cnn_input_pad[input_index + 2U] * weight_base[2];
                        input_index = (uint16_t)(input_index + 30U);
                        acc += (int32_t)g_cnn_input_pad[input_index] * weight_base[3];
                        acc += (int32_t)g_cnn_input_pad[input_index + 1U] * weight_base[4];
                        acc += (int32_t)g_cnn_input_pad[input_index + 2U] * weight_base[5];
                        input_index = (uint16_t)(input_index + 30U);
                        acc += (int32_t)g_cnn_input_pad[input_index] * weight_base[6];
                        acc += (int32_t)g_cnn_input_pad[input_index + 1U] * weight_base[7];
                        acc += (int32_t)g_cnn_input_pad[input_index + 2U] * weight_base[8];

                        acc = g_cnn_conv1_bias_q[out_ch] +
                              emnist_cnn_apply_mult(acc, g_cnn_conv1_input_mult);
                        if (dy == 0 && dx == 0)
                        {
                            max_value = emnist_cnn_relu_sat16(acc);
                        }
                        else
                        {
                            int16_t value = emnist_cnn_relu_sat16(acc);
                            if (value > max_value)
                            {
                                max_value = value;
                            }
                        }
                    }
                }
                g_cnn_pool1[((uint32_t)out_ch * EMNIST_CNN_POOL1_SIZE + py) *
                            EMNIST_CNN_POOL1_SIZE + px] = max_value;
            }
        }
    }
}

static int16_t emnist_cnn_conv2_point_fixed(uint8_t out_ch, uint8_t y, uint8_t x)
{
    int32_t acc = 0;
    uint8_t in_ch;

    for (in_ch = 0; in_ch < EMNIST_CNN_CONV1_OUT; in_ch++)
    {
        const int8_t *weight_base = &g_emnist_cnn_conv2_weight_q7[((uint32_t)out_ch * EMNIST_CNN_CONV1_OUT + in_ch) * 9U];
        uint16_t pool_index = ((uint16_t)in_ch * 16U + y) * 16U + x;

        acc += (int32_t)g_cnn_pool1_pad[pool_index] * weight_base[0];
        acc += (int32_t)g_cnn_pool1_pad[pool_index + 1U] * weight_base[1];
        acc += (int32_t)g_cnn_pool1_pad[pool_index + 2U] * weight_base[2];
        pool_index = (uint16_t)(pool_index + 16U);
        acc += (int32_t)g_cnn_pool1_pad[pool_index] * weight_base[3];
        acc += (int32_t)g_cnn_pool1_pad[pool_index + 1U] * weight_base[4];
        acc += (int32_t)g_cnn_pool1_pad[pool_index + 2U] * weight_base[5];
        pool_index = (uint16_t)(pool_index + 16U);
        acc += (int32_t)g_cnn_pool1_pad[pool_index] * weight_base[6];
        acc += (int32_t)g_cnn_pool1_pad[pool_index + 1U] * weight_base[7];
        acc += (int32_t)g_cnn_pool1_pad[pool_index + 2U] * weight_base[8];
    }

    acc = g_cnn_conv2_bias_q[out_ch] +
          emnist_cnn_apply_mult(acc, g_cnn_conv2_mult);
    return emnist_cnn_relu_sat16(acc);
}

static void emnist_cnn_conv2_pool_fixed(void)
{
    uint8_t out_ch;

    emnist_cnn_prepare_pool1_pad();

    for (out_ch = 0; out_ch < EMNIST_CNN_CONV2_OUT; out_ch++)
    {
        uint8_t py;
        for (py = 0; py < EMNIST_CNN_POOL2_SIZE; py++)
        {
            uint8_t px;
            for (px = 0; px < EMNIST_CNN_POOL2_SIZE; px++)
            {
                int16_t max_value = 0;
                uint8_t dy;
                for (dy = 0; dy < 2; dy++)
                {
                    uint8_t dx;
                    for (dx = 0; dx < 2; dx++)
                    {
                        int16_t value = emnist_cnn_conv2_point_fixed(out_ch,
                                                                     (uint8_t)(py * 2U + dy),
                                                                     (uint8_t)(px * 2U + dx));
                        if ((dy == 0 && dx == 0) || value > max_value)
                        {
                            max_value = value;
                        }
                    }
                }
                g_cnn_pool2[((uint32_t)out_ch * EMNIST_CNN_POOL2_SIZE + py) *
                            EMNIST_CNN_POOL2_SIZE + px] = max_value;
            }
        }
    }
}

static int32_t emnist_cnn_fc_score_fixed(uint8_t out)
{
    uint16_t index;
    uint32_t weight_offset = (uint32_t)out * EMNIST_CNN_FC_INPUT_SIZE;
    int32_t acc = 0;

    for (index = 0; index < EMNIST_CNN_FC_INPUT_SIZE; index++)
    {
        acc += (int32_t)g_cnn_pool2[index] *
               (int32_t)g_emnist_cnn_fc_weight_q7[weight_offset + index];
    }

    return g_cnn_fc_bias_q[out] + emnist_cnn_apply_mult(acc, g_cnn_fc_mult);
}
#endif

#ifndef EMNIST_CNN_MODEL_FROM_SD
uint8_t emnist_cnn_model_load(const char *path)
{
    (void)path;
    emnist_cnn_prepare_fixed();
    return 1;
}

uint8_t emnist_cnn_model_is_loaded(void)
{
    return 1;
}
#endif

uint8_t emnist_cnn_predict(const uint8_t image[EMNIST_CNN_INPUT_SIZE],
                           float logits[EMNIST_CNN_OUTPUT_SIZE])
{
    uint8_t out;
    uint8_t best = 0;

#ifdef EMNIST_CNN_MODEL_FROM_SD
    if (!g_cnn_model_loaded || f_open(&g_cnn_file, EMNIST_CNN_MODEL_PATH, FA_READ) != FR_OK)
    {
        memset(logits, 0, EMNIST_CNN_OUTPUT_SIZE * sizeof(float));
        return 0;
    }

    emnist_cnn_conv1_pool(image);
    if (!emnist_cnn_conv2_pool())
    {
        f_close(&g_cnn_file);
        memset(logits, 0, EMNIST_CNN_OUTPUT_SIZE * sizeof(float));
        return 0;
    }

    for (out = 0; out < EMNIST_CNN_OUTPUT_SIZE; out++)
    {
        float score;

        if (!emnist_cnn_fc_score(out, &score))
        {
            f_close(&g_cnn_file);
            memset(logits, 0, EMNIST_CNN_OUTPUT_SIZE * sizeof(float));
            return 0;
        }

        logits[out] = score;

        if (out > 0 && score > logits[best])
        {
            best = out;
        }
    }

    f_close(&g_cnn_file);
#else
    int32_t best_score;

    emnist_cnn_prepare_fixed();
    emnist_cnn_conv1_pool_fixed(image);
    emnist_cnn_conv2_pool_fixed();

    best_score = emnist_cnn_fc_score_fixed(0);
    logits[0] = (float)best_score;

    for (out = 1; out < EMNIST_CNN_OUTPUT_SIZE; out++)
    {
        int32_t score = emnist_cnn_fc_score_fixed(out);
        logits[out] = (float)score;

        if (score > best_score)
        {
            best_score = score;
            best = out;
        }
    }
#endif

    return best;
}
char emnist_cnn_label(uint8_t class_id)
{
    if (class_id >= EMNIST_CNN_OUTPUT_SIZE)
    {
        return '?';
    }

    return g_emnist_cnn_labels[class_id];
}


