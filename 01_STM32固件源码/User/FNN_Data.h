#ifndef FNN_DATA_H
#define FNN_DATA_H

#include <stdint.h>

#define FNN_INPUT_SIZE 784
#define FNN_HIDDEN_SIZE 24
#define FNN_OUTPUT_SIZE 10

#ifdef FNN_MODEL_FROM_SD
extern int8_t g_fnn_fc1_weight[FNN_HIDDEN_SIZE * FNN_INPUT_SIZE];
extern float g_fnn_fc1_bias[FNN_HIDDEN_SIZE];
extern int8_t g_fnn_fc2_weight[FNN_OUTPUT_SIZE * FNN_HIDDEN_SIZE];
extern float g_fnn_fc2_bias[FNN_OUTPUT_SIZE];
extern float g_fnn_fc1_scale;
extern float g_fnn_fc2_scale;
#else
extern const int8_t g_fnn_fc1_weight[FNN_HIDDEN_SIZE * FNN_INPUT_SIZE];
extern const float g_fnn_fc1_bias[FNN_HIDDEN_SIZE];
extern const int8_t g_fnn_fc2_weight[FNN_OUTPUT_SIZE * FNN_HIDDEN_SIZE];
extern const float g_fnn_fc2_bias[FNN_OUTPUT_SIZE];
extern const float g_fnn_fc1_scale;
extern const float g_fnn_fc2_scale;
extern const uint8_t g_digit_test_image[FNN_INPUT_SIZE];
extern const uint8_t g_digit_test_label;
#endif

#endif

