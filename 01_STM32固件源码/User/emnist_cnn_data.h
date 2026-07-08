#ifndef EMNIST_CNN_DATA_H
#define EMNIST_CNN_DATA_H

#include <stdint.h>

#define EMNIST_CNN_INPUT_SIZE 784
#define EMNIST_CNN_OUTPUT_SIZE 62
#define EMNIST_CNN_CONV1_OUT 16
#define EMNIST_CNN_CONV2_OUT 32
#define EMNIST_CNN_POOL1_SIZE 14
#define EMNIST_CNN_POOL2_SIZE 7
#define EMNIST_CNN_FC_INPUT_SIZE (EMNIST_CNN_CONV2_OUT * EMNIST_CNN_POOL2_SIZE * EMNIST_CNN_POOL2_SIZE)

extern const int8_t g_emnist_cnn_conv1_weight_q7[EMNIST_CNN_CONV1_OUT * 3 * 3];
extern const float g_emnist_cnn_conv1_weight_scale;
extern const float g_emnist_cnn_conv1_bias[EMNIST_CNN_CONV1_OUT];
extern const int8_t g_emnist_cnn_conv2_weight_q7[EMNIST_CNN_CONV2_OUT * EMNIST_CNN_CONV1_OUT * 3 * 3];
extern const float g_emnist_cnn_conv2_weight_scale;
extern const float g_emnist_cnn_conv2_bias[EMNIST_CNN_CONV2_OUT];
extern const int8_t g_emnist_cnn_fc_weight_q7[EMNIST_CNN_OUTPUT_SIZE * EMNIST_CNN_FC_INPUT_SIZE];
extern const float g_emnist_cnn_fc_weight_scale;
extern const float g_emnist_cnn_fc_bias[EMNIST_CNN_OUTPUT_SIZE];
extern const char g_emnist_cnn_labels[EMNIST_CNN_OUTPUT_SIZE + 1];

#endif

