#ifndef EMNIST_CNN_H
#define EMNIST_CNN_H

#include <stdint.h>

#include "emnist_cnn_data.h"

uint8_t emnist_cnn_model_load(const char *path);
uint8_t emnist_cnn_model_is_loaded(void);
uint8_t emnist_cnn_predict(const uint8_t image[EMNIST_CNN_INPUT_SIZE],
                           float logits[EMNIST_CNN_OUTPUT_SIZE]);
char emnist_cnn_label(uint8_t class_id);

#endif

