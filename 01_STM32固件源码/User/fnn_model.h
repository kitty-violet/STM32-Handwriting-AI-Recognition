#ifndef FNN_MODEL_H
#define FNN_MODEL_H

#include <stdint.h>

#define FNN_MODEL_PATH "0:/MODEL/FNN.BIN"

uint8_t fnn_model_load(const char *path);
uint8_t fnn_model_is_loaded(void);

#endif

