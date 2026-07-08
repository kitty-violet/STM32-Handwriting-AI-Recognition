#ifndef FNN_H
#define FNN_H

#include <stdint.h>

/**
 * @brief Run handwritten digit recognition on a 28x28 grayscale image.
 * @param[in] image Input image, 784 bytes, pixel range 0-255.
 * @param[out] logits Output score buffer with 10 elements.
 * @return Predicted digit, range 0-9.
 */
uint8_t fnn_predict(const uint8_t image[784], float logits[10]);

#endif

