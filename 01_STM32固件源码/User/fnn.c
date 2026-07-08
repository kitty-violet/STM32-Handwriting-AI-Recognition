#include "fnn.h"

#include "FNN_Data.h"

/**
 * @brief Calculate first fully-connected layer and ReLU activation.
 * @param[in] image Input image, pixel range 0-255.
 * @param[out] hidden Hidden activation buffer.
 */
static void fnn_calculate_hidden(const uint8_t image[FNN_INPUT_SIZE],
                                 float hidden[FNN_HIDDEN_SIZE])
{
    uint16_t hidden_index;

    for (hidden_index = 0; hidden_index < FNN_HIDDEN_SIZE; hidden_index++)
    {
        int32_t accumulator = 0;
        uint16_t input_index;
        uint32_t weight_offset = (uint32_t)hidden_index * FNN_INPUT_SIZE;

        for (input_index = 0; input_index < FNN_INPUT_SIZE; input_index++)
        {
            accumulator += (int32_t)g_fnn_fc1_weight[weight_offset + input_index] *
                           (int32_t)image[input_index];
        }

        hidden[hidden_index] = ((float)accumulator * (g_fnn_fc1_scale / 255.0f)) +
                               g_fnn_fc1_bias[hidden_index];
        if (hidden[hidden_index] < 0.0f)
        {
            hidden[hidden_index] = 0.0f;
        }
    }
}

/**
 * @brief Calculate output layer scores.
 * @param[in] hidden Hidden activation buffer.
 * @param[out] logits Output score buffer.
 */
static void fnn_calculate_logits(const float hidden[FNN_HIDDEN_SIZE],
                                 float logits[FNN_OUTPUT_SIZE])
{
    uint8_t output_index;

    for (output_index = 0; output_index < FNN_OUTPUT_SIZE; output_index++)
    {
        float score = g_fnn_fc2_bias[output_index];
        uint16_t hidden_index;
        uint32_t weight_offset = (uint32_t)output_index * FNN_HIDDEN_SIZE;

        for (hidden_index = 0; hidden_index < FNN_HIDDEN_SIZE; hidden_index++)
        {
            score += hidden[hidden_index] *
                     (float)g_fnn_fc2_weight[weight_offset + hidden_index] *
                     g_fnn_fc2_scale;
        }

        logits[output_index] = score;
    }
}

uint8_t fnn_predict(const uint8_t image[FNN_INPUT_SIZE], float logits[FNN_OUTPUT_SIZE])
{
    float hidden[FNN_HIDDEN_SIZE];
    uint8_t output_index;
    uint8_t best_index = 0;

    fnn_calculate_hidden(image, hidden);
    fnn_calculate_logits(hidden, logits);

    for (output_index = 1; output_index < FNN_OUTPUT_SIZE; output_index++)
    {
        if (logits[output_index] > logits[best_index])
        {
            best_index = output_index;
        }
    }

    return best_index;
}

