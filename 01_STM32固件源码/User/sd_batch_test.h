#ifndef SD_BATCH_TEST_H
#define SD_BATCH_TEST_H

#include <stdint.h>

typedef uint8_t (*SdBatchAbortCallback)(void);

uint8_t sd_batch_test_run_once(SdBatchAbortCallback abort_callback);

#endif

