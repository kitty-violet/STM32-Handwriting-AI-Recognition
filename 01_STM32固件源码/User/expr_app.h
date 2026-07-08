#ifndef EXPR_APP_H
#define EXPR_APP_H

#include <stdint.h>

typedef struct
{
    char text[27];
    int32_t value;
    uint8_t ok;
} ExprAppResult;

void expr_app_process_string(const char *text, uint8_t expression_mode, ExprAppResult *out);

#endif

