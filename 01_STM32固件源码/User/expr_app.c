#include "expr_app.h"

#include <string.h>

static int32_t expr_app_parse_number(const char **cursor, uint8_t *ok);
static uint8_t expr_app_eval(const char *text, int32_t *value);

void expr_app_process_string(const char *text, uint8_t expression_mode, ExprAppResult *out)
{
    uint8_t i;

    if (out == 0)
    {
        return;
    }

    memset(out, 0, sizeof(*out));

    for (i = 0U; i < (sizeof(out->text) - 1U) && text[i] != '\0'; i++)
    {
        out->text[i] = text[i];
    }
    out->text[i] = '\0';

    if (expression_mode)
    {
        out->ok = expr_app_eval(out->text, &out->value);
    }
}

static int32_t expr_app_parse_number(const char **cursor, uint8_t *ok)
{
    int32_t value = 0;
    uint8_t has_digit = 0U;

    while (**cursor >= '0' && **cursor <= '9')
    {
        has_digit = 1U;
        value = value * 10 + (int32_t)(**cursor - '0');
        (*cursor)++;
    }

    if (!has_digit)
    {
        *ok = 0U;
    }

    return value;
}

static uint8_t expr_app_eval(const char *text, int32_t *value)
{
    const char *cursor = text;
    int32_t acc;
    uint8_t ok = 1U;

    if (text == 0 || value == 0)
    {
        return 0U;
    }

    acc = expr_app_parse_number(&cursor, &ok);
    while (ok && *cursor != '\0' && *cursor != '=')
    {
        char op = *cursor++;
        int32_t rhs = expr_app_parse_number(&cursor, &ok);

        if (!ok)
        {
            break;
        }

        if (op == '+')
        {
            acc += rhs;
        }
        else if (op == '-')
        {
            acc -= rhs;
        }
        else if (op == 'x' || op == 'X' || op == '*')
        {
            acc *= rhs;
        }
        else if (op == '/')
        {
            if (rhs == 0)
            {
                ok = 0U;
            }
            else
            {
                acc /= rhs;
            }
        }
        else
        {
            ok = 0U;
        }
    }

    if (ok && *cursor == '=')
    {
        cursor++;
    }

    if (ok && *cursor != '\0')
    {
        ok = 0U;
    }

    *value = acc;
    return ok;
}

