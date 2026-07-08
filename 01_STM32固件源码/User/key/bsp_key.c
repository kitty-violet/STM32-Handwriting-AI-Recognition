#include "./key/bsp_key.h"

static uint8_t g_key1_last_pressed = 0;
static uint8_t g_key2_last_pressed = 0;

static void key_delay(volatile uint32_t count)
{
    while (count-- > 0U)
    {
    }
}

void Key_GPIO_Config(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;

    RCC_APB2PeriphClockCmd(KEY1_GPIO_CLK | KEY2_GPIO_CLK, ENABLE);

    GPIO_InitStructure.GPIO_Pin = KEY1_GPIO_PIN;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_Init(KEY1_GPIO_PORT, &GPIO_InitStructure);

    GPIO_InitStructure.GPIO_Pin = KEY2_GPIO_PIN;
    GPIO_Init(KEY2_GPIO_PORT, &GPIO_InitStructure);
}

uint8_t Key_Scan(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
    if (GPIO_ReadInputDataBit(GPIOx, GPIO_Pin) == KEY_ON)
    {
        key_delay(72000U);
        if (GPIO_ReadInputDataBit(GPIOx, GPIO_Pin) == KEY_ON)
        {
            while (GPIO_ReadInputDataBit(GPIOx, GPIO_Pin) == KEY_ON)
            {
            }
            return 1U;
        }
    }

    return 0U;
}

uint8_t Key_IsPressed(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
    return (GPIO_ReadInputDataBit(GPIOx, GPIO_Pin) == KEY_ON) ? 1U : 0U;
}

void Key_ResetEvent(void)
{
    Key_ResetAllEvents();
}

uint8_t Key_PressedEvent(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin)
{
    uint8_t pressed = Key_IsPressed(GPIOx, GPIO_Pin);
    uint8_t *last_pressed = &g_key1_last_pressed;

    if (GPIOx == KEY2_GPIO_PORT && GPIO_Pin == KEY2_GPIO_PIN)
    {
        last_pressed = &g_key2_last_pressed;
    }

    if (pressed && !(*last_pressed))
    {
        key_delay(72000U);
        pressed = Key_IsPressed(GPIOx, GPIO_Pin);
        *last_pressed = pressed;
        return pressed ? 1U : 0U;
    }

    *last_pressed = pressed;
    return 0U;
}

void Key_ResetAllEvents(void)
{
    g_key1_last_pressed = Key_IsPressed(KEY1_GPIO_PORT, KEY1_GPIO_PIN);
    g_key2_last_pressed = Key_IsPressed(KEY2_GPIO_PORT, KEY2_GPIO_PIN);
}

