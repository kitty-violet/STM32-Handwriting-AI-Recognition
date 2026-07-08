#ifndef BSP_KEY_H
#define BSP_KEY_H

#include "stm32f10x.h"

#define KEY1_GPIO_CLK  RCC_APB2Periph_GPIOA
#define KEY1_GPIO_PORT GPIOA
#define KEY1_GPIO_PIN  GPIO_Pin_0

#define KEY2_GPIO_CLK  RCC_APB2Periph_GPIOC
#define KEY2_GPIO_PORT GPIOC
#define KEY2_GPIO_PIN  GPIO_Pin_13

#define KEY_ON  0U
#define KEY_OFF 1U

void Key_GPIO_Config(void);
uint8_t Key_Scan(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);
uint8_t Key_IsPressed(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);
void Key_ResetEvent(void);
uint8_t Key_PressedEvent(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);
void Key_ResetAllEvents(void);

#endif

