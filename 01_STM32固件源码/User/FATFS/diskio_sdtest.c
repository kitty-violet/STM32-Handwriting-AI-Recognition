/* SDTEST-only FatFs disk I/O wrapper.
 *
 * The original example diskio.c uses SD_ReadMultiBlocks() and then waits for
 * DMA/IRQ completion. The SDTEST build uses polling mode, so mounting can hang
 * while reading the boot sector. This wrapper keeps FatFs read-only and reads
 * sectors one by one with SD_ReadBlock().
 */

#ifndef SD_BATCH_TEST_LCD
#include <stdio.h>
#endif
#include <string.h>

#include "diskio.h"
#include "stm32f10x.h"
#include "./sdio/bsp_sdio_sdcard.h"

#ifndef __weak
  #if defined(__clang__)
    #define __weak __attribute__((weak))
  #else
    #define __weak __weak
  #endif
#endif

#define SDTEST_DRIVE_SD 0
#define SDTEST_BLOCK_SIZE 512U
#define SDTEST_INIT_RETRY 3U

extern SD_CardInfo SDCardInfo;

static uint8_t g_sdtest_initialized = 0;
static uint8_t g_sdtest_diag_reads = 0;
static DWORD g_sdtest_scratch[SDTEST_BLOCK_SIZE / sizeof(DWORD)];

static void sdtest_delay(volatile uint32_t count)
{
    while (count-- > 0U)
    {
    }
}

DSTATUS disk_status(BYTE pdrv)
{
    if (pdrv != SDTEST_DRIVE_SD || !g_sdtest_initialized)
    {
        return STA_NOINIT;
    }
    return 0;
}

DSTATUS disk_initialize(BYTE pdrv)
{
    SD_Error result = SD_ERROR;
    uint8_t attempt;

    if (pdrv != SDTEST_DRIVE_SD)
    {
        return STA_NOINIT;
    }

    g_sdtest_initialized = 0;
#ifndef SD_BATCH_TEST_LCD
    printf("[DISK] SD_Init start\r\n");
#endif

    for (attempt = 0U; attempt < SDTEST_INIT_RETRY; attempt++)
    {
        SD_DeInit();
        sdtest_delay(720000U);
        result = SD_Init();
#ifndef SD_BATCH_TEST_LCD
        printf("[DISK] SD_Init try=%u result=%d\r\n", (uint16_t)(attempt + 1U), result);
#endif
        if (result == SD_OK)
        {
            break;
        }
    }

    if (result != SD_OK)
    {
        return STA_NOINIT;
    }

    g_sdtest_initialized = 1;
#ifndef SD_BATCH_TEST_LCD
    printf("[DISK] capacity=%lu block=%lu type=%u\r\n",
           (unsigned long)(SDCardInfo.CardCapacity / 1024ULL / 1024ULL),
           (unsigned long)SDCardInfo.CardBlockSize,
           SDCardInfo.CardType);
#endif
    return 0;
}

DRESULT disk_read(BYTE pdrv, BYTE *buff, DWORD sector, UINT count)
{
    BYTE *dst = buff;
    DWORD current_sector = sector;
    SD_Error result = SD_OK;

    if (pdrv != SDTEST_DRIVE_SD || buff == 0 || count == 0U)
    {
        return RES_PARERR;
    }

    if (!g_sdtest_initialized)
    {
        return RES_NOTRDY;
    }

    if (g_sdtest_diag_reads < 6U)
    {
#ifndef SD_BATCH_TEST_LCD
        printf("[DISK] read sector=%lu count=%u\r\n",
               (unsigned long)sector,
               count);
#endif
    }

    while (count-- > 0U)
    {
        BYTE *target = dst;

        if (((uint32_t)dst & 3U) != 0U)
        {
            target = (BYTE *)g_sdtest_scratch;
        }

        result = SD_ReadBlock(target,
                              (uint64_t)current_sector * SDTEST_BLOCK_SIZE,
                              SDTEST_BLOCK_SIZE);
        if (result != SD_OK)
        {
            break;
        }

        if (target != dst)
        {
            memcpy(dst, target, SDTEST_BLOCK_SIZE);
        }

        dst += SDTEST_BLOCK_SIZE;
        current_sector++;
    }

    if (g_sdtest_diag_reads < 6U)
    {
#ifndef SD_BATCH_TEST_LCD
        printf("[DISK] read result=%d\r\n", result);
#endif
        g_sdtest_diag_reads++;
    }

    return (result == SD_OK) ? RES_OK : RES_ERROR;
}

DRESULT disk_write(BYTE pdrv, const BYTE *buff, DWORD sector, UINT count)
{
    const BYTE *src = buff;
    DWORD current_sector = sector;
    SD_Error result = SD_OK;

    if (pdrv != SDTEST_DRIVE_SD || buff == 0 || count == 0U)
    {
        return RES_PARERR;
    }

    if (!g_sdtest_initialized)
    {
        return RES_NOTRDY;
    }

    while (count-- > 0U)
    {
        BYTE *target = (BYTE *)src;

        if (((uint32_t)src & 3U) != 0U)
        {
            memcpy(g_sdtest_scratch, src, SDTEST_BLOCK_SIZE);
            target = (BYTE *)g_sdtest_scratch;
        }

        result = SD_WriteBlock(target,
                               (uint64_t)current_sector * SDTEST_BLOCK_SIZE,
                               SDTEST_BLOCK_SIZE);
        if (result != SD_OK)
        {
            break;
        }

        result = SD_WaitWriteOperation();
        if (result != SD_OK)
        {
            break;
        }

        while (SD_GetStatus() != SD_TRANSFER_OK)
        {
        }

        src += SDTEST_BLOCK_SIZE;
        current_sector++;
    }

    return (result == SD_OK) ? RES_OK : RES_ERROR;
}

DRESULT disk_ioctl(BYTE pdrv, BYTE cmd, void *buff)
{
    if (pdrv != SDTEST_DRIVE_SD)
    {
        return RES_PARERR;
    }

    switch (cmd)
    {
    case CTRL_SYNC:
        return RES_OK;

    case GET_SECTOR_SIZE:
        *(WORD *)buff = SDTEST_BLOCK_SIZE;
        return RES_OK;

    case GET_BLOCK_SIZE:
        *(DWORD *)buff = 1U;
        return RES_OK;

    case GET_SECTOR_COUNT:
        if (SDCardInfo.CardBlockSize == 0U)
        {
            return RES_NOTRDY;
        }
        *(DWORD *)buff = (DWORD)(SDCardInfo.CardCapacity / SDCardInfo.CardBlockSize);
        return RES_OK;

    default:
        return RES_PARERR;
    }
}

__weak DWORD get_fattime(void)
{
    return ((DWORD)(2015U - 1980U) << 25) |
           ((DWORD)1U << 21) |
           ((DWORD)1U << 16);
}

