#include "nai_flash_port.h"
#include "nai_flash_layout.h"

#include <string.h>

extern "C" {
#include "hal_flash.h"
#include "bl702_sf_ctrl.h"
#include "bl702_xip_sflash.h"
#include "bl702_l1c.h"
#include "bl702_common.h"
}

static bool range_ok(uint32_t off, size_t len)
{
    if (len == 0) return true;
    if (off >= NAI_FLASH_TOTAL_BYTES) return false;
    if (len > NAI_FLASH_TOTAL_BYTES) return false;
    return off <= NAI_FLASH_TOTAL_BYTES - len;
}

bool nai_flash_init()
{
    /*
     * bflb_platform_init() already calls flash_init().
     *
     * Calling it again was one source of trouble in the earlier firmware,
     * so our wrapper deliberately does not reinitialize the flash controller.
     */
    return true;
}

bool nai_flash_read(uint32_t physical_offset, void *dst, size_t len)
{
    if ((!dst && len) || !range_ok(physical_offset, len)) return false;
    if (len == 0) return true;

    /*
     * Read through the already-live XIP mapping.
     *
     * The CPU XIP window maps flash physical "image_offset" to
     * BL702_FLASH_XIP_BASE, so convert physical offset -> mapped address.
     *
     * This avoids invoking flash_read()/XIP opt enter for ordinary reads.
     */
    const uint32_t image_offset = SF_Ctrl_Get_Flash_Image_Offset();

    if (physical_offset < image_offset) return false;

    const uintptr_t mapped =
        (uintptr_t)BL702_FLASH_XIP_BASE +
        (uintptr_t)(physical_offset - image_offset);

    memcpy(dst, reinterpret_cast<const void *>(mapped), len);
    return true;
}

bool nai_flash_read_fresh(uint32_t physical_offset, void *dst, size_t len)
{
    if ((!dst && len) || !range_ok(physical_offset, len)) return false;
    if (len == 0) return true;

    const uint32_t image_offset = SF_Ctrl_Get_Flash_Image_Offset();

    if (physical_offset < image_offset) return false;

    const uintptr_t mapped =
        (uintptr_t)BL702_FLASH_XIP_BASE +
        (uintptr_t)(physical_offset - image_offset);

    /*
     * The BL702 HAL's own flash_read_via_xip() flushes L1 cache before and
     * after cached flash reads.  Our original raw memcpy path did not.
     *
     * This matters for the slot header because that same address was recently
     * erased/programmed and may still be present in cache as stale 0xFF data.
     *
     * Keep interrupts disabled only for this short metadata read.
     */
    __disable_irq();
    L1C_Cache_Flush_Ext();

    memcpy(dst, reinterpret_cast<const void *>(mapped), len);

    L1C_Cache_Flush_Ext();
    __enable_irq();

    return true;
}

void nai_flash_flush_cache()
{
    /*
     * Keep this as a short one-shot barrier before opening a newly committed
     * model. File-backed FCN reads themselves keep the fast cached XIP path.
     */
    __disable_irq();
    L1C_Cache_Flush_Ext();
    __enable_irq();
}

bool nai_flash_write(uint32_t physical_offset, const void *src, size_t len)
{
    if ((!src && len) || !range_ok(physical_offset, len)) return false;
    if (len == 0) return true;

    const uint8_t *p = static_cast<const uint8_t *>(src);
    size_t done = 0;

    /*
     * Keep each program operation inside one 256-byte flash page.
     * BLE chunks may have arbitrary sizes.
     */
    while (done < len) {
        const uint32_t addr = physical_offset + (uint32_t)done;
        const uint32_t page_left =
            NAI_FLASH_PAGE_BYTES - (addr % NAI_FLASH_PAGE_BYTES);

        uint32_t n = (uint32_t)(len - done);
        if (n > page_left) n = page_left;

        if (flash_write(
                addr,
                const_cast<uint8_t *>(p + done),
                n) != SUCCESS) {
            return false;
        }

        done += n;
    }

    return true;
}

bool nai_flash_erase(uint32_t physical_offset, size_t len)
{
    if (len == 0 || !range_ok(physical_offset, len)) return false;
    if ((physical_offset % NAI_FLASH_SECTOR_BYTES) != 0) return false;
    if ((len % NAI_FLASH_SECTOR_BYTES) != 0) return false;

    return flash_erase(
        physical_offset,
        (uint32_t)len) == SUCCESS;
}
