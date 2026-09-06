#include "nai_storage_bridge.h"

#include "nai_slotfs.h"
#include "nai_flash_port.h"
#include "nai_flash_layout.h"

extern "C" bool nai_storage_init(void)
{
    return nai_slotfs_init();
}

extern "C" bool nai_storage_model_valid(void)
{
    return nai_slotfs_active_slot() == 'A';
}

extern "C" int nai_storage_deploy_begin(void)
{
    return (int)nai_slotfs_deploy_begin();
}

extern "C" int nai_storage_erase_sector(uint32_t relative_offset)
{
    if ((relative_offset % NAI_FLASH_SECTOR_BYTES) != 0u ||
        relative_offset >= NAI_SLOT_BYTES) return -1;

    return nai_flash_erase(
        NAI_SLOT_A_BASE + relative_offset,
        NAI_FLASH_SECTOR_BYTES) ? 0 : -4;
}

extern "C" uint32_t nai_storage_region_bytes(void)
{
    return NAI_SLOT_BYTES;
}

extern "C" int nai_storage_file_begin(
    const char *name,
    uint32_t size,
    uint32_t crc32)
{
    return (int)nai_slotfs_file_begin(name, size, crc32);
}

extern "C" int nai_storage_file_write(
    const uint8_t *data,
    size_t len)
{
    return (int)nai_slotfs_file_write(data, len);
}

extern "C" int nai_storage_file_end(void)
{
    return (int)nai_slotfs_file_end();
}

extern "C" int nai_storage_commit_begin(void)
{
    return (int)nai_slotfs_commit_begin();
}

extern "C" int nai_storage_commit_step(int *done)
{
    bool finished = false;
    const NaiSlotResult rc = nai_slotfs_commit_step(&finished);

    if (done) {
        *done = finished ? 1 : 0;
    }

    return (int)rc;
}

extern "C" void nai_storage_abort(int erase_target)
{
    nai_slotfs_abort(erase_target != 0);
}

extern "C" uint32_t nai_storage_payload_capacity(void)
{
    return NAI_SLOT_PAYLOAD_BYTES;
}
