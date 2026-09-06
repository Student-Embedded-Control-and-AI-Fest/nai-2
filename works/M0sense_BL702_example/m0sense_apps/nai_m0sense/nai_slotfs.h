#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "nai_flash_layout.h"

static constexpr uint16_t NAI_SLOT_MAX_FILES = 19;
static constexpr uint8_t NAI_SLOT_NAME_BYTES = 16;

#pragma pack(push, 1)
struct NaiSlotFileEntry {
    char name[NAI_SLOT_NAME_BYTES];
    uint32_t offset;
    uint32_t size;
    uint32_t crc32;
};

struct NaiSlotHeader {
    char magic[4];              /* "NSLT" */
    uint16_t version;           /* 1 */
    uint16_t header_bytes;
    uint32_t generation;
    uint16_t file_count;
    uint16_t reserved;
    uint32_t payload_bytes;
    NaiSlotFileEntry files[NAI_SLOT_MAX_FILES];
    uint32_t header_crc32;
    uint32_t commit_word;       /* 0 committed; 0xFFFFFFFF staged */
};
#pragma pack(pop)

static_assert(sizeof(NaiSlotHeader) < NAI_SLOT_HEADER_BYTES,
              "slot header must fit in first 4 KiB sector");

enum NaiSlotResult : int {
    NAI_SLOT_OK = 0,
    NAI_SLOT_ERR_STATE = -1,
    NAI_SLOT_ERR_NAME = -2,
    NAI_SLOT_ERR_SIZE = -3,
    NAI_SLOT_ERR_FLASH = -4,
    NAI_SLOT_ERR_COUNT = -5,
    NAI_SLOT_ERR_CRC = -6,
    NAI_SLOT_ERR_DUPLICATE = -7,
    NAI_SLOT_ERR_HEADER = -8,
};

bool nai_slotfs_init();

bool nai_slotfs_canonical_name_allowed(const char *name);
bool nai_slotfs_virtual_name(const char *virtual_name, char *slot_out,
                             char canonical_out[NAI_SLOT_NAME_BYTES]);

bool nai_slotfs_read_header(char slot, NaiSlotHeader *out, bool allow_staged);
bool nai_slotfs_lookup(char slot, const char *canonical,
                       uint32_t *physical_addr, uint32_t *size,
                       bool allow_staged);

char nai_slotfs_active_slot();

NaiSlotResult nai_slotfs_deploy_begin();
bool nai_slotfs_deploy_active();
bool nai_slotfs_file_active();

NaiSlotResult nai_slotfs_file_begin(const char *canonical,
                                    uint32_t expected_size,
                                    uint32_t expected_crc);
NaiSlotResult nai_slotfs_file_write(const uint8_t *data, size_t len);
NaiSlotResult nai_slotfs_file_end();

/*
 * Incremental final commit.
 *
 * begin(): prepares header CRC in RAM.
 * step(): writes at most 128 header bytes or performs one tiny verify/commit
 *         phase, then returns to the app loop.
 */
NaiSlotResult nai_slotfs_commit_begin();
NaiSlotResult nai_slotfs_commit_step(bool *done);

/* Legacy helpers retained for compatibility but unused by Mode 2 v0.2.2. */
NaiSlotResult nai_slotfs_stage_header();
NaiSlotResult nai_slotfs_activate_staged();

void nai_slotfs_abort(bool erase_target);

void nai_slotfs_set_staged_read(char slot, bool enable);
bool nai_slotfs_staged_read_allowed(char slot);

uint32_t nai_crc32_update(uint32_t crc, const uint8_t *data, size_t len);
uint32_t nai_crc32(const void *data, size_t len);
