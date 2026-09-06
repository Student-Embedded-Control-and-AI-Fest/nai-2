#pragma once
#include <stdint.h>

/*
 * M0Sense / BL702 physical flash = 512 KiB.
 *
 * App image starts at physical 0x2000.
 *
 * We deliberately reserve a little more app room than the old build:
 *
 *   0x00000 .. 0x01FFF   boot / RF / platform area
 *   0x02000 .. 0x4FFFF   NoodleAI application (max 312 KiB)
 *   0x50000 .. 0x7FFFF   ONE model slot (192 KiB)
 *
 * Model slot:
 *   first 4 KiB = header
 *   remaining   = 188 KiB payload
 *
 * Historical Float32 Mode-2 example (for comparison):
 *   raw NAI4 files ~= 165.2 KiB
 * so it fits with ~22.8 KiB payload margin.
 */
static constexpr uint32_t NAI_FLASH_TOTAL_BYTES = 0x00080000u;

static constexpr uint32_t NAI_APP_PHYS_BASE = 0x00002000u;
static constexpr uint32_t NAI_APP_RESERVED_END = 0x00050000u;
static constexpr uint32_t NAI_APP_RESERVED_BYTES =
    NAI_APP_RESERVED_END - NAI_APP_PHYS_BASE; /* 0x4E000 = 312 KiB */

static constexpr uint32_t NAI_SLOT_A_BASE = 0x00050000u;
static constexpr uint32_t NAI_SLOT_BYTES = 0x00030000u; /* 192 KiB */

static constexpr uint32_t NAI_SLOT_HEADER_BYTES = 0x00001000u; /* 4 KiB */
static constexpr uint32_t NAI_SLOT_PAYLOAD_BYTES =
    NAI_SLOT_BYTES - NAI_SLOT_HEADER_BYTES; /* 188 KiB */

static constexpr uint32_t NAI_FLASH_SECTOR_BYTES = 4096u;
static constexpr uint32_t NAI_FLASH_PAGE_BYTES = 256u;
