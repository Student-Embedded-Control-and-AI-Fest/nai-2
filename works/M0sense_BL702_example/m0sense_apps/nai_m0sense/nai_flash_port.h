#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

bool nai_flash_init();
bool nai_flash_read(uint32_t physical_offset, void *dst, size_t len);

/*
 * Cache-safe XIP read used for metadata/header validation immediately after
 * flash programming.  Ordinary payload verification keeps using nai_flash_read().
 */
bool nai_flash_read_fresh(uint32_t physical_offset, void *dst, size_t len);

/* Flush stale BL702 XIP cache lines after a model was programmed. */
void nai_flash_flush_cache();
bool nai_flash_write(uint32_t physical_offset, const void *src, size_t len);
bool nai_flash_erase(uint32_t physical_offset, size_t len);
