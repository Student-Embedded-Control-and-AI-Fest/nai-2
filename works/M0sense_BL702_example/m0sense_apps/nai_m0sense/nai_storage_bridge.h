#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool nai_storage_init(void);
bool nai_storage_model_valid(void);

int nai_storage_deploy_begin(void);
int nai_storage_erase_sector(uint32_t relative_offset);
uint32_t nai_storage_region_bytes(void);
int nai_storage_file_begin(const char *name, uint32_t size, uint32_t crc32);
int nai_storage_file_write(const uint8_t *data, size_t len);
int nai_storage_file_end(void);

/* Incremental commit: begin once, then call step() from the app loop. */
int nai_storage_commit_begin(void);
int nai_storage_commit_step(int *done);

void nai_storage_abort(int erase_target);

uint32_t nai_storage_payload_capacity(void);

#ifdef __cplusplus
}
#endif
