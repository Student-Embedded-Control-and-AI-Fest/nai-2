#include "nai_slotfs.h"
#include "nai_flash_port.h"

#include <stddef.h>
#include <string.h>

namespace {

static constexpr char MODEL_SLOT = 'A';
static constexpr uint16_t SLOT_VERSION = 1;
static constexpr uint32_t SLOT_COMMITTED = 0x00000000u;
static constexpr uint32_t SLOT_ERASED = 0xFFFFFFFFu;

struct DeployState {
    bool active = false;
    bool file_active = false;

    NaiSlotHeader header{};
    uint32_t payload_cursor = NAI_SLOT_HEADER_BYTES;

    int current_index = -1;
    uint32_t expected_size = 0;
    uint32_t expected_crc = 0;
    uint32_t received = 0;
    uint32_t running_crc = 0xFFFFFFFFu;
};

DeployState s_deploy;
bool s_staged_enabled = false;

/*
 * Keep the ~560-byte slot-header scratch buffer in BSS rather than on the
 * application task stack.
 */
NaiSlotHeader s_header_scratch{};

enum CommitPhase : uint8_t {
    COMMIT_IDLE = 0,
    COMMIT_WRITE_HEADER,
    COMMIT_VERIFY_STAGED,
    COMMIT_WRITE_WORD,
    COMMIT_VERIFY_FINAL,
};

CommitPhase s_commit_phase = COMMIT_IDLE;
uint32_t s_commit_offset = 0;

static constexpr uint32_t COMMIT_CHUNK_BYTES = 128u;

uint32_t align4(uint32_t x)
{
    return (x + 3u) & ~3u;
}

bool header_structurally_valid(const NaiSlotHeader &h, bool allow_staged)
{
    if (memcmp(h.magic, "NSLT", 4) != 0) return false;
    if (h.version != SLOT_VERSION) return false;
    if (h.header_bytes != sizeof(NaiSlotHeader)) return false;
    if (h.file_count == 0 || h.file_count > NAI_SLOT_MAX_FILES) return false;
    if (h.payload_bytes > NAI_SLOT_PAYLOAD_BYTES) return false;

    if (allow_staged) {
        if (h.commit_word != SLOT_ERASED &&
            h.commit_word != SLOT_COMMITTED) return false;
    } else {
        if (h.commit_word != SLOT_COMMITTED) return false;
    }

    const uint32_t want =
        nai_crc32(&h, offsetof(NaiSlotHeader, header_crc32));

    if (want != h.header_crc32) return false;

    for (uint16_t i = 0; i < h.file_count; ++i) {
        const NaiSlotFileEntry &e = h.files[i];

        if (e.name[0] == '\0' ||
            e.name[NAI_SLOT_NAME_BYTES - 1] != '\0') return false;

        if (e.offset < NAI_SLOT_HEADER_BYTES ||
            e.offset >= NAI_SLOT_BYTES ||
            e.size == 0 ||
            e.size > NAI_SLOT_BYTES - e.offset) return false;
    }

    return true;
}

bool file_duplicate(const NaiSlotHeader &h, const char *name)
{
    for (uint16_t i = 0; i < h.file_count; ++i) {
        if (strcmp(h.files[i].name, name) == 0) return true;
    }
    return false;
}

} // namespace

uint32_t nai_crc32_update(uint32_t crc, const uint8_t *data, size_t len)
{
    while (len--) {
        crc ^= *data++;

        for (uint8_t k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^
                  (0xEDB88320u & (0u - (crc & 1u)));
        }
    }

    return crc;
}

uint32_t nai_crc32(const void *data, size_t len)
{
    uint32_t crc = 0xFFFFFFFFu;
    crc = nai_crc32_update(crc, (const uint8_t *)data, len);
    return crc ^ 0xFFFFFFFFu;
}

bool nai_slotfs_init()
{
    return nai_flash_init();
}

bool nai_slotfs_canonical_name_allowed(const char *name)
{
    if (!name) return false;

    if (strcmp(name, "cfg.bin") == 0 ||
        strcmp(name, "mean.bin") == 0 ||
        strcmp(name, "scale.bin") == 0) return true;

    if (strlen(name) != 7) return false;

    if ((name[0] != 'w' && name[0] != 'b' &&
         name[0] != 'm' && name[0] != 's') ||
        name[3] != '.' ||
        strcmp(name + 4, "bin") != 0) return false;

    if (name[1] < '0' || name[1] > '9' ||
        name[2] < '0' || name[2] > '9') return false;

    const int idx = (name[1] - '0') * 10 + (name[2] - '0');
    return idx >= 0 && idx < 8;
}

bool nai_slotfs_virtual_name(const char *virtual_name,
                             char *slot_out,
                             char canonical_out[NAI_SLOT_NAME_BYTES])
{
    if (!virtual_name || !slot_out || !canonical_out) return false;

    /* Future Noodle backend compatibility: a_foo.bin -> single slot. */
    if ((virtual_name[0] != 'a' && virtual_name[0] != 'b') ||
        virtual_name[1] != '_') return false;

    const char *canonical = virtual_name + 2;

    if (!nai_slotfs_canonical_name_allowed(canonical) ||
        strlen(canonical) >= NAI_SLOT_NAME_BYTES) return false;

    *slot_out = MODEL_SLOT;
    strncpy(canonical_out, canonical, NAI_SLOT_NAME_BYTES);
    canonical_out[NAI_SLOT_NAME_BYTES - 1] = '\0';
    return true;
}

bool nai_slotfs_read_header(char slot,
                            NaiSlotHeader *out,
                            bool allow_staged)
{
    if (slot != MODEL_SLOT || !out) return false;

    /* Cheap empty-model test: only read four bytes first. */
    char magic[4];

    if (!nai_flash_read_fresh(
            NAI_SLOT_A_BASE,
            magic,
            sizeof(magic))) return false;
    if (memcmp(magic, "NSLT", 4) != 0) return false;

    /*
     * Read directly into caller storage.  Do not allocate another
     * ~560-byte NaiSlotHeader on the task stack.
     */
    if (!nai_flash_read_fresh(
            NAI_SLOT_A_BASE,
            out,
            sizeof(*out))) return false;
    if (!header_structurally_valid(*out, allow_staged)) return false;

    return true;
}

bool nai_slotfs_lookup(char slot,
                       const char *canonical,
                       uint32_t *physical_addr,
                       uint32_t *size,
                       bool allow_staged)
{
    if (slot != MODEL_SLOT || !canonical) return false;

    if (!nai_slotfs_read_header(
            MODEL_SLOT,
            &s_header_scratch,
            allow_staged)) {
        return false;
    }

    for (uint16_t i = 0; i < s_header_scratch.file_count; ++i) {
        if (strcmp(s_header_scratch.files[i].name, canonical) == 0) {
            if (physical_addr) {
                *physical_addr =
                    NAI_SLOT_A_BASE + s_header_scratch.files[i].offset;
            }
            if (size) *size = s_header_scratch.files[i].size;
            return true;
        }
    }

    return false;
}

char nai_slotfs_active_slot()
{
    return nai_slotfs_read_header(
        MODEL_SLOT,
        &s_header_scratch,
        false) ? MODEL_SLOT : 0;
}

NaiSlotResult nai_slotfs_deploy_begin()
{
    if (s_deploy.active) {
        nai_slotfs_abort(false);
    }

    /* Region was already erased sector-by-sector by main.c. */
    s_deploy = DeployState{};
    s_deploy.active = true;
    s_deploy.payload_cursor = NAI_SLOT_HEADER_BYTES;

    memcpy(s_deploy.header.magic, "NSLT", 4);
    s_deploy.header.version = SLOT_VERSION;
    s_deploy.header.header_bytes = sizeof(NaiSlotHeader);
    s_deploy.header.generation = 1u;
    s_deploy.header.file_count = 0;
    s_deploy.header.payload_bytes = 0;
    s_deploy.header.commit_word = SLOT_ERASED;

    s_staged_enabled = false;
    return NAI_SLOT_OK;
}

bool nai_slotfs_deploy_active()
{
    return s_deploy.active;
}

bool nai_slotfs_file_active()
{
    return s_deploy.file_active;
}

NaiSlotResult nai_slotfs_file_begin(const char *canonical,
                                    uint32_t expected_size,
                                    uint32_t expected_crc)
{
    if (!s_deploy.active || s_deploy.file_active)
        return NAI_SLOT_ERR_STATE;

    if (!nai_slotfs_canonical_name_allowed(canonical))
        return NAI_SLOT_ERR_NAME;

    if (expected_size == 0)
        return NAI_SLOT_ERR_SIZE;

    if (s_deploy.header.file_count >= NAI_SLOT_MAX_FILES)
        return NAI_SLOT_ERR_SIZE;

    if (file_duplicate(s_deploy.header, canonical))
        return NAI_SLOT_ERR_DUPLICATE;

    const uint32_t start = align4(s_deploy.payload_cursor);

    if (start < NAI_SLOT_HEADER_BYTES ||
        expected_size > NAI_SLOT_BYTES - start) {
        return NAI_SLOT_ERR_SIZE;
    }

    const uint16_t idx = s_deploy.header.file_count;
    NaiSlotFileEntry &e = s_deploy.header.files[idx];

    memset(&e, 0, sizeof(e));
    strncpy(e.name, canonical, sizeof(e.name) - 1);
    e.offset = start;
    e.size = expected_size;
    e.crc32 = expected_crc;

    s_deploy.current_index = (int)idx;
    s_deploy.expected_size = expected_size;
    s_deploy.expected_crc = expected_crc;
    s_deploy.received = 0;
    s_deploy.running_crc = 0xFFFFFFFFu;
    s_deploy.file_active = true;

    return NAI_SLOT_OK;
}

NaiSlotResult nai_slotfs_file_write(const uint8_t *data, size_t len)
{
    if (!s_deploy.active ||
        !s_deploy.file_active ||
        s_deploy.current_index < 0) {
        return NAI_SLOT_ERR_STATE;
    }

    if (!data && len) return NAI_SLOT_ERR_FLASH;
    if (len == 0) return NAI_SLOT_OK;

    if (s_deploy.received + len > s_deploy.expected_size)
        return NAI_SLOT_ERR_COUNT;

    const NaiSlotFileEntry &e =
        s_deploy.header.files[s_deploy.current_index];

    const uint32_t addr =
        NAI_SLOT_A_BASE + e.offset + s_deploy.received;

    if (!nai_flash_write(addr, data, len))
        return NAI_SLOT_ERR_FLASH;

    s_deploy.running_crc =
        nai_crc32_update(s_deploy.running_crc, data, len);

    s_deploy.received += (uint32_t)len;
    return NAI_SLOT_OK;
}

NaiSlotResult nai_slotfs_file_end()
{
    if (!s_deploy.active ||
        !s_deploy.file_active ||
        s_deploy.current_index < 0) {
        return NAI_SLOT_ERR_STATE;
    }

    if (s_deploy.received != s_deploy.expected_size) {
        s_deploy.file_active = false;
        s_deploy.current_index = -1;
        return NAI_SLOT_ERR_COUNT;
    }

    const uint32_t incoming_crc =
        s_deploy.running_crc ^ 0xFFFFFFFFu;

    if (incoming_crc != s_deploy.expected_crc) {
        s_deploy.file_active = false;
        s_deploy.current_index = -1;
        return NAI_SLOT_ERR_CRC;
    }

    /*
     * Real flash verification:
     * read the just-programmed payload back through XIP in tiny chunks
     * and recompute CRC. No giant stack buffer.
     */
    const NaiSlotFileEntry &e =
        s_deploy.header.files[s_deploy.current_index];

    uint8_t verify_buf[128];
    uint32_t verified = 0;
    uint32_t flash_crc = 0xFFFFFFFFu;

    while (verified < e.size) {
        uint32_t n = e.size - verified;
        if (n > sizeof(verify_buf)) n = sizeof(verify_buf);

        if (!nai_flash_read(
                NAI_SLOT_A_BASE + e.offset + verified,
                verify_buf,
                n)) {
            s_deploy.file_active = false;
            s_deploy.current_index = -1;
            return NAI_SLOT_ERR_FLASH;
        }

        flash_crc = nai_crc32_update(flash_crc, verify_buf, n);
        verified += n;
    }

    flash_crc ^= 0xFFFFFFFFu;

    if (flash_crc != s_deploy.expected_crc) {
        s_deploy.file_active = false;
        s_deploy.current_index = -1;
        return NAI_SLOT_ERR_CRC;
    }

    s_deploy.header.file_count++;
    s_deploy.payload_cursor = align4(e.offset + e.size);
    s_deploy.header.payload_bytes =
        s_deploy.payload_cursor - NAI_SLOT_HEADER_BYTES;

    s_deploy.file_active = false;
    s_deploy.current_index = -1;
    s_deploy.expected_size = 0;
    s_deploy.expected_crc = 0;
    s_deploy.received = 0;
    s_deploy.running_crc = 0xFFFFFFFFu;

    return NAI_SLOT_OK;
}

NaiSlotResult nai_slotfs_commit_begin()
{
    if (!s_deploy.active ||
        s_deploy.file_active ||
        s_deploy.header.file_count == 0 ||
        s_commit_phase != COMMIT_IDLE) {
        return NAI_SLOT_ERR_STATE;
    }

    /*
     * Header CRC deliberately excludes header_crc32 and commit_word.
     * The final commit word can therefore change from erased -> 0 without
     * changing the CRC.
     */
    s_deploy.header.header_crc32 =
        nai_crc32(
            &s_deploy.header,
            offsetof(NaiSlotHeader, header_crc32));

    s_deploy.header.commit_word = SLOT_ERASED;

    s_commit_offset = 0;
    s_commit_phase = COMMIT_WRITE_HEADER;
    s_staged_enabled = false;

    return NAI_SLOT_OK;
}

NaiSlotResult nai_slotfs_commit_step(bool *done)
{
    if (done) {
        *done = false;
    }

    if (!s_deploy.active || s_deploy.file_active) {
        s_commit_phase = COMMIT_IDLE;
        return NAI_SLOT_ERR_STATE;
    }

    switch (s_commit_phase) {

    case COMMIT_WRITE_HEADER: {
        /*
         * Do NOT write the commit_word yet.
         * Write no more than 128 bytes per app-loop iteration.
         */
        const uint32_t header_bytes =
            (uint32_t)offsetof(NaiSlotHeader, commit_word);

        if (s_commit_offset >= header_bytes) {
            s_commit_phase = COMMIT_VERIFY_STAGED;
            return NAI_SLOT_OK;
        }

        uint32_t n = header_bytes - s_commit_offset;
        if (n > COMMIT_CHUNK_BYTES) {
            n = COMMIT_CHUNK_BYTES;
        }

        const uint8_t *src =
            reinterpret_cast<const uint8_t *>(&s_deploy.header);

        if (!nai_flash_write(
                NAI_SLOT_A_BASE + s_commit_offset,
                src + s_commit_offset,
                n)) {
            s_commit_phase = COMMIT_IDLE;
            return NAI_SLOT_ERR_FLASH;
        }

        s_commit_offset += n;
        return NAI_SLOT_OK;
    }

    case COMMIT_VERIFY_STAGED: {
        /*
         * commit_word is still erased in flash, which is a valid staged header.
         * XIP read is fast and we return immediately afterward.
         */
        if (!nai_slotfs_read_header(
                MODEL_SLOT,
                &s_header_scratch,
                true)) {
            s_commit_phase = COMMIT_IDLE;
            return NAI_SLOT_ERR_HEADER;
        }

        s_staged_enabled = true;
        s_commit_phase = COMMIT_WRITE_WORD;
        return NAI_SLOT_OK;
    }

    case COMMIT_WRITE_WORD: {
        const uint32_t commit_addr =
            NAI_SLOT_A_BASE + offsetof(NaiSlotHeader, commit_word);

        const uint32_t committed = SLOT_COMMITTED;

        if (!nai_flash_write(
                commit_addr,
                &committed,
                sizeof(committed))) {
            s_commit_phase = COMMIT_IDLE;
            return NAI_SLOT_ERR_FLASH;
        }

        s_commit_phase = COMMIT_VERIFY_FINAL;
        return NAI_SLOT_OK;
    }

    case COMMIT_VERIFY_FINAL: {
        if (!nai_slotfs_read_header(
                MODEL_SLOT,
                &s_header_scratch,
                false)) {
            s_commit_phase = COMMIT_IDLE;
            return NAI_SLOT_ERR_HEADER;
        }

        s_deploy = DeployState{};
        s_staged_enabled = false;
        s_commit_phase = COMMIT_IDLE;
        s_commit_offset = 0;

        if (done) {
            *done = true;
        }

        return NAI_SLOT_OK;
    }

    case COMMIT_IDLE:
    default:
        return NAI_SLOT_ERR_STATE;
    }
}

NaiSlotResult nai_slotfs_stage_header()
{
    if (!s_deploy.active ||
        s_deploy.file_active ||
        s_deploy.header.file_count == 0) {
        return NAI_SLOT_ERR_STATE;
    }

    s_deploy.header.header_crc32 =
        nai_crc32(
            &s_deploy.header,
            offsetof(NaiSlotHeader, header_crc32));

    s_deploy.header.commit_word = SLOT_ERASED;

    if (!nai_flash_write(
            NAI_SLOT_A_BASE,
            &s_deploy.header,
            sizeof(s_deploy.header))) {
        return NAI_SLOT_ERR_FLASH;
    }

    if (!nai_slotfs_read_header(
            MODEL_SLOT,
            &s_header_scratch,
            true))
        return NAI_SLOT_ERR_HEADER;

    s_staged_enabled = true;
    return NAI_SLOT_OK;
}

NaiSlotResult nai_slotfs_activate_staged()
{
    if (!s_deploy.active ||
        s_deploy.file_active ||
        !s_staged_enabled) {
        return NAI_SLOT_ERR_STATE;
    }

    const uint32_t commit_addr =
        NAI_SLOT_A_BASE + offsetof(NaiSlotHeader, commit_word);

    const uint32_t committed = SLOT_COMMITTED;

    if (!nai_flash_write(
            commit_addr,
            &committed,
            sizeof(committed))) {
        return NAI_SLOT_ERR_FLASH;
    }

    if (!nai_slotfs_read_header(
            MODEL_SLOT,
            &s_header_scratch,
            false))
        return NAI_SLOT_ERR_HEADER;

    s_deploy = DeployState{};
    s_staged_enabled = false;
    return NAI_SLOT_OK;
}

void nai_slotfs_abort(bool erase_target)
{
    const bool was_active = s_deploy.active;

    s_deploy = DeployState{};
    s_staged_enabled = false;

    (void)erase_target;
    (void)was_active;
}

void nai_slotfs_set_staged_read(char slot, bool enable)
{
    s_staged_enabled = (slot == MODEL_SLOT) && enable;
}

bool nai_slotfs_staged_read_allowed(char slot)
{
    return slot == MODEL_SLOT && s_staged_enabled;
}
