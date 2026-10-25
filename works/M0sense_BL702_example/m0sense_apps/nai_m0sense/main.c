#include "bflb_platform.h"
#include "hal_gpio.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <FreeRTOS.h>
#include "task.h"

#include "bluetooth.h"
#include "conn.h"
#include "gatt.h"
#include "gap.h"
#include "bl702_glb.h"
#include "ble_lib_api.h"
#include "hci_driver.h"
#include "bl702_sec_eng.h"

#include "io_def.h"
#include "nai_imu_bridge.h"
#include "nai_storage_bridge.h"
#include "nai_infer_bridge.h"

/*
 * NoodleAI Mode 2 v0.4.5 INT8 + cfg-defined IMU filtering
 * Continuous Gesture Mode
 *
 * What this firmware does:
 *   - real QMI8658 sampling at 50 Hz
 *   - continuous BLE IMU streaming
 *   - raw six-axis rolling window; cfg.bin selects model representation
 *   - model-defined inference stride and confidence threshold
 *   - one-slot BLE model deployment with per-file CRC verification
 *
 * What this firmware intentionally does NOT do yet:
 *   - no BOOT button
 *   - no gesture start/end trigger
 *   - no A/B slots (one model region only)
 *   - no application queues
 *
 * The Bouffalo BLE stack still uses FreeRTOS underneath, but our application
 * remains one boring polling/state-machine task.
 */

/* -------------------------------------------------------------------------- */
/* FreeRTOS heap                                                               */
/* -------------------------------------------------------------------------- */

extern uint8_t _heap_start;
extern uint8_t _heap_size;

static HeapRegion_t xHeapRegions[] = {
    { &_heap_start, (unsigned int)&_heap_size },
    { NULL, 0 },
    { NULL, 0 }
};

/* -------------------------------------------------------------------------- */
/* NoodleAI UUID                                                               */
/* -------------------------------------------------------------------------- */

/* 7f8b0001-5f5b-4f4a-a5d5-2e889aa10001 */
#define BT_UUID_NAI_SERVICE \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0001, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

/* 7f8b0002-5f5b-4f4a-a5d5-2e889aa10001 */
#define BT_UUID_NAI_IMU \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0002, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

static volatile int s_imu_notify_enabled = 0;
static volatile int s_status_notify_enabled = 0;
static volatile int s_status_notify_just_enabled = 0;

/* One pending CONTROL command. Web protocol sends a command then waits. */
#define NAI_CONTROL_MAX 32
static uint8_t s_control_buf[NAI_CONTROL_MAX];
static volatile uint16_t s_control_len = 0;
static volatile int s_control_pending = 0;

/* Deployment state. Model chunks are written directly through the flash HAL. */
static volatile int s_deploy_active = 0;
static volatile int s_file_active = 0;
static volatile int s_model_write_error = 0;
static char s_current_file[16] = {0};

static volatile int s_erase_active = 0;
static uint32_t s_erase_offset = 0;

/*
 * Commit is also incremental.  Never do several flash page programs
 * back-to-back while BLE is alive.
 */
static volatile int s_commit_active = 0;

/* Mode 2 runtime command state is simple enough to keep as one flag. */
static int s_mode2_inference_enabled = 0;


enum {
    OP_SET_TRAINING     = 0x01,
    OP_SET_INFERENCE    = 0x02,
    OP_DEPLOY_BEGIN     = 0x20,
    OP_FILE_BEGIN       = 0x21,
    OP_FILE_END         = 0x22,
    OP_DEPLOY_COMMIT    = 0x23,
    OP_DEPLOY_ABORT     = 0x24,
    OP_GET_CAPABILITIES = 0x25,
};

static void imu_ccc_changed(
    const struct bt_gatt_attr *attr,
    u16_t value)
{
    (void)attr;

    s_imu_notify_enabled = (value == BT_GATT_CCC_NOTIFY);
}

/* 7f8b0003-5f5b-4f4a-a5d5-2e889aa10001 */
#define BT_UUID_NAI_CONTROL \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0003, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

static int control_write(
    struct bt_conn *conn,
    const struct bt_gatt_attr *attr,
    const void *buf,
    u16_t len,
    u16_t offset,
    u8_t flags)
{
    (void)conn;
    (void)attr;
    (void)flags;

    if (offset != 0 || !buf || len == 0 || len > NAI_CONTROL_MAX) {
        return 0;
    }

    /*
     * Callback rule: copy bytes, set flag, return.
     * Erase / header work happens in the one boring app loop.
     */
    if (s_control_pending) {
        return 0;
    }

    memcpy(s_control_buf, buf, len);
    s_control_len = len;
    s_control_pending = 1;
    return len;
}

/* 7f8b0004-5f5b-4f4a-a5d5-2e889aa10001 */
#define BT_UUID_NAI_MODEL \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0004, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

static int model_write(
    struct bt_conn *conn,
    const struct bt_gatt_attr *attr,
    const void *buf,
    u16_t len,
    u16_t offset,
    u8_t flags)
{
    (void)conn;
    (void)attr;
    (void)flags;

    if (offset != 0 || !buf || len == 0 ||
        !s_deploy_active || !s_file_active) {
        return 0;
    }

    /*
     * Small BLE chunks (default 160 B) are programmed synchronously.
     * hal_flash's flash_write() is TCM-resident and manages XIP/IRQs.
     *
     * If a write fails, remember the error; FILE_END will report it.
     */
    if (!s_model_write_error) {
        const int rc = nai_storage_file_write(
            (const uint8_t *)buf, (size_t)len);

        if (rc != 0) {
            s_model_write_error = rc;
        }
    }

    return len;
}

/* 7f8b0005-5f5b-4f4a-a5d5-2e889aa10001 */
#define BT_UUID_NAI_STATUS \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0005, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

static int status_read(
    struct bt_conn *conn,
    const struct bt_gatt_attr *attr,
    void *buf,
    u16_t len,
    u16_t offset)
{
    (void)conn;
    (void)attr;

    const char *text =
        s_mode2_inference_enabled ? "MODE:I" : "MODE:T";
    const u16_t total = (u16_t)strlen(text);

    if (offset >= total) {
        return 0;
    }

    u16_t n = total - offset;
    if (n > len) {
        n = len;
    }

    memcpy(buf, text + offset, n);
    return n;
}

static void status_ccc_changed(
    const struct bt_gatt_attr *attr,
    u16_t value)
{
    (void)attr;
    s_status_notify_enabled = (value == BT_GATT_CCC_NOTIFY);
    if (s_status_notify_enabled) {
        s_status_notify_just_enabled = 1;
    }
}


/* -------------------------------------------------------------------------- */
/* Continuous IMU packet                                                  */
/* -------------------------------------------------------------------------- */

/*
 * This matches the already-proven NoodleAI Web packet exactly:
 *
 *   layout: <IB3x30f
 *
 *   uint32 t0_ms
 *   uint8  count
 *   3 bytes padding
 *   5 samples * 6 float32 = ax ay az gx gy gz
 *
 * Total = 128 bytes.
 */
#define NAI_IMU_BATCH_SIZE 5

#pragma pack(push, 1)
struct NaiImuSample {
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
};

struct NaiImuPacket {
    uint32_t t0_ms;
    uint8_t count;
    uint8_t reserved[3];
    struct NaiImuSample samples[NAI_IMU_BATCH_SIZE];
};
#pragma pack(pop)

_Static_assert(sizeof(struct NaiImuPacket) == 128,
               "NaiImuPacket must be exactly 128 bytes");

static struct bt_conn *s_conn = NULL;
static void connected(struct bt_conn *conn, u8_t err)
{
    if (err) {
        return;
    }

    s_conn = conn;
}

static void disconnected(struct bt_conn *conn, u8_t reason)
{
    (void)conn;
    (void)reason;

    s_conn = NULL;
    s_imu_notify_enabled = 0;
    s_status_notify_enabled = 0;
    s_status_notify_just_enabled = 0;

    /*
     * Do not try to erase flash from the BLE callback.
     * An interrupted deployment remains uncommitted and therefore invalid.
     */
    s_erase_active = 0;
    s_erase_offset = 0;
    s_commit_active = 0;
    s_deploy_active = 0;
    s_file_active = 0;
}

static struct bt_conn_cb nai_conn_callbacks = {
    .connected = connected,
    .disconnected = disconnected,
};

/*
 * Keep the Web-compatible NoodleAI GATT table unchanged.
 */
static struct bt_gatt_attr nai_attrs[] = {
    BT_GATT_PRIMARY_SERVICE(BT_UUID_NAI_SERVICE),

    BT_GATT_CHARACTERISTIC(
        BT_UUID_NAI_IMU,
        BT_GATT_CHRC_NOTIFY,
        0,
        NULL,
        NULL,
        NULL),

    BT_GATT_CCC(
        imu_ccc_changed,
        BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),

    BT_GATT_CHARACTERISTIC(
        BT_UUID_NAI_CONTROL,
        BT_GATT_CHRC_WRITE | BT_GATT_CHRC_WRITE_WITHOUT_RESP,
        BT_GATT_PERM_WRITE,
        NULL,
        control_write,
        NULL),

    BT_GATT_CHARACTERISTIC(
        BT_UUID_NAI_MODEL,
        BT_GATT_CHRC_WRITE | BT_GATT_CHRC_WRITE_WITHOUT_RESP,
        BT_GATT_PERM_WRITE,
        NULL,
        model_write,
        NULL),

    BT_GATT_CHARACTERISTIC(
        BT_UUID_NAI_STATUS,
        BT_GATT_CHRC_READ | BT_GATT_CHRC_NOTIFY,
        BT_GATT_PERM_READ,
        status_read,
        NULL,
        NULL),

    BT_GATT_CCC(
        status_ccc_changed,
        BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
};

static struct bt_gatt_service nai_service = BT_GATT_SERVICE(nai_attrs);


static int notify_imu_packet(const struct NaiImuPacket *packet)
{
    if (!packet || !s_conn || !s_imu_notify_enabled) {
        return 0;
    }

    /*
     * Attribute 2 is the IMU VALUE:
     *   0 service
     *   1 IMU declaration
     *   2 IMU value
     *   3 IMU CCC
     */
    return bt_gatt_notify(
        s_conn,
        &nai_attrs[2],
        packet,
        sizeof(*packet));
}


static int notify_status(const char *text)
{
    if (!text || !s_conn || !s_status_notify_enabled) {
        return 0;
    }

    /* STATUS value attribute index = 9. */
    return bt_gatt_notify(
        s_conn,
        &nai_attrs[9],
        text,
        strlen(text));
}

static uint32_t read_u32_le(const uint8_t *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void mode2_reset_ring(void);
static bool mode2_reload_model(void);
static void mode2_clear_model(void);
static bool mode2_model_ready(void);
static void mode2_notify_model_info(void);
static void mode2_notify_load_diag(void);

static void deploy_error(const char *what, int rc)
{
    char msg[40];
    snprintf(msg, sizeof(msg), "ERR:%s:%d", what, rc);
    (void)notify_status(msg);
}

/*
 * Handle exactly one queued CONTROL command.
 *
 * Deployment protocol:
 *   0x20 DEPLOY_BEGIN
 *   0x21 FILE_BEGIN [nameLen][size:u32][crc:u32][name]
 *   MODEL characteristic carries the raw file bytes
 *   0x22 FILE_END
 *   0x23 DEPLOY_COMMIT
 *   0x24 DEPLOY_ABORT
 */
static void process_deploy_erase_step(void)
{
    if (!s_erase_active) return;

    const uint32_t region_bytes = nai_storage_region_bytes();

    if (s_erase_offset >= region_bytes) {
        const int rc = nai_storage_deploy_begin();
        s_erase_active = 0;

        if (rc == 0) {
            s_deploy_active = 1;
            s_file_active = 0;
            s_model_write_error = 0;
            s_current_file[0] = '\0';
            (void)notify_status("DEPLOY:SINGLE");
        } else {
            s_deploy_active = 0;
            deploy_error("BEGIN", rc);
        }
        return;
    }

    const int rc = nai_storage_erase_sector(s_erase_offset);
    if (rc != 0) {
        s_erase_active = 0;
        s_deploy_active = 0;
        deploy_error("ERASE", rc);
        return;
    }

    s_erase_offset += 4096u;
}

static void process_deploy_commit_step(void)
{
    if (!s_commit_active) {
        return;
    }

    int done = 0;
    const int rc = nai_storage_commit_step(&done);

    if (rc != 0) {
        s_commit_active = 0;
        deploy_error("COMMIT", rc);
        return;
    }

    if (done) {
        s_commit_active = 0;
        s_deploy_active = 0;

        /*
         * Storage is durably committed first, so the browser's deployment
         * transaction can finish immediately. Then load the just-stored model
         * through the exact same single-slot Noodle backend used after reboot.
         */
        (void)notify_status("MODEL_STORED");

        if (mode2_reload_model()) {
            mode2_notify_model_info();
        } else {
            (void)notify_status("ERR:MODEL_LOAD");
            mode2_notify_load_diag();
        }
    }
}

static void process_control_command(void)
{
    if (!s_control_pending) {
        return;
    }

    uint8_t data[NAI_CONTROL_MAX];
    uint16_t len = s_control_len;

    if (len > NAI_CONTROL_MAX) {
        len = NAI_CONTROL_MAX;
    }

    memcpy(data, s_control_buf, len);
    s_control_pending = 0;
    s_control_len = 0;

    if (len == 0) {
        return;
    }

    const uint8_t op = data[0];

    switch (op) {
    case OP_SET_TRAINING:
        s_mode2_inference_enabled = 0;
        mode2_reset_ring();
        (void)notify_status("MODE:T");
        break;

    case OP_SET_INFERENCE:
        if (!mode2_model_ready() && !mode2_reload_model()) {
            s_mode2_inference_enabled = 0;
            (void)notify_status("NO_MODEL");
            mode2_notify_load_diag();
            break;
        }
        s_mode2_inference_enabled = 1;
        mode2_reset_ring();
        (void)notify_status("MODE:I");
        mode2_notify_model_info();
        break;

    case OP_GET_CAPABILITIES:
        /* Non-destructive response, before browser issues DEPLOY_BEGIN. */
        (void)notify_status("CAPS:PACKED_INT8:1");
        break;

    case OP_DEPLOY_BEGIN: {
        /* Single slot: erase means the old runtime model must be dropped too. */
        s_mode2_inference_enabled = 0;
        mode2_clear_model();
        nai_storage_abort(0);
        s_deploy_active = 0;
        s_file_active = 0;
        s_model_write_error = 0;
        s_current_file[0] = '\0';

        s_erase_offset = 0;
        s_erase_active = 1;
        (void)notify_status("DEPLOY:ERASING");
        break;
    }

    case OP_FILE_BEGIN: {
        if (!s_deploy_active || len < 10) {
            deploy_error("FILE_BEGIN", -1);
            break;
        }

        const uint8_t name_len = data[1];

        if (name_len == 0 || name_len > 15 ||
            len != (uint16_t)(10u + name_len)) {
            deploy_error("FILE_BEGIN", -2);
            break;
        }

        char name[16] = {0};
        memcpy(name, &data[10], name_len);

        const uint32_t size = read_u32_le(&data[2]);
        const uint32_t crc  = read_u32_le(&data[6]);

        const int rc = nai_storage_file_begin(name, size, crc);

        if (rc == 0) {
            strncpy(s_current_file, name, sizeof(s_current_file) - 1);
            s_current_file[sizeof(s_current_file) - 1] = '\0';
            s_model_write_error = 0;
            s_file_active = 1;

            char msg[40];
            snprintf(msg, sizeof(msg), "FILE_READY:%s", name);
            (void)notify_status(msg);
        } else {
            deploy_error("FILE_BEGIN", rc);
        }
        break;
    }

    case OP_FILE_END: {
        if (!s_deploy_active || !s_file_active) {
            deploy_error("FILE_END", -1);
            break;
        }

        if (s_model_write_error) {
            const int rc = s_model_write_error;
            s_file_active = 0;
            deploy_error("WRITE", rc);
            break;
        }

        const int rc = nai_storage_file_end();

        if (rc == 0) {
            s_file_active = 0;

            char msg[40];
            snprintf(msg, sizeof(msg), "FILE_OK:%s", s_current_file);
            (void)notify_status(msg);
            s_current_file[0] = '\0';
        } else {
            s_file_active = 0;
            deploy_error("FILE_END", rc);
        }
        break;
    }

    case OP_DEPLOY_COMMIT: {
        if (!s_deploy_active || s_file_active || s_commit_active) {
            deploy_error("COMMIT", -1);
            break;
        }

        const int rc = nai_storage_commit_begin();

        if (rc == 0) {
            s_commit_active = 1;
            (void)notify_status("COMMIT:START");
        } else {
            deploy_error("COMMIT", rc);
        }
        break;
    }

    case OP_DEPLOY_ABORT:
        nai_storage_abort(0);
        s_erase_active = 0;
        s_erase_offset = 0;
        s_commit_active = 0;
        s_deploy_active = 0;
        s_file_active = 0;
        s_model_write_error = 0;
        s_current_file[0] = '\0';
        (void)notify_status("DEPLOY:ABORTED");
        break;

    default:
        break;
    }
}

/* -------------------------------------------------------------------------- */
/* Mode 2 model + rolling raw-IMU window                                      */
/* -------------------------------------------------------------------------- */

/*
 * The sensor side is deliberately fixed and boring: always collect the raw
 * QMI8658 six-axis stream at 50 Hz. cfg.bin decides what Noodle sees.
 *
 * The representation bridge converts each complete chronological raw window
 * into one of the six NAI4 representations before StandardScaler + INT8 MLP.
 * Since MAX_DIM is 768 and the smallest representation has 3 channels, the
 * largest legal model window is 256 samples. The raw ring therefore needs
 * 256 x 6 floats regardless of the deployed model representation.
 */
#define MODE2_SAMPLE_HZ             50u
#define MODE2_RAW_CHANNELS           6u
#define MODE2_MAX_INPUT_DIM        768u
#define MODE2_MIN_REPR_CHANNELS      3u
#define MODE2_MAX_WINDOW_SAMPLES \
    (MODE2_MAX_INPUT_DIM / MODE2_MIN_REPR_CHANNELS)
#define MODE2_MAX_RAW_FLOATS \
    (MODE2_MAX_WINDOW_SAMPLES * MODE2_RAW_CHANNELS)

static float s_mode2_ring[MODE2_MAX_RAW_FLOATS];
static float s_mode2_raw_window[MODE2_MAX_RAW_FLOATS];

static uint16_t s_mode2_window_samples = 0;
static uint16_t s_mode2_input_dim = 0;
static uint16_t s_mode2_head = 0;       /* next raw sample position to write */
static uint16_t s_mode2_count = 0;      /* valid raw sample count */
static uint16_t s_mode2_stride_count = 0;
static uint16_t s_mode2_stride_samples = 5;
static int s_mode2_model_ready = 0;
static int s_mode2_last_load_rc = -10;

static void mode2_reset_ring(void)
{
    s_mode2_head = 0;
    s_mode2_count = 0;
    s_mode2_stride_count = 0;
}

static void mode2_clear_model(void)
{
    nai_infer_clear();
    s_mode2_model_ready = 0;
    s_mode2_window_samples = 0;
    s_mode2_input_dim = 0;
    s_mode2_stride_samples = 5;
    mode2_reset_ring();
}

static bool mode2_model_ready(void)
{
    return s_mode2_model_ready && nai_infer_loaded();
}

static bool mode2_reload_model(void)
{
    mode2_clear_model();

    if (!nai_storage_model_valid()) {
        s_mode2_last_load_rc = -10;
        return false;
    }

    const int load_rc = nai_infer_load_active();
    if (load_rc != 0) {
        s_mode2_last_load_rc = load_rc;
        mode2_clear_model();
        return false;
    }

    const uint16_t n = nai_infer_window_length();
    const uint16_t ch = nai_infer_channel_count();
    const uint16_t dim = nai_infer_input_dim();
    const uint16_t stride = nai_infer_stride_samples();
    const uint16_t rep = nai_infer_representation();

    if (nai_infer_sample_rate_hz() != MODE2_SAMPLE_HZ ||
        rep < 1u || rep > 6u ||
        ch < MODE2_MIN_REPR_CHANNELS || ch > 7u ||
        n < 2u || n > MODE2_MAX_WINDOW_SAMPLES ||
        dim == 0 || dim > MODE2_MAX_INPUT_DIM ||
        dim != (uint32_t)n * ch ||
        stride == 0 || stride > n) {
        s_mode2_last_load_rc = -18;
        mode2_clear_model();
        return false;
    }

    s_mode2_last_load_rc = 0;
    s_mode2_window_samples = n;
    s_mode2_input_dim = dim;
    s_mode2_stride_samples = stride;
    s_mode2_model_ready = 1;
    mode2_reset_ring();
    return true;
}

static void mode2_notify_model_info(void)
{
    if (!mode2_model_ready()) return;

    float threshold = nai_infer_confidence_threshold();
    if (threshold < 0.0f) threshold = 0.0f;
    if (threshold > 1.0f) threshold = 1.0f;
    const uint16_t threshold_milli =
        (uint16_t)(threshold * 1000.0f + 0.5f);

    char msg[64];
    /* Keep the v0.5 browser-compatible MODEL:READY wire format unchanged. */
    snprintf(msg, sizeof(msg), "MODEL:READY:%u:%u:%u:%u:%u",
             (unsigned)s_mode2_window_samples,
             (unsigned)s_mode2_input_dim,
             (unsigned)nai_infer_class_count(),
             (unsigned)s_mode2_stride_samples,
             (unsigned)threshold_milli);
    (void)notify_status(msg);

    for (uint16_t i = 0; i < nai_infer_class_count(); ++i) {
        const char *label = nai_infer_label(i);
        if (!label || !label[0]) continue;
        snprintf(msg, sizeof(msg), "L:%u:%s", (unsigned)i, label);
        (void)notify_status(msg);
    }
}


static void mode2_notify_load_diag(void)
{
    char msg[32];
    snprintf(msg, sizeof(msg), "LOADRC:%d", s_mode2_last_load_rc);
    (void)notify_status(msg);
}


static bool mode2_push_sample(const float s[MODE2_RAW_CHANNELS])
{
    if (!mode2_model_ready()) return false;

    const uint16_t base =
        (uint16_t)(s_mode2_head * MODE2_RAW_CHANNELS);

    memcpy(&s_mode2_ring[base],
           s,
           sizeof(float) * MODE2_RAW_CHANNELS);

    s_mode2_head =
        (uint16_t)((s_mode2_head + 1u) % s_mode2_window_samples);

    if (s_mode2_count < s_mode2_window_samples) {
        ++s_mode2_count;

        if (s_mode2_count == s_mode2_window_samples) {
            s_mode2_stride_count = 0;
            return true;
        }

        return false;
    }

    ++s_mode2_stride_count;
    if (s_mode2_stride_count >= s_mode2_stride_samples) {
        s_mode2_stride_count = 0;
        return true;
    }

    return false;
}

static bool mode2_copy_raw_window(void)
{
    if (!mode2_model_ready() ||
        s_mode2_count < s_mode2_window_samples) {
        return false;
    }

    uint16_t src = s_mode2_head; /* oldest sample when ring is full */
    size_t out = 0;

    for (uint16_t i = 0; i < s_mode2_window_samples; ++i) {
        const uint16_t base =
            (uint16_t)(src * MODE2_RAW_CHANNELS);
        for (uint16_t c = 0; c < MODE2_RAW_CHANNELS; ++c) {
            s_mode2_raw_window[out++] = s_mode2_ring[base + c];
        }
        src = (uint16_t)((src + 1u) % s_mode2_window_samples);
    }

    return out == (size_t)s_mode2_window_samples * MODE2_RAW_CHANNELS;
}

static void mode2_run_prediction(void)
{
    if (!s_mode2_inference_enabled || !mode2_copy_raw_window()) {
        return;
    }

    uint16_t cls = 0;
    float confidence = 0.0f;
    const int rc = nai_infer_predict_raw6(
        s_mode2_raw_window,
        s_mode2_window_samples,
        &cls,
        &confidence);

    if (rc != 0) {
        (void)notify_status("ERR:PRED");
        return;
    }

    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;
    const uint16_t milli =
        (uint16_t)(confidence * 1000.0f + 0.5f);

    char msg[40];
    const float threshold = nai_infer_confidence_threshold();
    if (confidence + 1e-6f < threshold) {
        snprintf(msg, sizeof(msg), "U:%u.%03u",
                 (unsigned)(milli / 1000u),
                 (unsigned)(milli % 1000u));
    } else {
        snprintf(msg, sizeof(msg), "P:%u:%u.%03u",
                 (unsigned)cls,
                 (unsigned)(milli / 1000u),
                 (unsigned)(milli % 1000u));
    }
    (void)notify_status(msg);
}

/* -------------------------------------------------------------------------- */
/* BLE                                                                         */
/* -------------------------------------------------------------------------- */

static int start_advertising(void)
{
    static const char name[] = "NoodleAI";

    struct bt_le_adv_param p = {
        .options = BT_LE_ADV_OPT_CONNECTABLE,
        .interval_min = BT_GAP_ADV_FAST_INT_MIN_3,
        .interval_max = BT_GAP_ADV_FAST_INT_MAX_3,
    };

    /*
     * 31-byte legacy advertising packet:
     *   flags       = 3 bytes
     *   128-bit UUID= 18 bytes
     *   "NoodleAI"  = 10 bytes
     * total         = 31 bytes exactly
     */
    struct bt_data ad[] = {
        BT_DATA_BYTES(
            BT_DATA_FLAGS,
            BT_LE_AD_NO_BREDR | BT_LE_AD_GENERAL
        ),

        BT_DATA_BYTES(
            BT_DATA_UUID128_ALL,
            BT_UUID_128_ENCODE(
                0x7f8b0001,
                0x5f5b,
                0x4f4a,
                0xa5d5,
                0x2e889aa10001
            )
        ),

        BT_DATA(
            BT_DATA_NAME_COMPLETE,
            name,
            sizeof(name) - 1
        ),
    };

    return bt_le_adv_start(
        &p,
        ad,
        sizeof(ad) / sizeof(ad[0]),
        NULL,
        0
    );
}

static void bluetooth_ready(int err)
{
    if (err) {
        gpio_write(LED_G_PIN, 1);
        gpio_write(LED_R_PIN, 0);
        return;
    }

    bt_conn_cb_register(&nai_conn_callbacks);
    bt_gatt_service_register(&nai_service);

    err = start_advertising();

    if (err == 0) {
        /* GREEN = BLE advertising started */
        gpio_write(LED_R_PIN, 1);
        gpio_write(LED_G_PIN, 0);
    } else {
        gpio_write(LED_G_PIN, 1);
        gpio_write(LED_R_PIN, 0);
    }
}

static void ble_init_task(void *arg)
{
    (void)arg;

    GLB_Set_EM_Sel(GLB_EM_8KB);
    ble_controller_init(configMAX_PRIORITIES - 1);
    hci_driver_init();
    bt_enable(bluetooth_ready);

    vTaskDelete(NULL);
}

/* -------------------------------------------------------------------------- */
/* Heartbeat                                                                   */
/* -------------------------------------------------------------------------- */

static void app_task(void *arg)
{
    (void)arg;

    uint32_t last_sample_ms = bflb_platform_get_time_ms();
    uint32_t last_blink_ms = last_sample_ms;

    int blue_on = 0;

    struct NaiImuPacket packet;
    memset(&packet, 0, sizeof(packet));
    uint8_t batch = 0;

    /*
     * Load the committed model after the scheduler starts. This keeps boot/BLE
     * bring-up simple and avoids doing Noodle/flash work before RTOS services.
     */
    (void)mode2_reload_model();

    for (;;) {
        const uint32_t now = bflb_platform_get_time_ms();

        /* -------------------------------------------------------------- */
        /* Heartbeat                                                      */
        /* -------------------------------------------------------------- */

        if ((uint32_t)(now - last_blink_ms) >= 500u) {
            last_blink_ms += 500u;
            blue_on = !blue_on;
            gpio_write(LED_B_PIN, blue_on ? 0 : 1);
        }

        /* -------------------------------------------------------------- */
        /* BLE control / deployment                                      */
        /* -------------------------------------------------------------- */

        process_control_command();
        process_deploy_erase_step();
        process_deploy_commit_step();

        if (s_status_notify_just_enabled) {
            s_status_notify_just_enabled = 0;
            if (mode2_model_ready()) {
                mode2_notify_model_info();
            } else {
                (void)notify_status("NO_MODEL");
                mode2_notify_load_diag();
            }
            (void)notify_status(
                s_mode2_inference_enabled ? "MODE:I" : "MODE:T");
        }

        /* -------------------------------------------------------------- */
        /* Continuous 50 Hz QMI8658                                      */
        /* -------------------------------------------------------------- */

        if ((uint32_t)(now - last_sample_ms) >= 20u) {
            last_sample_ms += 20u;

            float s[MODE2_RAW_CHANNELS];

            if (nai_imu_bridge_read(s)) {

                /*
                 * 1) Feed the model-defined rolling window. Training mode still
                 *    maintains the ring; only inference mode executes Noodle.
                 */
                const bool inference_due = mode2_push_sample(s);

                if (inference_due && s_mode2_inference_enabled) {
                    mode2_run_prediction();
                }

                /*
                 * 2) Independently keep streaming the same real IMU samples
                 *    to the browser for plotting and training.
                 */
                if (batch == 0) {
                    packet.t0_ms = now;
                }

                packet.samples[batch].ax = s[0];
                packet.samples[batch].ay = s[1];
                packet.samples[batch].az = s[2];
                packet.samples[batch].gx = s[3];
                packet.samples[batch].gy = s[4];
                packet.samples[batch].gz = s[5];

                ++batch;

                if (batch >= NAI_IMU_BATCH_SIZE) {
                    packet.count = batch;

                    /*
                     * During model upload, reserve BLE bandwidth for flash
                     * chunks. Sampling/ring-buffer maintenance still continues.
                     */
                    if (!s_deploy_active &&
                        !s_erase_active &&
                        !s_commit_active) {
                        (void)notify_imu_packet(&packet);
                    }

                    batch = 0;
                    packet.count = 0;
                }
            }
        }

        /*
         * The vendor BLE stack runs in its own FreeRTOS tasks.
         * Our application remains this one boring loop.
         */
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

/* -------------------------------------------------------------------------- */
/* Required FreeRTOS hooks                                                     */
/* -------------------------------------------------------------------------- */

void user_vAssertCalled(void)
    __attribute__((weak, alias("vAssertCalled")));

void vAssertCalled(void)
{
    gpio_write(LED_R_PIN, 0);
    for (;;) {}
}

void vApplicationTickHook(void)
{
}

void vApplicationStackOverflowHook(TaskHandle_t task, char *name)
{
    (void)task;
    (void)name;
    gpio_write(LED_R_PIN, 0);
    for (;;) {}
}

void vApplicationMallocFailedHook(void)
{
    gpio_write(LED_R_PIN, 0);
    for (;;) {}
}

void vApplicationGetIdleTaskMemory(
    StaticTask_t **t,
    StackType_t **s,
    uint32_t *n)
{
    static StaticTask_t tcb;
    static StackType_t stack[configMINIMAL_STACK_SIZE];

    *t = &tcb;
    *s = stack;
    *n = configMINIMAL_STACK_SIZE;
}

void vApplicationGetTimerTaskMemory(
    StaticTask_t **t,
    StackType_t **s,
    uint32_t *n)
{
    static StaticTask_t tcb;
    static StackType_t stack[configTIMER_TASK_STACK_DEPTH];

    *t = &tcb;
    *s = stack;
    *n = configTIMER_TASK_STACK_DEPTH;
}

/*
 * Keep the SDK's known-working tickless hook for this baseline.
 * It is not application logic. Once BLE is proven, we can disable tickless
 * in the FreeRTOS configuration and delete this entire function.
 */
extern uint64_t ullNextTime;
extern const size_t uxTimerIncrementsForOneTick;

void vApplicationSleep(uint32_t xExpectedIdleTime)
{
    uint64_t sleep_start_time = 0;
    uint64_t sleep_end_time = 0;
    uint64_t record_ullNextTime = 0;

    uint32_t actual_compensate_mstime = 0;
    static uint32_t actual_compensate_ustime = 0;

    if (eTaskConfirmSleepModeStatus() != eAbortSleep) {

        if (xExpectedIdleTime > 60 * 1000) {
            xExpectedIdleTime = 60 * 1000;
        }

        cpu_global_irq_disable();

        sleep_start_time = bflb_platform_get_time_us();

        ullNextTime -= uxTimerIncrementsForOneTick;
        record_ullNextTime = ullNextTime;

        ullNextTime +=
            uxTimerIncrementsForOneTick *
            (xExpectedIdleTime - 1);

        *(volatile uint64_t *)(0x02000000 + 0x4000) =
            ullNextTime;

        ullNextTime += uxTimerIncrementsForOneTick;

        __WFI();

        sleep_end_time = bflb_platform_get_time_us();

        if (xExpectedIdleTime ==
            pdMS_TO_TICKS(
                (sleep_end_time - sleep_start_time + 500) / 1000)) {

            vTaskStepTick(xExpectedIdleTime);

        } else {

            ullNextTime = record_ullNextTime;
            ullNextTime +=
                (sleep_end_time - sleep_start_time);

            *(volatile uint64_t *)(0x02000000 + 0x4000) =
                ullNextTime;

            ullNextTime += uxTimerIncrementsForOneTick;

            actual_compensate_ustime +=
                (uint32_t)(sleep_end_time - sleep_start_time);

            if (actual_compensate_ustime / 1000) {

                actual_compensate_mstime =
                    actual_compensate_ustime / 1000;

                actual_compensate_ustime -=
                    actual_compensate_mstime * 1000;
            }

            vTaskStepTick(
                pdMS_TO_TICKS(actual_compensate_mstime));
        }

        cpu_global_irq_enable();
    }
}

/* -------------------------------------------------------------------------- */
/* Main                                                                        */
/* -------------------------------------------------------------------------- */

int main(void)
{
    bflb_platform_init(0);

    gpio_set_mode(LED_B_PIN, GPIO_OUTPUT_PP_MODE);
    gpio_set_mode(LED_G_PIN, GPIO_OUTPUT_PP_MODE);
    gpio_set_mode(LED_R_PIN, GPIO_OUTPUT_PP_MODE);

    /* active-low LEDs: all off */
    gpio_write(LED_B_PIN, 1);
    gpio_write(LED_G_PIN, 1);
    gpio_write(LED_R_PIN, 1);

    /*
     * Same crystal setup as the known-working M0Sense BLE example.
     */
    HBN_Set_XCLK_CLK_Sel(HBN_XCLK_CLK_XTAL);

    uint32_t tmp = BL_RD_REG(AON_BASE, AON_XTAL_CFG);
    tmp = BL_SET_REG_BITS_VAL(
        tmp, AON_XTAL_CAPCODE_IN_AON, 33);
    tmp = BL_SET_REG_BITS_VAL(
        tmp, AON_XTAL_CAPCODE_OUT_AON, 33);
    BL_WR_REG(AON_BASE, AON_XTAL_CFG, tmp);

    Sec_Eng_Trng_Enable();

    vPortDefineHeapRegions(xHeapRegions);

    /*
     * Mode 2 uses the real QMI8658 continuously.
     * Keep BLE alive even if IMU init fails so debugging stays easy.
     */
    const bool imu_ok = nai_imu_bridge_init();
    if (!imu_ok) {
        gpio_write(LED_R_PIN, 0);
    }

    /*
     * bflb_platform_init() has already initialized the BL702 flash HAL.
     * This merely enables our single-slot wrapper.
     */
    const bool storage_ok = nai_storage_init();
    if (!storage_ok) {
        gpio_write(LED_R_PIN, 0);
    }

    /*
     * Only two user-created tasks:
     *   app      -> one boring continuous-mode + Noodle inference loop
     *   ble_init -> starts BLE then deletes itself
     */
    xTaskCreate(
        app_task,
        "app",
        1536,
        NULL,
        3,
        NULL
    );

    xTaskCreate(
        ble_init_task,
        "ble_init",
        1024,
        NULL,
        15,
        NULL
    );

    vTaskStartScheduler();

    for (;;) {
        __WFI();
    }
}
