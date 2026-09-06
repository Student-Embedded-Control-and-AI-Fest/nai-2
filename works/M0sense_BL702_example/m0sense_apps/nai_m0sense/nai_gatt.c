#include "nai_gatt.h"

#include <errno.h>
#include <string.h>

#include <FreeRTOS.h>
#include "task.h"

#include "bluetooth.h"
#include "conn.h"
#include "gatt.h"
#include "gap.h"
#include "hci_core.h"
#include "uuid.h"

#include "ble_lib_api.h"
#include "hci_driver.h"
#include "bl702_glb.h"

#include "hal_gpio.h"
#include "io_def.h"

/*
 * Keep all Bouffalo GATT macros in C.
 *
 * This avoids the C++ compound-literal lifetime problem we hit previously.
 */

/* -------------------------------------------------------------------------- */
/* UUIDs                                                                       */
/* -------------------------------------------------------------------------- */

#define BT_UUID_NAI_SERVICE \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0001, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

#define BT_UUID_NAI_IMU \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0002, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

#define BT_UUID_NAI_CONTROL \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0003, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

#define BT_UUID_NAI_MODEL \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0004, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

#define BT_UUID_NAI_STATUS \
    BT_UUID_DECLARE_128( \
        BT_UUID_128_ENCODE(0x7f8b0005, 0x5f5b, 0x4f4a, 0xa5d5, 0x2e889aa10001))

/* -------------------------------------------------------------------------- */
/* Attribute indexes                                                          */
/* -------------------------------------------------------------------------- */

/*
 * Macro expansion gives:
 *
 *  0 service
 *  1 IMU declaration
 *  2 IMU value
 *  3 IMU CCC
 *  4 CONTROL declaration
 *  5 CONTROL value
 *  6 MODEL declaration
 *  7 MODEL value
 *  8 STATUS declaration
 *  9 STATUS value
 * 10 STATUS CCC
 */
enum {
    NAI_ATTR_SERVICE = 0,
    NAI_ATTR_IMU_DECL,
    NAI_ATTR_IMU_VALUE,
    NAI_ATTR_IMU_CCC,
    NAI_ATTR_CONTROL_DECL,
    NAI_ATTR_CONTROL_VALUE,
    NAI_ATTR_MODEL_DECL,
    NAI_ATTR_MODEL_VALUE,
    NAI_ATTR_STATUS_DECL,
    NAI_ATTR_STATUS_VALUE,
    NAI_ATTR_STATUS_CCC
};

/* -------------------------------------------------------------------------- */
/* Tiny shared state                                                          */
/* -------------------------------------------------------------------------- */

static struct bt_conn *s_conn = NULL;

static volatile bool s_imu_notify = false;
static volatile bool s_status_notify = false;

#define CONTROL_BUF_BYTES 64
#define MODEL_BUF_BYTES   244

static uint8_t s_control_buf[CONTROL_BUF_BYTES];
static volatile uint16_t s_control_len = 0;
static volatile bool s_control_pending = false;

static uint8_t s_model_buf[MODEL_BUF_BYTES];
static volatile uint16_t s_model_len = 0;
static volatile bool s_model_pending = false;

static char s_status[32] = "READY";

static bool s_registered = false;

/* Forward declarations. */
static int start_advertising(void);
static struct bt_gatt_attr nai_attrs[];

/* -------------------------------------------------------------------------- */
/* CCC callbacks                                                               */
/* -------------------------------------------------------------------------- */

static void imu_ccc_changed(
    const struct bt_gatt_attr *attr,
    u16_t value)
{
    (void)attr;

    s_imu_notify = (value == BT_GATT_CCC_NOTIFY);
}

static void status_ccc_changed(
    const struct bt_gatt_attr *attr,
    u16_t value)
{
    (void)attr;

    s_status_notify = (value == BT_GATT_CCC_NOTIFY);

    /*
     * As soon as the browser subscribes, give it one simple known status.
     */
    if (s_status_notify) {
        nai_gatt_status("READY");
    }
}

/* -------------------------------------------------------------------------- */
/* Characteristic callbacks                                                   */
/* -------------------------------------------------------------------------- */

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

    if (offset != 0) {
        return 0;
    }

    if (len > CONTROL_BUF_BYTES) {
        len = CONTROL_BUF_BYTES;
    }

    memcpy(s_control_buf, buf, len);
    s_control_len = len;
    s_control_pending = true;

    return len;
}

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

    /*
     * Stage 2 intentionally does not implement long/prepare writes.
     * Normal Web-Bluetooth write chunks have offset 0.
     */
    if (offset != 0) {
        return 0;
    }

    if (len > MODEL_BUF_BYTES) {
        len = MODEL_BUF_BYTES;
    }

    memcpy(s_model_buf, buf, len);
    s_model_len = len;
    s_model_pending = true;

    return len;
}

static int status_read(
    struct bt_conn *conn,
    const struct bt_gatt_attr *attr,
    void *buf,
    u16_t len,
    u16_t offset)
{
    (void)conn;
    (void)attr;

    const uint16_t n = (uint16_t)strlen(s_status);

    if (offset >= n) {
        return 0;
    }

    uint16_t remain = n - offset;

    if (remain > len) {
        remain = len;
    }

    memcpy(buf, s_status + offset, remain);

    return remain;
}

/* -------------------------------------------------------------------------- */
/* GATT table                                                                  */
/* -------------------------------------------------------------------------- */

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

static struct bt_gatt_service nai_service =
    BT_GATT_SERVICE(nai_attrs);

/* -------------------------------------------------------------------------- */
/* Connection callbacks                                                       */
/* -------------------------------------------------------------------------- */

static void connected(struct bt_conn *conn, u8_t err)
{
    if (err) {
        return;
    }

    s_conn = conn;
}

static void disconnected(
    struct bt_conn *conn,
    u8_t reason)
{
    (void)conn;
    (void)reason;

    s_conn = NULL;
    s_imu_notify = false;
    s_status_notify = false;

    /*
     * A connectable advertiser normally stops after connection.
     * Start it again so reconnecting is boring and predictable.
     */
    (void)start_advertising();
}

static struct bt_conn_cb conn_callbacks = {
    .connected = connected,
    .disconnected = disconnected,
};

/* -------------------------------------------------------------------------- */
/* Advertising                                                                */
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
     * Legacy advertising payload is exactly 31 bytes:
     *
     *   flags        3
     *   service UUID 18
     *   NoodleAI     10
     *               --
     *                31
     */
    struct bt_data ad[] = {
        BT_DATA_BYTES(
            BT_DATA_FLAGS,
            BT_LE_AD_NO_BREDR | BT_LE_AD_GENERAL),

        BT_DATA_BYTES(
            BT_DATA_UUID128_ALL,
            BT_UUID_128_ENCODE(
                0x7f8b0001,
                0x5f5b,
                0x4f4a,
                0xa5d5,
                0x2e889aa10001)),

        BT_DATA(
            BT_DATA_NAME_COMPLETE,
            name,
            sizeof(name) - 1),
    };

    return bt_le_adv_start(
        &p,
        ad,
        sizeof(ad) / sizeof(ad[0]),
        NULL,
        0);
}

/* -------------------------------------------------------------------------- */
/* BLE startup                                                                */
/* -------------------------------------------------------------------------- */

static void bluetooth_ready(int err)
{
    if (err) {
        gpio_write(LED_G_PIN, 1);
        gpio_write(LED_R_PIN, 0);
        return;
    }

    if (!s_registered) {
        s_registered = true;

        bt_conn_cb_register(&conn_callbacks);
        bt_gatt_service_register(&nai_service);
    }

    err = start_advertising();

    if (err == 0) {
        /* GREEN = advertising successfully started */
        gpio_write(LED_R_PIN, 1);
        gpio_write(LED_G_PIN, 0);
    } else {
        gpio_write(LED_G_PIN, 1);
        gpio_write(LED_R_PIN, 0);
    }
}

void nai_ble_start_stack(void)
{
    GLB_Set_EM_Sel(GLB_EM_8KB);

    ble_controller_init(configMAX_PRIORITIES - 1);
    hci_driver_init();
    bt_enable(bluetooth_ready);
}

/* -------------------------------------------------------------------------- */
/* State-machine API                                                          */
/* -------------------------------------------------------------------------- */

bool nai_ble_connected(void)
{
    return s_conn != NULL;
}

bool nai_ble_imu_notify_enabled(void)
{
    return s_imu_notify;
}

bool nai_ble_status_notify_enabled(void)
{
    return s_status_notify;
}

bool nai_gatt_take_control(
    uint8_t *dst,
    uint16_t dst_capacity,
    uint16_t *len_out)
{
    if (!s_control_pending) {
        return false;
    }

    uint16_t n = s_control_len;

    if (n > dst_capacity) {
        n = dst_capacity;
    }

    if (dst && n) {
        memcpy(dst, s_control_buf, n);
    }

    if (len_out) {
        *len_out = n;
    }

    s_control_pending = false;
    s_control_len = 0;

    return true;
}

bool nai_gatt_take_model(
    uint8_t *dst,
    uint16_t dst_capacity,
    uint16_t *len_out)
{
    if (!s_model_pending) {
        return false;
    }

    uint16_t n = s_model_len;

    if (n > dst_capacity) {
        n = dst_capacity;
    }

    if (dst && n) {
        memcpy(dst, s_model_buf, n);
    }

    if (len_out) {
        *len_out = n;
    }

    s_model_pending = false;
    s_model_len = 0;

    return true;
}

int nai_gatt_notify_imu(
    const void *data,
    uint16_t len)
{
    if (!s_conn || !s_imu_notify || !data || !len) {
        return -ENOTCONN;
    }

    return bt_gatt_notify(
        s_conn,
        &nai_attrs[NAI_ATTR_IMU_VALUE],
        data,
        len);
}

int nai_gatt_status(const char *text)
{
    if (!text) {
        return -EINVAL;
    }

    strncpy(s_status, text, sizeof(s_status) - 1);
    s_status[sizeof(s_status) - 1] = '\0';

    if (!s_conn || !s_status_notify) {
        return 0;
    }

    return bt_gatt_notify(
        s_conn,
        &nai_attrs[NAI_ATTR_STATUS_VALUE],
        s_status,
        strlen(s_status));
}
