#include <stdint.h>
#include <string.h>
#include <errno.h>

extern "C" {
#include "FreeRTOS.h"
#include "queue.h"

#include "bluetooth.h"
#include "conn.h"
#include "gatt.h"
#include "gap.h"
#include "hci_core.h"

#include "ble_lib_api.h"
#include "hci_driver.h"

#include "bl702_glb.h"
#include "bl702_sec_eng.h"
}

#include "nai_protocol.h"
#include "nai_ble.h"

#include "hal_gpio.h"

#include "io_def.h"

/* bl_mcu_sdk v1.4.x ARRAY_SIZE uses C-only __builtin_types_compatible_p. */
#ifdef ARRAY_SIZE
#undef ARRAY_SIZE
#endif
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))


#define BT_UUID_SVC_NAI \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE( \
        0x7f8b0001,0x5f5b,0x4f4a,0xa5d5,0x2e889aa10001))
#define BT_UUID_CHAR_NAI_IMU \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE( \
        0x7f8b0002,0x5f5b,0x4f4a,0xa5d5,0x2e889aa10001))
#define BT_UUID_CHAR_NAI_CONTROL \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE( \
        0x7f8b0003,0x5f5b,0x4f4a,0xa5d5,0x2e889aa10001))
#define BT_UUID_CHAR_NAI_MODEL \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE( \
        0x7f8b0004,0x5f5b,0x4f4a,0xa5d5,0x2e889aa10001))
#define BT_UUID_CHAR_NAI_STATUS \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE( \
        0x7f8b0005,0x5f5b,0x4f4a,0xa5d5,0x2e889aa10001))

constexpr uint8_t IMU_VALUE_INDEX=2;
constexpr uint8_t STATUS_VALUE_INDEX=9;
constexpr uint16_t REQUIRED_IMU_MTU=sizeof(NaiImuPacket)+3;

static struct bt_conn *s_conn=nullptr;
static struct bt_gatt_exchange_params s_mtu_exchange;
static QueueHandle_t s_queue=nullptr;
static volatile bool s_imu_notify=false;
static volatile bool s_status_notify=false;
static volatile uint16_t s_mtu=20;
static bool s_registered=false;
static char s_status[32]="BOOT";

static int start_adv();

static bool enqueue(uint8_t type,const void *data,uint16_t len)
{
    if (!s_queue || len>NAI_EVENT_PAYLOAD_MAX) return false;
    NaiAppEvent e{};
    e.type=type; e.len=len;
    if (len && data) memcpy(e.data,data,len);
    return xQueueSend(s_queue,&e,0)==pdTRUE;
}

static void mtu_cb(struct bt_conn *conn,u8_t err,
                   struct bt_gatt_exchange_params *params)
{
    (void)params;
    if (!err && conn) s_mtu=bt_gatt_get_mtu(conn);
}

static void connected(struct bt_conn *conn,u8_t err)
{
    if (err) return;
    s_conn=conn;
    s_mtu=bt_gatt_get_mtu(conn);
    bt_le_set_data_len(conn,0x00fb,0x0848);
    s_mtu_exchange.func=mtu_cb;
    bt_gatt_exchange_mtu(conn,&s_mtu_exchange);
    gpio_write(LED_G_PIN,0);
    enqueue(NAI_EVT_CONNECTED,nullptr,0);
}

static void disconnected(struct bt_conn *conn,u8_t reason)
{
    (void)conn;(void)reason;
    s_conn=nullptr;
    s_imu_notify=false;
    s_status_notify=false;
    s_mtu=20;
    enqueue(NAI_EVT_DISCONNECTED,nullptr,0);
    start_adv();
}

static struct bt_conn_cb conn_cb={
    .connected=connected,
    .disconnected=disconnected,
};

static int status_read(struct bt_conn *conn,
                       const struct bt_gatt_attr *attr,
                       void *buf,u16_t len,u16_t offset)
{
    (void)conn;(void)attr;
    const uint16_t total=(uint16_t)strlen(s_status);
    if (offset>=total) return 0;
    uint16_t n=total-offset;
    if (n>len) n=len;
    memcpy(buf,s_status+offset,n);
    return n;
}

static void imu_ccc(const struct bt_gatt_attr *attr,u16_t value)
{
    (void)attr;
    s_imu_notify=(value==BT_GATT_CCC_NOTIFY);
}

static void status_ccc(const struct bt_gatt_attr *attr,u16_t value)
{
    (void)attr;
    s_status_notify=(value==BT_GATT_CCC_NOTIFY);
}

static int control_write(struct bt_conn *conn,
                         const struct bt_gatt_attr *attr,
                         const void *buf,u16_t len,u16_t offset,u8_t flags)
{
    (void)conn;(void)attr;(void)offset;(void)flags;
    if (!len || len>NAI_EVENT_PAYLOAD_MAX) return -EINVAL;
    return enqueue(NAI_EVT_CONTROL,buf,len) ? len : -ENOMEM;
}

static int model_write(struct bt_conn *conn,
                       const struct bt_gatt_attr *attr,
                       const void *buf,u16_t len,u16_t offset,u8_t flags)
{
    (void)conn;(void)attr;(void)offset;(void)flags;
    if (!len || len>NAI_EVENT_PAYLOAD_MAX) return -EINVAL;
    return enqueue(NAI_EVT_MODEL_DATA,buf,len) ? len : -ENOMEM;
}

static struct bt_gatt_attr attrs[]={
    BT_GATT_PRIMARY_SERVICE(BT_UUID_SVC_NAI),

    BT_GATT_CHARACTERISTIC(BT_UUID_CHAR_NAI_IMU,
        BT_GATT_CHRC_NOTIFY,0,NULL,NULL,NULL),
    BT_GATT_CCC(imu_ccc,BT_GATT_PERM_READ|BT_GATT_PERM_WRITE),

    BT_GATT_CHARACTERISTIC(BT_UUID_CHAR_NAI_CONTROL,
        BT_GATT_CHRC_WRITE|BT_GATT_CHRC_WRITE_WITHOUT_RESP,
        BT_GATT_PERM_WRITE,NULL,control_write,NULL),

    BT_GATT_CHARACTERISTIC(BT_UUID_CHAR_NAI_MODEL,
        BT_GATT_CHRC_WRITE|BT_GATT_CHRC_WRITE_WITHOUT_RESP,
        BT_GATT_PERM_WRITE,NULL,model_write,NULL),

    BT_GATT_CHARACTERISTIC(BT_UUID_CHAR_NAI_STATUS,
        BT_GATT_CHRC_READ|BT_GATT_CHRC_NOTIFY,
        BT_GATT_PERM_READ,status_read,NULL,NULL),
    BT_GATT_CCC(status_ccc,BT_GATT_PERM_READ|BT_GATT_PERM_WRITE),
};

static struct bt_gatt_service service=BT_GATT_SERVICE(attrs);

static int start_adv()
{
    static const char name[]="NoodleAI";

    struct bt_le_adv_param p={
        .options=BT_LE_ADV_OPT_CONNECTABLE,
        .interval_min=BT_GAP_ADV_FAST_INT_MIN_3,
        .interval_max=BT_GAP_ADV_FAST_INT_MAX_3,
    };

    struct bt_data ad[]={
        BT_DATA_BYTES(
            BT_DATA_FLAGS,
            BT_LE_AD_NO_BREDR | BT_LE_AD_GENERAL
        ),
        BT_DATA(
            BT_DATA_NAME_COMPLETE,
            name,
            sizeof(name)-1
        ),
    };

    return bt_le_adv_start(
        &p,
        ad,
        sizeof(ad)/sizeof(ad[0]),
        NULL,
        0
    );
}

static void enabled(int err)
{
    if (err) {
        gpio_write(LED_G_PIN,1);
        gpio_write(LED_R_PIN,0);
        return;
    }

    if (!s_registered) {
        s_registered=true;
        bt_conn_cb_register(&conn_cb);
        bt_gatt_service_register(&service);
    }

    if (start_adv()==0) {
        gpio_write(LED_R_PIN,1);
        gpio_write(LED_G_PIN,0);
    } else {
        gpio_write(LED_G_PIN,1);
        gpio_write(LED_R_PIN,0);
    }
}

void nai_ble_bind_event_queue(QueueHandle_t q) { s_queue=q; }

void nai_ble_start_stack()
{
    GLB_Set_EM_Sel(GLB_EM_8KB);
    ble_controller_init(configMAX_PRIORITIES-1);
    hci_driver_init();
    bt_enable(enabled);
}

bool nai_ble_connected() { return s_conn!=nullptr; }
bool nai_ble_imu_notify_enabled() { return s_imu_notify; }
uint16_t nai_ble_mtu() { return s_mtu; }

int nai_ble_notify_imu(const NaiImuPacket *pkt)
{
    if (!s_conn || !s_imu_notify || !pkt) return -ENOTCONN;
    if (s_mtu<REQUIRED_IMU_MTU) return -EMSGSIZE;
    return bt_gatt_notify(s_conn,&attrs[IMU_VALUE_INDEX],pkt,sizeof(*pkt));
}

void nai_ble_status(const char *ascii)
{
    if (!ascii) ascii="";
    strncpy(s_status,ascii,sizeof(s_status)-1);
    s_status[sizeof(s_status)-1]='\0';

    if (s_conn && s_status_notify) {
        bt_gatt_notify(s_conn,&attrs[STATUS_VALUE_INDEX],
                       s_status,strlen(s_status));
    }
}
