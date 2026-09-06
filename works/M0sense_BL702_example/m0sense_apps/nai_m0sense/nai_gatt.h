#ifndef NAI_GATT_H
#define NAI_GATT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NoodleAI BLE UUID family
 *
 * service  7f8b0001-5f5b-4f4a-a5d5-2e889aa10001
 * IMU      7f8b0002-5f5b-4f4a-a5d5-2e889aa10001
 * CONTROL  7f8b0003-5f5b-4f4a-a5d5-2e889aa10001
 * MODEL    7f8b0004-5f5b-4f4a-a5d5-2e889aa10001
 * STATUS   7f8b0005-5f5b-4f4a-a5d5-2e889aa10001
 */

void nai_ble_start_stack(void);

bool nai_ble_connected(void);
bool nai_ble_imu_notify_enabled(void);
bool nai_ble_status_notify_enabled(void);

/* BLE callback -> state-machine handoff. */
bool nai_gatt_take_control(
    uint8_t *dst,
    uint16_t dst_capacity,
    uint16_t *len_out);

bool nai_gatt_take_model(
    uint8_t *dst,
    uint16_t dst_capacity,
    uint16_t *len_out);

/* Notifications used by later stages. */
int nai_gatt_notify_imu(const void *data, uint16_t len);
int nai_gatt_status(const char *text);

#ifdef __cplusplus
}
#endif

#endif
