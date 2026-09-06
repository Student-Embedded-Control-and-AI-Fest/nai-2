#pragma once
#include <stdbool.h>
#include <stdint.h>

#include <FreeRTOS.h>
#include <queue.h>

#include "nai_protocol.h"

void nai_ble_bind_event_queue(QueueHandle_t q);
void nai_ble_start_stack();

bool nai_ble_connected();
bool nai_ble_imu_notify_enabled();
uint16_t nai_ble_mtu();

int nai_ble_notify_imu(const NaiImuPacket *pkt);
void nai_ble_status(const char *ascii);
