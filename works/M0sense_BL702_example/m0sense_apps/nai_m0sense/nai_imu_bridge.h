#ifndef NAI_IMU_BRIDGE_H
#define NAI_IMU_BRIDGE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool nai_imu_bridge_init(void);
bool nai_imu_bridge_read(float s[6]);
bool nai_imu_bridge_gyro_calibrated(void);
void nai_imu_bridge_gyro_bias(float out3[3]);

#ifdef __cplusplus
}
#endif

#endif
