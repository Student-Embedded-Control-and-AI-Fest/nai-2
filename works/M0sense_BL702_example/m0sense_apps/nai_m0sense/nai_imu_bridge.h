#ifndef NAI_IMU_BRIDGE_H
#define NAI_IMU_BRIDGE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool nai_imu_bridge_init(void);
bool nai_imu_bridge_read(float s[6]);

#ifdef __cplusplus
}
#endif

#endif
