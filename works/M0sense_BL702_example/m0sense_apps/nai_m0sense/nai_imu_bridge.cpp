#include "nai_imu_bridge.h"
#include "nai_imu.h"

extern "C" bool nai_imu_bridge_init(void)
{
    return nai_imu_init();
}

extern "C" bool nai_imu_bridge_read(float s[6])
{
    return nai_imu_read(s);
}
