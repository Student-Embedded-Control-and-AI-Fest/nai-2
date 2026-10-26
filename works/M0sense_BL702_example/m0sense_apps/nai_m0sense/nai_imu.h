#ifndef NAI_IMU_H
#define NAI_IMU_H

bool nai_imu_init();
bool nai_imu_read(float out6[6]);

// Startup zero-rate calibration diagnostics (raw gyro bias in deg/s).
bool nai_imu_gyro_calibrated();
void nai_imu_gyro_bias(float out3[3]);

#endif
