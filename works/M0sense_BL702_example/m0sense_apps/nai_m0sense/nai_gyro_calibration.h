#ifndef NAI_GYRO_CALIBRATION_H
#define NAI_GYRO_CALIBRATION_H

#include <math.h>
#include <stdint.h>

// Header-only, allocation-free stationarity check for the 3-second startup
// zero-rate calibration.  Input order: ax, ay, az [g], gx, gy, gz [deg/s].
// Zero rate is only an assumption when the device is known to be stationary;
// no statistic alone can distinguish stillness from every constant rotation.
struct NaiGyroCalAccumulator {
    double sum[6] = {};
    double sumsq[6] = {};
    double gyro_first[3] = {};
    double gyro_second[3] = {};
    double accel_norm_sum = 0.0;
    double accel_norm_sumsq = 0.0;
    uint16_t n = 0;

    void add(const float s[6]) {
        if (!s || n >= 300) return;
        for (uint8_t k = 0; k < 6; ++k) {
            const double v = (double)s[k];
            sum[k] += v;
            sumsq[k] += v * v;
        }
        for (uint8_t k = 0; k < 3; ++k) {
            if (n < 150) gyro_first[k] += s[k + 3];
            else gyro_second[k] += s[k + 3];
        }
        const double norm = sqrt((double)s[0] * s[0] +
                                 (double)s[1] * s[1] +
                                 (double)s[2] * s[2]);
        accel_norm_sum += norm;
        accel_norm_sumsq += norm * norm;
        ++n;
    }

    bool stationary_bias(float bias[3]) const {
        if (!bias || n != 300) return false;
        const double inv_n = 1.0 / 300.0;
        const double norm_mean = accel_norm_sum * inv_n;
        const double norm_var = accel_norm_sumsq * inv_n - norm_mean * norm_mean;
        if (!isfinite(norm_mean) || fabs(norm_mean - 1.0) > 0.20 ||
            !isfinite(norm_var) || norm_var > 0.035 * 0.035) return false;

        for (uint8_t k = 0; k < 6; ++k) {
            const double mean = sum[k] * inv_n;
            const double var = sumsq[k] * inv_n - mean * mean;
            const double std_limit = (k < 3) ? 0.035 : 1.50; // g or deg/s
            if (!isfinite(mean) || !isfinite(var) ||
                var > std_limit * std_limit) return false;
            if (k >= 3) {
                const uint8_t axis = k - 3;
                const double first = gyro_first[axis] / 150.0;
                const double second = gyro_second[axis] / 150.0;
                if (fabs(second - first) > 0.80) return false;
                bias[axis] = (float)mean;
            }
        }
        return true;
    }
};

#endif
