#include "nai_imu.h"
#include "nai_gyro_calibration.h"

extern "C" {
#include "bflb_platform.h"
#include "bl702_glb.h"
#include "hal_i2c.h"
}

#include "io_def.h"

namespace {

constexpr uint8_t ADDR = 0x6A;
constexpr uint8_t WHOAMI = 0x00;
constexpr uint8_t CTRL1 = 0x02;
constexpr uint8_t CTRL2 = 0x03;
constexpr uint8_t CTRL3 = 0x04;
constexpr uint8_t CTRL5 = 0x06;
constexpr uint8_t CTRL7 = 0x08;
constexpr uint8_t REG_RESET = 0x60;
constexpr uint8_t AX_L = 0x35;

constexpr float ACC_SCALE_G = 4.0f / 32768.0f;
constexpr float GYR_SCALE_DPS = 1024.0f / 32768.0f;

/*
 * Startup zero-rate calibration requires known stillness.  The QMI8658 has
 * independent accel/gyro sample rates; these are approximate sampling periods.
 * Do not overwrite the bias when the stationarity checks fail.
 */
constexpr uint16_t GYRO_CAL_WARMUP_SAMPLES = 50;  // approx. 0.5 s
constexpr uint16_t GYRO_CAL_SAMPLES = 300;        // approx. 3.0 s
constexpr uint16_t GYRO_CAL_DELAY_MS = 10;

struct device *s_i2c = nullptr;

float s_gyro_bias_dps[3] = {0.0f, 0.0f, 0.0f};
bool s_gyro_calibrated = false;

bool write_reg(uint8_t reg, uint8_t val)
{
    i2c_msg_t m{};
    m.buf = &val;
    m.slaveaddr = ADDR;
    m.len = 1;
    m.flags = SUB_ADDR_1BYTE | I2C_WR;
    m.subaddr = reg;
    return i2c_transfer(s_i2c, &m, 1) == 0;
}

bool read_reg(uint8_t reg, uint8_t &val)
{
    i2c_msg_t m{};
    m.buf = &val;
    m.slaveaddr = ADDR;
    m.len = 1;
    m.flags = SUB_ADDR_1BYTE | I2C_RD;
    m.subaddr = reg;
    return i2c_transfer(s_i2c, &m, 1) == 0;
}

bool masked_is(uint8_t reg, uint8_t mask, uint8_t expected)
{
    uint8_t v = 0;
    return read_reg(reg, v) && (v & mask) == expected;
}

bool read_raw_scaled(float out6[6])
{
    if (!s_i2c || !out6) return false;

    uint8_t b[12];
    for (uint8_t i = 0; i < 12; ++i) {
        i2c_msg_t m{};
        m.buf = &b[i];
        m.slaveaddr = ADDR;
        m.len = 1;
        m.flags = SUB_ADDR_1BYTE | I2C_RD;
        m.subaddr = (uint8_t)(AX_L + i);
        if (i2c_transfer(s_i2c, &m, 1) != 0) return false;
    }

    const int16_t ax = (int16_t)(((uint16_t)b[1] << 8) | b[0]);
    const int16_t ay = (int16_t)(((uint16_t)b[3] << 8) | b[2]);
    const int16_t az = (int16_t)(((uint16_t)b[5] << 8) | b[4]);
    const int16_t gx = (int16_t)(((uint16_t)b[7] << 8) | b[6]);
    const int16_t gy = (int16_t)(((uint16_t)b[9] << 8) | b[8]);
    const int16_t gz = (int16_t)(((uint16_t)b[11] << 8) | b[10]);

    out6[0] = ax * ACC_SCALE_G;
    out6[1] = ay * ACC_SCALE_G;
    out6[2] = az * ACC_SCALE_G;
    out6[3] = gx * GYR_SCALE_DPS;
    out6[4] = gy * GYR_SCALE_DPS;
    out6[5] = gz * GYR_SCALE_DPS;

    return true;
}

// Return false only for an IMU read error; a moving board remains usable
// with correction DISABLED, and the browser is told the calibration failed.
bool calibrate_gyro_zero_rate()
{
    float s[6];
    s_gyro_calibrated = false;
    s_gyro_bias_dps[0] = 0.0f;
    s_gyro_bias_dps[1] = 0.0f;
    s_gyro_bias_dps[2] = 0.0f;

    for (uint16_t i = 0; i < GYRO_CAL_WARMUP_SAMPLES; ++i) {
        if (!read_raw_scaled(s)) return false;
        bflb_platform_delay_ms(GYRO_CAL_DELAY_MS);
    }

    NaiGyroCalAccumulator stats;
    for (uint16_t i = 0; i < GYRO_CAL_SAMPLES; ++i) {
        if (!read_raw_scaled(s)) return false;
        stats.add(s);
        bflb_platform_delay_ms(GYRO_CAL_DELAY_MS);
    }

    float candidate[3];
    if (!stats.stationary_bias(candidate)) {
        // Never force a stable but unverified motion interval to gyro zero.
        return true;
    }

    s_gyro_bias_dps[0] = candidate[0];
    s_gyro_bias_dps[1] = candidate[1];
    s_gyro_bias_dps[2] = candidate[2];
    s_gyro_calibrated = true;
    return true;
}

} // namespace

bool nai_imu_init()
{
    GLB_GPIO_Type gpios[] = {
        (GLB_GPIO_Type)IMU_SCK_PIN,
        (GLB_GPIO_Type)IMU_SDA_PIN
    };
    GLB_GPIO_Func_Init(GPIO_FUN_I2C, gpios, 2);

    i2c_register(I2C0_INDEX, "nai_i2c");
    s_i2c = device_find("nai_i2c");
    if (!s_i2c) return false;
    if (device_open(s_i2c, DEVICE_OFLAG_DEFAULT) != 0) return false;

    uint8_t who = 0;
    if (!read_reg(WHOAMI, who) || who != 0x05) return false;

    /*
     * accel ±4 g,       ODR 125 Hz
     * gyro  ±1024 dps,  ODR 112.1 Hz
     */
    if (!write_reg(REG_RESET, 0xB0)) return false;

    /* Restore the short delay used by the older working driver. */
    bflb_platform_delay_ms(20);

    if (!write_reg(CTRL1, 0x40)) return false;
    if (!write_reg(CTRL2, 0x16)) return false;
    if (!write_reg(CTRL3, 0x66)) return false;
    if (!write_reg(CTRL5, 0x00)) return false;
    if (!write_reg(CTRL7, 0x03)) return false;

    /* Let gyro output settle before estimating zero-rate bias. */
    bflb_platform_delay_ms(100);

    if (!masked_is(CTRL1, 0x40, 0x40)) return false;
    if (!masked_is(CTRL2, 0x7F, 0x16)) return false;
    if (!masked_is(CTRL3, 0x7F, 0x66)) return false;
    if (!masked_is(CTRL5, 0x11, 0x00)) return false;
    if (!masked_is(CTRL7, 0x03, 0x03)) return false;

    /*
     * Board must remain motionless for approx. 3.5 seconds after init.
     * Gyro bias is applied only if BOTH accelerometer and gyro were stable.
     */
    if (!calibrate_gyro_zero_rate()) return false;

    return true;
}

bool nai_imu_read(float out6[6])
{
    if (!out6) return false;

    float s[6];
    if (!read_raw_scaled(s)) return false;

    out6[0] = s[0];
    out6[1] = s[1];
    out6[2] = s[2];

    out6[3] = s[3] - (s_gyro_calibrated ? s_gyro_bias_dps[0] : 0.0f);
    out6[4] = s[4] - (s_gyro_calibrated ? s_gyro_bias_dps[1] : 0.0f);
    out6[5] = s[5] - (s_gyro_calibrated ? s_gyro_bias_dps[2] : 0.0f);

    return true;
}

// Used by the existing BLE status channel.  No new protocol/opcode is needed.
bool nai_imu_gyro_calibrated()
{
    return s_gyro_calibrated;
}

void nai_imu_gyro_bias(float out3[3])
{
    if (!out3) return;
    for (uint8_t k = 0; k < 3; ++k) {
        out3[k] = s_gyro_calibrated ? s_gyro_bias_dps[k] : 0.0f;
    }
}
