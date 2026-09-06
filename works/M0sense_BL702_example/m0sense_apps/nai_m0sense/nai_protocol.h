#pragma once
#include <stdint.h>

static constexpr uint16_t NAI_SAMPLE_RATE_HZ = 50;
static constexpr uint8_t NAI_BLE_BATCH_SIZE = 5;
static constexpr uint16_t NAI_MIN_GESTURE_SAMPLES = 10;
static constexpr uint16_t NAI_MAX_GESTURE_SAMPLES = 250;
static constexpr uint32_t NAI_SAMPLE_PERIOD_MS = 20;
static constexpr uint32_t NAI_BOOT_TRAIN_WINDOW_MS = 1200;
static constexpr uint32_t NAI_RUNTIME_BOOT_DEBOUNCE_MS = 30;
static constexpr float NAI_MADGWICK_BETA = 0.10f;
static constexpr float NAI_GRAVITY_MPS2 = 9.80665f;

enum NaiControlOpcode : uint8_t {
    OP_SET_TRAINING  = 0x01,
    OP_SET_INFERENCE = 0x02,
    OP_ERASE_MODEL   = 0x12,
    OP_DEPLOY_BEGIN  = 0x20,
    OP_FILE_BEGIN    = 0x21, // [op][nameLen:u8][size:u32][crc:u32][name]
    OP_FILE_END      = 0x22,
    OP_DEPLOY_COMMIT = 0x23,
    OP_DEPLOY_ABORT  = 0x24,
};

enum class NaiDeviceMode : uint8_t {
    TRAINING = 0,
    INFERENCE = 1,
};

#pragma pack(push, 1)
struct NaiImuSample {
    float ax, ay, az;
    float gx, gy, gz;
};

struct NaiImuPacket {
    uint32_t t0_ms;
    uint8_t count;
    uint8_t reserved[3];
    NaiImuSample samples[NAI_BLE_BATCH_SIZE];
};
#pragma pack(pop)

static_assert(sizeof(NaiImuPacket) == 128, "NoodleAI IMU packet must be 128 bytes");

enum NaiEventType : uint8_t {
    NAI_EVT_CONNECTED = 1,
    NAI_EVT_DISCONNECTED = 2,
    NAI_EVT_CONTROL = 3,
    NAI_EVT_MODEL_DATA = 4,
};

static constexpr uint16_t NAI_EVENT_PAYLOAD_MAX = 244;

struct NaiAppEvent {
    uint8_t type;
    uint16_t len;
    uint8_t data[NAI_EVENT_PAYLOAD_MAX];
};
