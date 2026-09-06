#pragma once

#include <stddef.h>
#include <stdint.h>

#include "noodle/noodle.h"

class NoodleAIModelBL702 {
public:
    static constexpr uint16_t MAX_LAYERS = 8;
    static constexpr uint16_t MAX_CLASSES = 10;
    static constexpr uint16_t MAX_DIM = 768;
    static constexpr uint8_t MAX_LABEL_BYTES = 31;

    enum InputRepresentation : uint16_t {
        REP_ACCEL = 1,
        REP_GYRO = 2,
        REP_ACCEL_GYRO = 3,
        REP_QUATERNION = 4,
        REP_VELOCITY = 5,
        REP_VELOCITY_QUATERNION = 6,
    };

    /*
     * Detailed loader result.  Kept negative so 0 remains success across the
     * small C bridge.
     */
    enum FilterFlags : uint16_t {
        FILTER_NONE = 0x0000,
        FILTER_HP   = 0x0001,
        FILTER_LP   = 0x0002,
    };

    enum LoadError : int {
        LOAD_OK               = 0,
        LOAD_ERR_NO_SLOT      = -10,
        LOAD_ERR_CONFIG       = -11,
        LOAD_ERR_FILES        = -12,
        LOAD_ERR_SCALER_ALLOC = -13,
        LOAD_ERR_SCALER_READ  = -14,
        LOAD_ERR_SCALER_VALUE = -15,
        LOAD_ERR_TENSOR_A     = -16,
        LOAD_ERR_TENSOR_B     = -17,
        LOAD_ERR_POSTCHECK    = -18,
    };

    NoodleAIModelBL702();
    ~NoodleAIModelBL702();

    bool loadActive();
    bool loadSlot(char slot, bool allowStaged = false);
    void clearRuntime();

    bool loaded() const { return _loaded; }
    int lastLoadError() const { return _lastLoadError; }
    char activeSlot() const { return _slot; }
    uint16_t inputDim() const { return _inputDim; }
    uint16_t windowLength() const { return _windowLength; }
    uint16_t sampleRateHz() const { return _sampleRateHz; }
    uint16_t strideSamples() const { return _strideSamples; }
    float confidenceThreshold() const { return _confidenceThreshold; }
    uint16_t classCount() const { return _nClasses; }
    uint16_t layerCount() const { return _nLayers; }
    InputRepresentation representation() const { return _representation; }
    uint16_t filterFlags() const { return _filterFlags; }
    float highpassCutoffHz() const { return _highpassCutoffHz; }
    float lowpassCutoffHz() const { return _lowpassCutoffHz; }

    uint16_t channelCount() const;
    const char *representationName() const;
    const char *label(uint16_t index) const;

    bool predict(const float *x, uint16_t &classIndex, float &confidence);
    size_t tensorArenaBytes() const;

    static bool canonicalFilenameAllowed(const char *name);
    static bool slotFilename(char slot, const char *canonical,
                             char *out, size_t outCap);

private:
#pragma pack(push, 1)
    struct ConfigHeader {
        char magic[4];
        uint16_t version;
        uint16_t flags;
        uint16_t windowLength;
        uint16_t sampleRateHz;
        uint16_t nLayers;
        uint16_t nClasses;
        uint16_t inputDim;
        uint16_t reserved;   /* input representation code */
    };

    struct QuantConfigHeader {
        char magic[4];       /* "Q8M1" */
        uint16_t strideSamples;
        uint16_t reserved;
        float thresholdMin;
        float thresholdMax;
        float confidenceThreshold;
        float inputScale;
        int32_t inputZeroPoint;
    };

    struct QuantLayerConfig {
        float outputScale;
        int32_t outputZeroPoint;
    };

    struct FilterConfigHeader {
        char magic[4];       /* "FLT1" */
        uint16_t flags;      /* FILTER_HP | FILTER_LP */
        uint16_t reserved;
        float highpassHz;
        float lowpassHz;
    };
#pragma pack(pop)

    bool readExact(NDL_File &f, void *dst, size_t n) const;
    bool readConfig(char slot);
    bool validateParameterFiles(char slot);
    bool loadScaler(char slot);
    void configureLayers(char slot);
    void initTensorsIfNeeded();
    static bool fileHasSize(const char *name, size_t expected);
    static uint16_t representationChannelCount(InputRepresentation rep);

    bool _loaded = false;
    int _lastLoadError = LOAD_OK;
    bool _binaryClassifier = false;
    bool _tensorsInitialized = false;
    char _slot = 0;

    InputRepresentation _representation = REP_ACCEL;
    uint16_t _windowLength = 0;
    uint16_t _sampleRateHz = 0;
    uint16_t _strideSamples = 5;
    uint16_t _nLayers = 0;
    uint16_t _nClasses = 0;
    uint16_t _inputDim = 0;

    uint16_t _filterFlags = FILTER_NONE;
    float _highpassCutoffHz = 0.0f;
    float _lowpassCutoffHz = 0.0f;

    float _thresholdMin = 0.30f;
    float _thresholdMax = 0.90f;
    float _confidenceThreshold = 0.50f;
    float _inputScale = 1.0f;
    int32_t _inputZeroPoint = 0;

    uint16_t _dims[MAX_LAYERS + 1]{};
    float _outputScales[MAX_LAYERS]{};
    int32_t _outputZeroPoints[MAX_LAYERS]{};

    float *_mean = nullptr;
    float *_scale = nullptr;
    FCNFile _layers[MAX_LAYERS]{};
    char _labels[MAX_CLASSES][MAX_LABEL_BYTES + 1]{};

    char _weightNames[MAX_LAYERS][NOODLE_MAX_FILENAME + 1]{};
    char _biasNames[MAX_LAYERS][NOODLE_MAX_FILENAME + 1]{};
    char _multiplierNames[MAX_LAYERS][NOODLE_MAX_FILENAME + 1]{};
    char _shiftNames[MAX_LAYERS][NOODLE_MAX_FILENAME + 1]{};

    NoodleTensor _actA{};
    NoodleTensor _actB{};
};
