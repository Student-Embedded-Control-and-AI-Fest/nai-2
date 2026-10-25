#include "NoodleAIModelBL702.h"

#include <math.h>
#include <new>
#include <stdio.h>
#include <string.h>

#include "nai_slotfs.h"
#include "noodle/noodle_int8.h"

namespace {
constexpr uint16_t MODEL_VERSION = 4;
constexpr uint16_t FLAG_BINARY_CLASSIFIER = 0x0001;
constexpr uint16_t FLAG_INT8 = 0x0002;
constexpr const char *CFG_FILE = "cfg.bin";
constexpr const char *MEAN_FILE = "mean.bin";
constexpr const char *SCALE_FILE = "scale.bin";
}

NoodleAIModelBL702::NoodleAIModelBL702() = default;
NoodleAIModelBL702::~NoodleAIModelBL702() { clearRuntime(); }

bool NoodleAIModelBL702::readExact(NDL_File &f, void *dst, size_t n) const
{
    return f.readBytes((char *)dst, n) == n;
}

void NoodleAIModelBL702::initTensorsIfNeeded()
{
    if (_tensorsInitialized) return;
    noodle_tensor_init(&_actA);
    noodle_tensor_init(&_actB);
    _tensorsInitialized = true;
}

void NoodleAIModelBL702::clearRuntime()
{
    delete[] _mean;
    delete[] _scale;
    _mean = nullptr;
    _scale = nullptr;

    noodle_i8_temp_buffers_free();

    if (_tensorsInitialized) {
        noodle_tensor_free(&_actA);
        noodle_tensor_free(&_actB);
        _tensorsInitialized = false;
    }

    for (uint16_t i = 0; i < MAX_LAYERS; ++i) {
        _layers[i] = FCNFile{};
        _packedLayers[i] = false;
        _metadataNames[i][0] = '\0';
        _weightNames[i][0] = '\0';
        _biasNames[i][0] = '\0';
        _multiplierNames[i][0] = '\0';
        _shiftNames[i][0] = '\0';
    }
    memset(_labels, 0, sizeof(_labels));
    memset(_dims, 0, sizeof(_dims));
    memset(_outputScales, 0, sizeof(_outputScales));
    memset(_outputZeroPoints, 0, sizeof(_outputZeroPoints));

    _loaded = false;
    _binaryClassifier = false;
    _slot = 0;
    _representation = REP_ACCEL;
    _windowLength = 0;
    _sampleRateHz = 0;
    _strideSamples = 5;
    _nLayers = 0;
    _nClasses = 0;
    _inputDim = 0;
    _filterFlags = FILTER_NONE;
    _highpassCutoffHz = 0.0f;
    _lowpassCutoffHz = 0.0f;
    _thresholdMin = 0.30f;
    _thresholdMax = 0.90f;
    _confidenceThreshold = 0.50f;
    _inputScale = 1.0f;
    _inputZeroPoint = 0;
}

bool NoodleAIModelBL702::canonicalFilenameAllowed(const char *name)
{
    return nai_slotfs_canonical_name_allowed(name);
}

bool NoodleAIModelBL702::slotFilename(char slot, const char *canonical,
                                      char *out, size_t outCap)
{
    if (!out || !outCap || !canonical ||
        (slot != 'A' && slot != 'B') ||
        !canonicalFilenameAllowed(canonical)) return false;

    const int n = snprintf(out, outCap, "%c_%s",
                           slot == 'A' ? 'a' : 'b', canonical);
    return n > 0 && (size_t)n < outCap && (size_t)n <= NOODLE_MAX_FILENAME;
}

bool NoodleAIModelBL702::fileHasSize(const char *name, size_t expected)
{
    NDL_File f = noodle_fs_open_read(name);
    if (!f) return false;
    const size_t n = f.size();
    f.close();
    return n == expected;
}

bool NoodleAIModelBL702::readConfig(char slot)
{
    char name[NOODLE_MAX_FILENAME + 1];
    if (!slotFilename(slot, CFG_FILE, name, sizeof(name))) return false;

    NDL_File f = noodle_fs_open_read(name);
    if (!f) return false;

    ConfigHeader h{};
    if (!readExact(f, &h, sizeof(h))) { f.close(); return false; }

    if (memcmp(h.magic, "NAI4", 4) != 0 || h.version != MODEL_VERSION ||
        (h.flags & FLAG_INT8) == 0) {
        f.close();
        return false;
    }

    if (h.reserved < REP_ACCEL || h.reserved > REP_VELOCITY_QUATERNION) {
        f.close();
        return false;
    }

    const InputRepresentation rep = (InputRepresentation)h.reserved;
    const uint16_t channels = representationChannelCount(rep);

    if (h.nLayers == 0 || h.nLayers > MAX_LAYERS ||
        h.nClasses < 2 || h.nClasses > MAX_CLASSES ||
        h.windowLength == 0 || h.sampleRateHz == 0 ||
        h.inputDim == 0 || h.inputDim > MAX_DIM ||
        h.inputDim != (uint32_t)h.windowLength * channels) {
        f.close();
        return false;
    }

    _binaryClassifier = (h.flags & FLAG_BINARY_CLASSIFIER) != 0;
    _representation = rep;
    _windowLength = h.windowLength;
    _sampleRateHz = h.sampleRateHz;
    _nLayers = h.nLayers;
    _nClasses = h.nClasses;
    _inputDim = h.inputDim;

    if (!readExact(f, _dims, sizeof(uint16_t) * (_nLayers + 1)) ||
        _dims[0] != _inputDim) {
        f.close();
        return false;
    }

    for (uint16_t l = 0; l < _nLayers; ++l) {
        if (_dims[l] == 0 || _dims[l] > MAX_DIM ||
            _dims[l + 1] == 0 || _dims[l + 1] > MAX_DIM) {
            f.close();
            return false;
        }
    }

    const uint16_t finalDim = _dims[_nLayers];
    if (_binaryClassifier) {
        if (_nClasses != 2 || finalDim != 1) { f.close(); return false; }
    } else if (finalDim != _nClasses) {
        f.close();
        return false;
    }

    for (uint16_t c = 0; c < _nClasses; ++c) {
        uint8_t len = 0;
        if (!readExact(f, &len, 1) || len == 0 || len > MAX_LABEL_BYTES) {
            f.close(); return false;
        }
        if (!readExact(f, _labels[c], len)) { f.close(); return false; }
        _labels[c][len] = '\0';
    }

    QuantConfigHeader q{};
    if (!readExact(f, &q, sizeof(q)) || memcmp(q.magic, "Q8M1", 4) != 0) {
        f.close();
        return false;
    }

    if (q.strideSamples == 0 || q.strideSamples > _windowLength ||
        !isfinite(q.thresholdMin) || !isfinite(q.thresholdMax) ||
        !isfinite(q.confidenceThreshold) ||
        q.thresholdMin < 0.0f || q.thresholdMax > 1.0f ||
        q.thresholdMin > q.thresholdMax ||
        q.confidenceThreshold < q.thresholdMin ||
        q.confidenceThreshold > q.thresholdMax ||
        !isfinite(q.inputScale) || q.inputScale <= 0.0f ||
        q.inputZeroPoint < -128 || q.inputZeroPoint > 127) {
        f.close();
        return false;
    }

    _strideSamples = q.strideSamples;
    _thresholdMin = q.thresholdMin;
    _thresholdMax = q.thresholdMax;
    _confidenceThreshold = q.confidenceThreshold;
    _inputScale = q.inputScale;
    _inputZeroPoint = q.inputZeroPoint;

    for (uint16_t l = 0; l < _nLayers; ++l) {
        QuantLayerConfig ql{};
        if (!readExact(f, &ql, sizeof(ql)) ||
            !isfinite(ql.outputScale) || ql.outputScale <= 0.0f ||
            ql.outputZeroPoint < -128 || ql.outputZeroPoint > 127) {
            f.close();
            return false;
        }
        _outputScales[l] = ql.outputScale;
        _outputZeroPoints[l] = ql.outputZeroPoint;
    }

    /*
     * Optional NAI4 preprocessing extension.  Old Q8M1 packages ended here
     * and remain valid with FILTER_NONE so already-deployed models keep their
     * original preprocessing behavior.
     */
    _filterFlags = FILTER_NONE;
    _highpassCutoffHz = 0.0f;
    _lowpassCutoffHz = 0.0f;

    if (f.available()) {
        FilterConfigHeader flt{};
        if (!readExact(f, &flt, sizeof(flt)) ||
            memcmp(flt.magic, "FLT1", 4) != 0 ||
            (flt.flags & ~(FILTER_HP | FILTER_LP)) != 0 ||
            !isfinite(flt.highpassHz) || !isfinite(flt.lowpassHz)) {
            f.close();
            return false;
        }

        const float nyquist = 0.5f * (float)_sampleRateHz;
        if ((flt.flags & FILTER_HP) != 0) {
            if (flt.highpassHz <= 0.0f || flt.highpassHz >= nyquist) {
                f.close();
                return false;
            }
        } else if (fabsf(flt.highpassHz) > 1.0e-6f) {
            f.close();
            return false;
        }

        if ((flt.flags & FILTER_LP) != 0) {
            if (flt.lowpassHz <= 0.0f || flt.lowpassHz >= nyquist) {
                f.close();
                return false;
            }
        } else if (fabsf(flt.lowpassHz) > 1.0e-6f) {
            f.close();
            return false;
        }

        /* Quaternion/velocity paths must preserve DC gravity and slow turns. */
        if (_representation >= REP_QUATERNION &&
            (flt.flags & FILTER_HP) != 0) {
            f.close();
            return false;
        }

        _filterFlags = flt.flags;
        _highpassCutoffHz = flt.highpassHz;
        _lowpassCutoffHz = flt.lowpassHz;
    }

    const bool noTrailing = !f.available();
    f.close();
    return noTrailing;
}

bool NoodleAIModelBL702::validateParameterFiles(char slot)
{
    char name[NOODLE_MAX_FILENAME + 1];

    if (!slotFilename(slot, MEAN_FILE, name, sizeof(name)) ||
        !fileHasSize(name, sizeof(float) * _inputDim)) return false;
    if (!slotFilename(slot, SCALE_FILE, name, sizeof(name)) ||
        !fileHasSize(name, sizeof(float) * _inputDim)) return false;

    for (uint16_t l = 0; l < _nLayers; ++l) {
        char canonical[8];
        const size_t O = _dims[l + 1];

        snprintf(canonical, sizeof(canonical), "w%02u.bin", l);
        if (!slotFilename(slot, canonical, name, sizeof(name)) ||
            !fileHasSize(name, (size_t)_dims[l] * O)) return false;

        // New Noodle packed FCN files store 3 x int32 LE for each output.
        snprintf(canonical, sizeof(canonical), "p%02u.bin", l);
        if (!slotFilename(slot, canonical, name, sizeof(name))) return false;
        _packedLayers[l] = fileHasSize(name, 12u * O);
        if (_packedLayers[l]) continue;

        // Read-only legacy compatibility for packages already on devices.
        snprintf(canonical, sizeof(canonical), "b%02u.bin", l);
        if (!slotFilename(slot, canonical, name, sizeof(name)) ||
            !fileHasSize(name, sizeof(int32_t) * O)) return false;
        snprintf(canonical, sizeof(canonical), "m%02u.bin", l);
        if (!slotFilename(slot, canonical, name, sizeof(name)) ||
            !fileHasSize(name, sizeof(int32_t) * O)) return false;
        snprintf(canonical, sizeof(canonical), "s%02u.bin", l);
        if (!slotFilename(slot, canonical, name, sizeof(name)) ||
            !fileHasSize(name, sizeof(int32_t) * O)) return false;
    }
    return true;
}

bool NoodleAIModelBL702::loadScaler(char slot)
{
    _mean = new (std::nothrow) float[_inputDim];
    _scale = new (std::nothrow) float[_inputDim];
    if (!_mean || !_scale) {
        _lastLoadError = LOAD_ERR_SCALER_ALLOC;
        return false;
    }

    char name[NOODLE_MAX_FILENAME + 1];

    if (!slotFilename(slot, MEAN_FILE, name, sizeof(name))) {
        _lastLoadError = LOAD_ERR_SCALER_READ;
        return false;
    }
    NDL_File fm = noodle_fs_open_read(name);
    if (!fm || !readExact(fm, _mean, sizeof(float) * _inputDim)) {
        if (fm) fm.close();
        _lastLoadError = LOAD_ERR_SCALER_READ;
        return false;
    }
    fm.close();

    if (!slotFilename(slot, SCALE_FILE, name, sizeof(name))) {
        _lastLoadError = LOAD_ERR_SCALER_READ;
        return false;
    }
    NDL_File fs = noodle_fs_open_read(name);
    if (!fs || !readExact(fs, _scale, sizeof(float) * _inputDim)) {
        if (fs) fs.close();
        _lastLoadError = LOAD_ERR_SCALER_READ;
        return false;
    }
    fs.close();

    for (uint16_t i = 0; i < _inputDim; ++i) {
        if (!isfinite(_mean[i]) || !isfinite(_scale[i]) ||
            fabsf(_scale[i]) < 1e-12f) {
            _lastLoadError = LOAD_ERR_SCALER_VALUE;
            return false;
        }
    }
    return true;
}

void NoodleAIModelBL702::configureLayers(char slot)
{
    for (uint16_t l = 0; l < _nLayers; ++l) {
        char canonical[8];

        snprintf(canonical, sizeof(canonical), "w%02u.bin", l);
        slotFilename(slot, canonical, _weightNames[l], sizeof(_weightNames[l]));
        snprintf(canonical, sizeof(canonical), "p%02u.bin", l);
        slotFilename(slot, canonical, _metadataNames[l], sizeof(_metadataNames[l]));
        snprintf(canonical, sizeof(canonical), "b%02u.bin", l);
        slotFilename(slot, canonical, _biasNames[l], sizeof(_biasNames[l]));
        snprintf(canonical, sizeof(canonical), "m%02u.bin", l);
        slotFilename(slot, canonical, _multiplierNames[l], sizeof(_multiplierNames[l]));
        snprintf(canonical, sizeof(canonical), "s%02u.bin", l);
        slotFilename(slot, canonical, _shiftNames[l], sizeof(_shiftNames[l]));

        _layers[l] = FCNFile{};
        _layers[l].weight_fn = _weightNames[l];
        if (_packedLayers[l]) {
            _layers[l].metadata_fn = _metadataNames[l];
        } else {
            _layers[l].bias_fn = _biasNames[l];
            _layers[l].multiplier_fn = _multiplierNames[l];
            _layers[l].shift_fn = _shiftNames[l];
        }
        _layers[l].O = _dims[l + 1];
        _layers[l].act = (l + 1 < _nLayers) ? ACT_RELU : ACT_NONE;
        _layers[l].input_scale = (l == 0) ? _inputScale : _outputScales[l - 1];
        _layers[l].input_zero_point = (l == 0) ? _inputZeroPoint : _outputZeroPoints[l - 1];
        _layers[l].output_scale = _outputScales[l];
        _layers[l].output_zero_point = _outputZeroPoints[l];
        _layers[l].activation_min = -128;
        _layers[l].activation_max = 127;
    }
}

bool NoodleAIModelBL702::loadSlot(char slot, bool allowStaged)
{
    clearRuntime();
    _lastLoadError = LOAD_OK;

    if (slot != 'A' && slot != 'B') {
        _lastLoadError = LOAD_ERR_NO_SLOT;
        return false;
    }

    nai_slotfs_set_staged_read(slot, allowStaged);

    if (!readConfig(slot)) {
        _lastLoadError = LOAD_ERR_CONFIG;
        nai_slotfs_set_staged_read(slot, false);
        clearRuntime();
        return false;
    }

    if (!validateParameterFiles(slot)) {
        _lastLoadError = LOAD_ERR_FILES;
        nai_slotfs_set_staged_read(slot, false);
        clearRuntime();
        return false;
    }

    if (!loadScaler(slot)) {
        /* loadScaler() sets the more precise scaler error. */
        nai_slotfs_set_staged_read(slot, false);
        clearRuntime();
        return false;
    }

    configureLayers(slot);
    initTensorsIfNeeded();

    if (!noodle_i8_tensor_require_vector(
            &_actA, _inputDim, _inputScale, _inputZeroPoint)) {
        _lastLoadError = LOAD_ERR_TENSOR_A;
        nai_slotfs_set_staged_read(slot, false);
        clearRuntime();
        return false;
    }

    if (!noodle_i8_tensor_require_vector(
            &_actB, _dims[1], _outputScales[0], _outputZeroPoints[0])) {
        _lastLoadError = LOAD_ERR_TENSOR_B;
        nai_slotfs_set_staged_read(slot, false);
        clearRuntime();
        return false;
    }

    nai_slotfs_set_staged_read(slot, false);
    _slot = slot;
    _loaded = true;
    _lastLoadError = LOAD_OK;
    return true;
}

bool NoodleAIModelBL702::loadActive()
{
    /*
     * Mode 2 is deliberately single-slot.  The old A/B fallback only repeated
     * the same physical slot through the compatibility virtual-name layer and
     * could overwrite the useful first failure code.
     */
    const char slot = nai_slotfs_active_slot();
    if (slot != 'A') {
        clearRuntime();
        _lastLoadError = LOAD_ERR_NO_SLOT;
        return false;
    }

    return loadSlot('A', false);
}

bool NoodleAIModelBL702::predict(const float *x,
                                 uint16_t &classIndex,
                                 float &confidence)
{
    if (!_loaded || !x || !_tensorsInitialized) return false;

    int8_t *input = noodle_i8_tensor_require_vector(
        &_actA, _inputDim, _inputScale, _inputZeroPoint);
    if (!input) return false;

    for (uint16_t i = 0; i < _inputDim; ++i) {
        const float standardized = (x[i] - _mean[i]) / _scale[i];
        input[i] = (int8_t)noodle_quantize_float(
            standardized, _inputScale, _inputZeroPoint);
    }

    NoodleTensor *current = &_actA;
    NoodleTensor *next = &_actB;

    for (uint16_t l = 0; l < _nLayers; ++l) {
        if (noodle_fcn(current, next, _layers[l]) != _dims[l + 1])
            return false;
        NoodleTensor *tmp = current;
        current = next;
        next = tmp;
    }

    if (_binaryClassifier) {
        if (noodle_sigmoid(current) != 1) return false;
        const int8_t q = current->buffer.data[0];
        float p1 = noodle_dequantize_int8(q, current->scale, current->zero_point);
        if (p1 < 0.0f) p1 = 0.0f;
        if (p1 > 1.0f) p1 = 1.0f;
        if (p1 >= 0.5f) {
            classIndex = 1;
            confidence = p1;
        } else {
            classIndex = 0;
            confidence = 1.0f - p1;
        }
        return true;
    }

    if (noodle_soft_max(current) != _nClasses) return false;

    classIndex = 0;
    confidence = noodle_dequantize_int8(
        current->buffer.data[0], current->scale, current->zero_point);

    for (uint16_t i = 1; i < _nClasses; ++i) {
        const float p = noodle_dequantize_int8(
            current->buffer.data[i], current->scale, current->zero_point);
        if (p > confidence) {
            confidence = p;
            classIndex = i;
        }
    }
    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 1.0f) confidence = 1.0f;
    return true;
}

uint16_t NoodleAIModelBL702::representationChannelCount(InputRepresentation rep)
{
    switch (rep) {
    case REP_ACCEL: return 3;
    case REP_GYRO: return 3;
    case REP_ACCEL_GYRO: return 6;
    case REP_QUATERNION: return 4;
    case REP_VELOCITY: return 3;
    case REP_VELOCITY_QUATERNION: return 7;
    default: return 0;
    }
}

uint16_t NoodleAIModelBL702::channelCount() const
{
    return representationChannelCount(_representation);
}

const char *NoodleAIModelBL702::representationName() const
{
    switch (_representation) {
    case REP_ACCEL: return "accel";
    case REP_GYRO: return "gyro";
    case REP_ACCEL_GYRO: return "accel+gyro";
    case REP_QUATERNION: return "quaternion";
    case REP_VELOCITY: return "velocity";
    case REP_VELOCITY_QUATERNION: return "velocity+quaternion";
    default: return "unknown";
    }
}

const char *NoodleAIModelBL702::label(uint16_t index) const
{
    if (!_loaded || index >= _nClasses) return "";
    return _labels[index];
}

size_t NoodleAIModelBL702::tensorArenaBytes() const
{
    return noodle_buffer_arena_used_bytes();
}
