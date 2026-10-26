#include "nai_infer_bridge.h"
#include "NoodleAIModelBL702.h"
#include "nai_flash_port.h"

#include <math.h>
#include <stddef.h>

/*
 * NAI4 motion-preprocessing v2: model-defined first-order filtering.
 *
 * These constants intentionally match the browser trainer (js/core.js):
 *   MADGWICK_BETA = 0.10
 *   GRAVITY_MPS2  = 9.80665
 *
 * The browser derives every motion representation from a raw six-axis window.
 * Device inference must do exactly the same thing before Noodle sees the input.
 */
static constexpr float NAI_MADGWICK_BETA = 0.10f;
static constexpr float NAI_GRAVITY_MPS2 = 9.80665f;
static constexpr float NAI_DEG_TO_RAD = 0.01745329251994329577f;
static constexpr uint16_t NAI_RAW_CHANNELS = 6u;

static NoodleAIModelBL702 s_model;
static float s_repr[NoodleAIModelBL702::MAX_DIM];

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Quat {
    float w;
    float x;
    float y;
    float z;
};

struct Filter1State {
    float x_prev;
    float hp_prev;
    float lp_prev;
    bool initialized;
};

static float hp_alpha(float cutoff_hz, float sample_rate_hz)
{
    const float dt = 1.0f / sample_rate_hz;
    const float tau = 1.0f / (2.0f * 3.14159265358979323846f * cutoff_hz);
    return tau / (tau + dt);
}

static float lp_beta(float cutoff_hz, float sample_rate_hz)
{
    const float dt = 1.0f / sample_rate_hz;
    const float tau = 1.0f / (2.0f * 3.14159265358979323846f * cutoff_hz);
    return dt / (tau + dt);
}

static float filter_one(float x, Filter1State &st,
                        uint16_t flags, float hp_a, float lp_b)
{
    const bool use_hp = (flags & NoodleAIModelBL702::FILTER_HP) != 0;
    const bool use_lp = (flags & NoodleAIModelBL702::FILTER_LP) != 0;

    if (!st.initialized) {
        st.x_prev = x;
        st.hp_prev = 0.0f;
        float v = use_hp ? 0.0f : x;
        st.lp_prev = v;
        st.initialized = true;
        return v;
    }

    float v = x;
    if (use_hp) {
        const float hp = hp_a * (st.hp_prev + x - st.x_prev);
        st.x_prev = x;
        st.hp_prev = hp;
        v = hp;
    }

    if (use_lp) {
        st.lp_prev += lp_b * (v - st.lp_prev);
        v = st.lp_prev;
    }
    return v;
}

static Vec3 raw_vec3(const float *raw6, uint16_t k, uint16_t offset)
{
    const size_t b = (size_t)k * NAI_RAW_CHANNELS + offset;
    return {raw6[b], raw6[b + 1u], raw6[b + 2u]};
}

static float vec_norm(Vec3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static float vec_dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 vec_cross(Vec3 a, Vec3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

static Quat quat_norm(Quat q)
{
    const float n = sqrtf(q.w * q.w + q.x * q.x +
                          q.y * q.y + q.z * q.z);
    if (!isfinite(n) || n < 1.0e-12f) {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }
    const float inv = 1.0f / n;
    return {q.w * inv, q.x * inv, q.y * inv, q.z * inv};
}

static Quat quat_conj(Quat q)
{
    return {q.w, -q.x, -q.y, -q.z};
}

static Quat quat_mul(Quat a, Quat b)
{
    return {
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    };
}

static float quat_dot(Quat a, Quat b)
{
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 quat_rotate(Quat q, Vec3 v)
{
    const Quat p{0.0f, v.x, v.y, v.z};
    const Quat r = quat_mul(quat_mul(q, p), quat_conj(q));
    return {r.x, r.y, r.z};
}

/* Match browser quatAlignAccelToWorldZ(). */
static Quat quat_align_accel_to_world_z(Vec3 acc)
{
    float n = vec_norm(acc);
    if (!isfinite(n) || n < 1.0e-8f) {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }

    Vec3 u{acc.x / n, acc.y / n, acc.z / n};
    const Vec3 world_z{0.0f, 0.0f, 1.0f};
    const float d = vec_dot(u, world_z);

    if (d < -0.999999f) {
        const Vec3 base = fabsf(u.x) < 0.9f
                              ? Vec3{1.0f, 0.0f, 0.0f}
                              : Vec3{0.0f, 1.0f, 0.0f};
        Vec3 axis = vec_cross(u, base);
        n = vec_norm(axis);
        if (!isfinite(n) || n < 1.0e-12f) {
            return {1.0f, 0.0f, 0.0f, 0.0f};
        }
        axis.x /= n;
        axis.y /= n;
        axis.z /= n;
        return {0.0f, axis.x, axis.y, axis.z};
    }

    const Vec3 c = vec_cross(u, world_z);
    return quat_norm({1.0f + d, c.x, c.y, c.z});
}

/* Match browser madgwickUpdateImu(). Gyroscope input is degrees/second. */
static Quat madgwick_update_imu(Quat q, Vec3 accel, Vec3 gyro_dps,
                                float dt)
{
    float q1 = q.w;
    float q2 = q.x;
    float q3 = q.y;
    float q4 = q.z;

    const float gx = gyro_dps.x * NAI_DEG_TO_RAD;
    const float gy = gyro_dps.y * NAI_DEG_TO_RAD;
    const float gz = gyro_dps.z * NAI_DEG_TO_RAD;

    float qdot1 = 0.5f * (-q2 * gx - q3 * gy - q4 * gz);
    float qdot2 = 0.5f * ( q1 * gx + q3 * gz - q4 * gy);
    float qdot3 = 0.5f * ( q1 * gy - q2 * gz + q4 * gx);
    float qdot4 = 0.5f * ( q1 * gz + q2 * gy - q3 * gx);

    float ax = accel.x;
    float ay = accel.y;
    float az = accel.z;
    const float an = sqrtf(ax * ax + ay * ay + az * az);

    if (isfinite(an) && an > 1.0e-8f) {
        ax /= an;
        ay /= an;
        az /= an;

        const float _2q1 = 2.0f * q1;
        const float _2q2 = 2.0f * q2;
        const float _2q3 = 2.0f * q3;
        const float _2q4 = 2.0f * q4;
        const float _4q1 = 4.0f * q1;
        const float _4q2 = 4.0f * q2;
        const float _4q3 = 4.0f * q3;
        const float _8q2 = 8.0f * q2;
        const float _8q3 = 8.0f * q3;
        const float q1q1 = q1 * q1;
        const float q2q2 = q2 * q2;
        const float q3q3 = q3 * q3;
        const float q4q4 = q4 * q4;

        float s1 = _4q1 * q3q3 + _2q3 * ax +
                   _4q1 * q2q2 - _2q2 * ay;
        float s2 = _4q2 * q4q4 - _2q4 * ax +
                   4.0f * q1q1 * q2 - _2q1 * ay - _4q2 +
                   _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az;
        float s3 = 4.0f * q1q1 * q3 + _2q1 * ax +
                   _4q3 * q4q4 - _2q4 * ay - _4q3 +
                   _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az;
        float s4 = 4.0f * q2q2 * q4 - _2q2 * ax +
                   4.0f * q3q3 * q4 - _2q3 * ay;

        const float sn = sqrtf(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4);
        if (isfinite(sn) && sn > 1.0e-12f) {
            const float inv = 1.0f / sn;
            s1 *= inv;
            s2 *= inv;
            s3 *= inv;
            s4 *= inv;
            qdot1 -= NAI_MADGWICK_BETA * s1;
            qdot2 -= NAI_MADGWICK_BETA * s2;
            qdot3 -= NAI_MADGWICK_BETA * s3;
            qdot4 -= NAI_MADGWICK_BETA * s4;
        }
    }

    return quat_norm({
        q1 + qdot1 * dt,
        q2 + qdot2 * dt,
        q3 + qdot3 * dt,
        q4 + qdot4 * dt,
    });
}

/*
 * Raw accel/gyro NAI4 representations are centered per channel after
 * resampling. Live inference already has exactly N samples, so resampling is
 * the identity operation and only the per-window centering remains.
 */
static bool build_centered_raw(const float *raw6, uint16_t n,
                               uint16_t first_channel, uint16_t channels)
{
    if (!raw6 || n == 0 || channels == 0 ||
        (uint32_t)n * channels > NoodleAIModelBL702::MAX_DIM) {
        return false;
    }

    float means[NAI_RAW_CHANNELS]{};
    for (uint16_t k = 0; k < n; ++k) {
        const size_t b = (size_t)k * NAI_RAW_CHANNELS + first_channel;
        for (uint16_t c = 0; c < channels; ++c) {
            means[c] += raw6[b + c];
        }
    }
    const float inv_n = 1.0f / (float)n;
    for (uint16_t c = 0; c < channels; ++c) {
        means[c] *= inv_n;
    }

    size_t out = 0;
    for (uint16_t k = 0; k < n; ++k) {
        const size_t b = (size_t)k * NAI_RAW_CHANNELS + first_channel;
        for (uint16_t c = 0; c < channels; ++c) {
            s_repr[out++] = raw6[b + c] - means[c];
        }
    }
    return out == s_model.inputDim();
}


static bool build_filtered_direct(const float *raw6, uint16_t n,
                                  uint16_t first_channel,
                                  uint16_t channels)
{
    if (!raw6 || n == 0 || channels == 0 ||
        (uint32_t)n * channels != s_model.inputDim()) {
        return false;
    }

    const uint16_t flags = s_model.filterFlags();
    if (flags == NoodleAIModelBL702::FILTER_NONE) {
        return build_centered_raw(raw6, n, first_channel, channels);
    }

    const float fs = (float)s_model.sampleRateHz();
    if (fs <= 0.0f) return false;
    const float hp_a = (flags & NoodleAIModelBL702::FILTER_HP)
                           ? hp_alpha(s_model.highpassCutoffHz(), fs)
                           : 0.0f;
    const float lp_b = (flags & NoodleAIModelBL702::FILTER_LP)
                           ? lp_beta(s_model.lowpassCutoffHz(), fs)
                           : 0.0f;

    Filter1State state[NAI_RAW_CHANNELS]{};
    size_t out = 0;
    for (uint16_t k = 0; k < n; ++k) {
        const size_t b = (size_t)k * NAI_RAW_CHANNELS + first_channel;
        for (uint16_t c = 0; c < channels; ++c) {
            s_repr[out++] = filter_one(raw6[b + c], state[c],
                                       flags, hp_a, lp_b);
        }
    }
    return out == s_model.inputDim();
}

static void store_quat(float *dst, Quat q)
{
    q = quat_norm(q);
    dst[0] = q.w;
    dst[1] = q.x;
    dst[2] = q.y;
    dst[3] = q.z;
}

/* Match browser deriveQuatVelocity() for a live N-sample raw window. */
static bool build_motion_repr(const float *raw6, uint16_t n,
                              NoodleAIModelBL702::InputRepresentation rep,
                              uint16_t sample_rate_hz)
{
    if (!raw6 || n < 2 || sample_rate_hz == 0) return false;

    const bool want_vel =
        rep == NoodleAIModelBL702::REP_VELOCITY ||
        rep == NoodleAIModelBL702::REP_VELOCITY_QUATERNION;
    const bool want_quat =
        rep == NoodleAIModelBL702::REP_QUATERNION ||
        rep == NoodleAIModelBL702::REP_VELOCITY_QUATERNION;

    if (!want_vel && !want_quat) return false;

    const uint16_t stride =
        rep == NoodleAIModelBL702::REP_QUATERNION ? 4u :
        rep == NoodleAIModelBL702::REP_VELOCITY ? 3u : 7u;
    if ((uint32_t)n * stride != s_model.inputDim()) return false;

    const float dt = 1.0f / (float)sample_rate_hz;
    const uint16_t filter_flags = s_model.filterFlags();
    if ((filter_flags & NoodleAIModelBL702::FILTER_HP) != 0) return false;
    const float lp_b = (filter_flags & NoodleAIModelBL702::FILTER_LP)
                           ? lp_beta(s_model.lowpassCutoffHz(),
                                     (float)sample_rate_hz)
                           : 0.0f;

    Filter1State fstate[NAI_RAW_CHANNELS]{};
    auto filtered_vec3 = [&](uint16_t k, uint16_t offset) -> Vec3 {
        const size_t b = (size_t)k * NAI_RAW_CHANNELS + offset;
        return {
            filter_one(raw6[b],     fstate[offset],     filter_flags, 0.0f, lp_b),
            filter_one(raw6[b + 1], fstate[offset + 1], filter_flags, 0.0f, lp_b),
            filter_one(raw6[b + 2], fstate[offset + 2], filter_flags, 0.0f, lp_b),
        };
    };

    Vec3 accel_k = filtered_vec3(0, 0);
    Vec3 gyro_k = filtered_vec3(0, 3);
    Quat q_abs = quat_align_accel_to_world_z(accel_k);
    const Quat q_ref = q_abs;
    Quat q_prev_rel{1.0f, 0.0f, 0.0f, 0.0f};

    if (want_quat) {
        float *qdst = rep == NoodleAIModelBL702::REP_QUATERNION
                          ? &s_repr[0]
                          : &s_repr[3];
        store_quat(qdst, q_prev_rel);
    }
    if (want_vel) {
        s_repr[0] = 0.0f;
        s_repr[1] = 0.0f;
        s_repr[2] = 0.0f;
    }

    Vec3 a_prev = quat_rotate(q_abs, accel_k);
    a_prev.x *= NAI_GRAVITY_MPS2;
    a_prev.y *= NAI_GRAVITY_MPS2;
    a_prev.z = (a_prev.z - 1.0f) * NAI_GRAVITY_MPS2;

    Vec3 v_prev{0.0f, 0.0f, 0.0f};

    for (uint16_t k = 1; k < n; ++k) {
        accel_k = filtered_vec3(k, 0);
        gyro_k = filtered_vec3(k, 3);
        q_abs = madgwick_update_imu(q_abs, accel_k, gyro_k, dt);

        if (want_quat) {
            Quat q_rel = quat_norm(quat_mul(quat_conj(q_ref), q_abs));
            if (quat_dot(q_rel, q_prev_rel) < 0.0f) {
                q_rel.w = -q_rel.w;
                q_rel.x = -q_rel.x;
                q_rel.y = -q_rel.y;
                q_rel.z = -q_rel.z;
            }
            q_prev_rel = q_rel;
            float *qdst = rep == NoodleAIModelBL702::REP_QUATERNION
                              ? &s_repr[(size_t)k * 4u]
                              : &s_repr[(size_t)k * 7u + 3u];
            store_quat(qdst, q_rel);
        }

        Vec3 a = quat_rotate(q_abs, accel_k);
        a.x *= NAI_GRAVITY_MPS2;
        a.y *= NAI_GRAVITY_MPS2;
        a.z = (a.z - 1.0f) * NAI_GRAVITY_MPS2;

        if (want_vel) {
            Vec3 v{
                v_prev.x + 0.5f * (a_prev.x + a.x) * dt,
                v_prev.y + 0.5f * (a_prev.y + a.y) * dt,
                v_prev.z + 0.5f * (a_prev.z + a.z) * dt,
            };
            float *vdst = rep == NoodleAIModelBL702::REP_VELOCITY
                              ? &s_repr[(size_t)k * 3u]
                              : &s_repr[(size_t)k * 7u];
            vdst[0] = v.x;
            vdst[1] = v.y;
            vdst[2] = v.z;
            v_prev = v;
        }

        a_prev = a;
    }

    /* Browser endpoint drift correction: v[k] -= k/(N-1) * v[N-1]. */
    if (want_vel) {
        const Vec3 end = v_prev;
        const float denom = (float)(n - 1u);
        for (uint16_t k = 0; k < n; ++k) {
            const float alpha = (float)k / denom;
            float *vdst = rep == NoodleAIModelBL702::REP_VELOCITY
                              ? &s_repr[(size_t)k * 3u]
                              : &s_repr[(size_t)k * 7u];
            vdst[0] -= alpha * end.x;
            vdst[1] -= alpha * end.y;
            vdst[2] -= alpha * end.z;
        }

        float *first = &s_repr[0];
        float *last = rep == NoodleAIModelBL702::REP_VELOCITY
                          ? &s_repr[(size_t)(n - 1u) * 3u]
                          : &s_repr[(size_t)(n - 1u) * 7u];
        first[0] = first[1] = first[2] = 0.0f;
        last[0] = last[1] = last[2] = 0.0f;
    }

    return true;
}

static bool build_representation(const float *raw6, uint16_t n)
{
    if (!s_model.loaded() || !raw6 || n != s_model.windowLength()) {
        return false;
    }

    switch (s_model.representation()) {
    case NoodleAIModelBL702::REP_ACCEL:
        return build_filtered_direct(raw6, n, 0u, 3u);
    case NoodleAIModelBL702::REP_GYRO:
        return build_filtered_direct(raw6, n, 3u, 3u);
    case NoodleAIModelBL702::REP_ACCEL_GYRO:
        return build_filtered_direct(raw6, n, 0u, 6u);
    case NoodleAIModelBL702::REP_QUATERNION:
    case NoodleAIModelBL702::REP_VELOCITY:
    case NoodleAIModelBL702::REP_VELOCITY_QUATERNION:
        return build_motion_repr(raw6, n, s_model.representation(),
                                 s_model.sampleRateHz());
    default:
        return false;
    }
}

extern "C" int nai_infer_load_active(void)
{
    s_model.clearRuntime();

    /*
     * A model may have been programmed only milliseconds ago. Flush stale XIP
     * lines once here before cfg.bin/scaler/weights are opened. This is cheap
     * compared with doing it on every file-backed FCN block read.
     */
    nai_flash_flush_cache();

    if (!s_model.loadActive()) {
        return s_model.lastLoadError();
    }

    const uint16_t ch = s_model.channelCount();
    if (s_model.sampleRateHz() != 50u ||
        ch == 0u ||
        s_model.inputDim() !=
            (uint32_t)s_model.windowLength() * ch) {
        s_model.clearRuntime();
        return NoodleAIModelBL702::LOAD_ERR_POSTCHECK;
    }

    return 0;
}

#if defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
extern "C" uint32_t nai_infer_last_fcn_us(void)
{
    return s_model.lastFcnUs();
}
#endif

extern "C" void nai_infer_clear(void)
{
    s_model.clearRuntime();
}

extern "C" bool nai_infer_loaded(void)
{
    return s_model.loaded();
}

extern "C" uint16_t nai_infer_window_length(void)
{
    return s_model.loaded() ? s_model.windowLength() : 0u;
}

extern "C" uint16_t nai_infer_sample_rate_hz(void)
{
    return s_model.loaded() ? s_model.sampleRateHz() : 0u;
}

extern "C" uint16_t nai_infer_stride_samples(void)
{
    return s_model.loaded() ? s_model.strideSamples() : 0u;
}

extern "C" float nai_infer_confidence_threshold(void)
{
    return s_model.loaded() ? s_model.confidenceThreshold() : 1.0f;
}

extern "C" uint16_t nai_infer_channel_count(void)
{
    return s_model.loaded() ? s_model.channelCount() : 0u;
}

extern "C" uint16_t nai_infer_input_dim(void)
{
    return s_model.loaded() ? s_model.inputDim() : 0u;
}

extern "C" uint16_t nai_infer_class_count(void)
{
    return s_model.loaded() ? s_model.classCount() : 0u;
}

extern "C" uint16_t nai_infer_representation(void)
{
    return s_model.loaded() ? (uint16_t)s_model.representation() : 0u;
}

extern "C" const char *nai_infer_label(uint16_t index)
{
    return s_model.loaded() ? s_model.label(index) : "";
}

extern "C" int nai_infer_predict(
    const float *x,
    uint16_t *class_index,
    float *confidence)
{
    if (!x || !class_index || !confidence || !s_model.loaded()) {
        return -1;
    }

    uint16_t c = 0;
    float p = 0.0f;

    if (!s_model.predict(x, c, p)) {
        return -2;
    }

    *class_index = c;
    *confidence = p;
    return 0;
}

extern "C" int nai_infer_predict_raw6(
    const float *raw6,
    uint16_t sample_count,
    uint16_t *class_index,
    float *confidence)
{
    if (!raw6 || !class_index || !confidence || !s_model.loaded()) {
        return -1;
    }
    if (!build_representation(raw6, sample_count)) {
        return -2;
    }

    return nai_infer_predict(s_repr, class_index, confidence);
}
