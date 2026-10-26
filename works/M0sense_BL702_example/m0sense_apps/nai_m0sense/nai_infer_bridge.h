#pragma once
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tiny C bridge around the C++ NoodleAIModelBL702 runtime.
 *
 * Mode 2 v0.4.1 always acquires raw ax/ay/az/gx/gy/gz at 50 Hz. The bridge
 * then reproduces the NAI4 representation selected in cfg.bin before running
 * Noodle INT8 inference:
 *   1 accel
 *   2 gyro
 *   3 accel+gyro
 *   4 relative quaternion
 *   5 estimated velocity
 *   6 velocity+quaternion
 */
int nai_infer_load_active(void);
/* The FCN-only timing is available when NAI_BENCH_FCN is enabled. */
#if defined(NAI_BENCH_FCN) && NAI_BENCH_FCN
uint32_t nai_infer_last_fcn_us(void);
#endif
void nai_infer_clear(void);
bool nai_infer_loaded(void);
uint16_t nai_infer_window_length(void);
uint16_t nai_infer_sample_rate_hz(void);
uint16_t nai_infer_stride_samples(void);
float nai_infer_confidence_threshold(void);
uint16_t nai_infer_channel_count(void);
uint16_t nai_infer_input_dim(void);
uint16_t nai_infer_class_count(void);
uint16_t nai_infer_representation(void);
const char *nai_infer_label(uint16_t index);

/* Predict an already-prepared model input vector. */
int nai_infer_predict(const float *x, uint16_t *class_index, float *confidence);

/*
 * Preferred Mode 2 entry point: raw chronological six-axis window -> selected
 * NAI4 representation -> StandardScaler -> INT8 Noodle MLP.
 */
int nai_infer_predict_raw6(const float *raw6,
                           uint16_t sample_count,
                           uint16_t *class_index,
                           float *confidence);

#ifdef __cplusplus
}
#endif
