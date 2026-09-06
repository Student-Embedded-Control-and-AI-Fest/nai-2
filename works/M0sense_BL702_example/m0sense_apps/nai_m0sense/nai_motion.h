#pragma once
#include <stdint.h>
#include "NoodleAIModelBL702.h"

bool nai_build_inference_representation(
    NoodleAIModelBL702::InputRepresentation rep,
    const float *raw6,
    uint16_t rawSamples,
    uint16_t sampleRateHz,
    uint16_t normalizedSamples,
    float *out,
    float *derivedScratch);
