# NoodleAI Mode 2 packed INT8 browser update

The browser trainer uses the unchanged INT8 PTQ/calibration equations, threshold sweep, and Q8M1/FLT1 config. Each Dense layer now writes `wNN.bin` (INT8 output-major) and `pNN.bin` (int32 little-endian bias, multiplier, shift, 12 bytes per output). Existing `w/b/m/s` `.nai` files can still be loaded for redeployment. The browser requires device confirmation `CAPS:PACKED_INT8:1` before `DEPLOY_BEGIN`, so attempting to deploy to old firmware cannot erase its stored model. Upgrade the M0Sense firmware first.

See the combined package's README for build and test details.
