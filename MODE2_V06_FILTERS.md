# Mode 2 v0.6 — model-defined IMU filters

Training and device inference now share the same first-order filter settings stored in `cfg.bin` as the optional `FLT1` extension.

- Accelerometer: 0.10–8 Hz band-pass
- Gyroscope: 0.10–8 Hz band-pass
- Accel + Gyro: 0.10–8 Hz band-pass
- Relative Quaternion: 8 Hz low-pass before Madgwick
- Estimated Velocity: 8 Hz low-pass before Madgwick
- Velocity + Quaternion: 8 Hz low-pass before Madgwick

The confidence threshold sweep and INT8 calibration operate on the filtered representation, so the exact preprocessing used for training is deployed with the model.
