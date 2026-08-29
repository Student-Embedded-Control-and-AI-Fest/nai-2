# NoodleAI Web

A static, client-side NoodleAI application for the M0Sense BL702/QMI8658 platform.

**Stream. Record. Window. Train. Deploy. Infer.**

The application mirrors the desktop NoodleAI workflow:

- **Live IMU** — raw 6-axis accelerometer + gyroscope plots over Web Bluetooth.
- **Dataset & Train (Mode 2)** — define labels, press START/STOP around a repetitive continuous gesture session, automatically slice the IMU stream into overlapping fixed windows, save/load Python-compatible `.npz` datasets, and train an N-layer MLP with TensorFlow.js.
- **Training Curves** — train/validation loss and accuracy history.
- **Deploy & Infer** — save the binary `.nai` package, deploy it transactionally over BLE, switch device modes, and show the latest Noodle prediction.

There is no application server. TensorFlow.js training, dataset processing, NAI4 packaging, and BLE communication all happen in the browser.

## Run locally

Web Bluetooth needs a secure browser context. `localhost` is allowed for development:

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

The Python command is only a static file server. It does **not** train models or handle Bluetooth.

## Publish with GitHub Pages

1. Create a GitHub repository and put these files in the repository root.
2. Push to `main`.
3. Open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select `main` and `/ (root)`.
6. Save and open the HTTPS Pages URL GitHub gives you.

GitHub Pages provides the secure HTTPS context Web Bluetooth expects.

## Browser support

Use a Chromium-based browser with Web Bluetooth support. On Chrome/Linux, `navigator.bluetooth` may require enabling **Experimental Web Platform features** in `chrome://flags`.

## Dependencies

Pinned CDN dependencies:

- TensorFlow.js 4.22.0
- JSZip 3.10.1

Because they are loaded from CDN, the first page load needs internet access. Model training and BLE data stay local in the browser.

## NAI4 representations

The browser stores each Mode 2 raw 6-axis window and can train any supported representation from the same dataset:

- Accelerometer
- Gyroscope
- Accel + Gyro
- Relative Quaternion
- Estimated Velocity
- Velocity + Quaternion

Derived representations mirror the current NAI4 motion pipeline (6-axis Madgwick orientation, gravity compensation, velocity integration with endpoint correction, then temporal normalization).

## Dataset compatibility

Saved `.npz` files contain the same NAI4 fields as the desktop GUI, including `X`, `y`, `labels`, `normalized_length`, `raw_lengths`, `durations_ms`, and—when available—`raw_data` + `raw_offsets`.

Older normalized-only NAI3 datasets can still train the raw accelerometer/gyro representations. Quaternion/velocity modes require retained raw 6-axis window data.

## Model deployment

The Mode 2 web build can already train and save the binary `.nai` package. **Device deployment is intentionally disabled in this v0.1 web build** while the BL702 firmware is being simplified to one model area with no A/B slots. The next firmware stage will reconnect the existing MODEL characteristic to that one-slot storage path.

The NAI4 archive contains binary float32 files such as:

```text
cfg.bin
mean.bin
scale.bin
w00.bin
b00.bin
...
```

TensorFlow.js Dense kernels `[I,O]` are transposed to the Noodle `[O,I]` layout before serialization.


## Mode 2 continuous training

The current Mode 2 defaults match the M0Sense firmware:

- Sampling: **50 Hz**
- Window: **75 samples** = **1.5 s**
- Stride: **5 samples** = **0.1 s**
- Raw channels: **6** (`ax ay az gx gy gz`)
- Raw input size for Accel + Gyro: **450 values**

Training capture is browser-controlled. The device streams continuously; pressing **START** begins labeling the incoming stream with the selected class and **STOP** ends the session. The browser then creates overlapping windows using the configured window length and stride.
