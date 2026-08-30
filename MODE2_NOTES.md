# NoodleAI Mode 2 Web v0.1

This build changes dataset capture from BOOT-delimited single gestures to continuous browser-controlled recording.

## Workflow

1. Connect M0Sense.
2. Open **Dataset & Train**.
3. Define labels (include an `IDLE` class if you want continuous recognition to have a neutral state).
4. Default Mode 2 geometry matches the firmware:
   - 50 Hz
   - 75-sample window = 1.5 s
   - 5-sample stride = 0.1 s
   - Accel + Gyro = 450 MLP inputs
5. Lock setup.
6. Select a target label.
7. Press **START recording**.
8. Perform that gesture repeatedly.
9. Press **STOP**.
10. The browser slices the captured stream into overlapping windows.
11. Press **Save windows** or **Discard**.
12. Repeat for every class, then train.

The device is never told to start or stop recording. It streams QMI8658 continuously; START/STOP only labels the incoming browser-side stream.

## Important evaluation note

Windows from the same recording session overlap heavily and are therefore correlated. This v0.1 keeps the existing per-window stratified train/validation split for simplicity. For publication-quality evaluation, the next improvement should split by recording session so windows from one session cannot appear in both training and validation sets.

## Deployment

Deployment is intentionally disabled in this web build. The next BL702 firmware stage will implement one-slot model storage with no A/B slots, then the Deploy button can be re-enabled.


## v0.2 — extend loaded datasets

A loaded `.npz` is no longer a dead-end.

After loading, **Unlock setup** preserves all existing windows and lets you:

- add new class labels,
- record more data for existing classes,
- record data for newly added classes,
- change the stride used for future recordings.

The existing window length is intentionally immutable once saved windows exist,
because mixing different input dimensions in one MLP dataset would be invalid.

An existing class that already owns saved windows cannot be removed. Empty
newly-added classes can be removed safely.


## Deployment stage

This web build supports the Mode 2 v0.2 firmware single-slot protocol.

- raw model payload capacity: 188 KiB
- no A/B
- each file has CRC32
- firmware reads each file back from flash and verifies CRC
- final status is `MODEL_STORED`

`MODEL_STORED` deliberately means storage verification only. Noodle parsing and
continuous inference are the next firmware stage.

The Deploy tab can also load an existing `.nai` directly, so a browser refresh
does not force retraining.


## v0.4 web update for firmware v0.3 inference

- TRAINING MODE / INFERENCE MODE are enabled immediately after BLE connection.
- `MODEL:READY:N:D:K` updates the device-model summary.
- `P:<index>:<confidence>` is rendered with labels from the trained or loaded
  `.nai` package when available.
- The same single-slot deployment protocol is unchanged.
