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
