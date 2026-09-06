NoodleAI Mode 2 v0.2 — SINGLE-SLOT DEPLOYMENT
================================================

This stage adds model TRANSPORT + FLASH STORAGE only.

Still working:
- NoodleAI BLE
- QMI8658 continuous 50 Hz streaming
- Mode 2 rolling-window skeleton

New:
- one physical flash model region
- NO A/B
- NO metadata journal
- per-file incoming CRC32
- read-back CRC32 from flash
- staged header
- final one-word commit marker

Not yet:
- Noodle model loading
- inference
- prediction notifications
- INT8

Flash map
---------
512 KiB physical flash:

  0x00000 .. 0x01FFF   boot/platform
  0x02000 .. 0x4FFFF   application, MAX 312 KiB
  0x50000 .. 0x7FFFF   model region, 192 KiB

Inside model region:

  first 4 KiB          header
  remaining 188 KiB    NAI4 raw files

Current float32 example is ~165.2 KiB, so it fits.

VERY IMPORTANT BUILD CHECK
--------------------------
After building, check binary size BEFORE flashing:

  cd ~/works/M0sense_BL702_example
  stat -c '%s bytes' \
    bl_mcu_sdk/out/m0sense_apps/nai_m0sense/nai_m0sense_bl702.bin

The .bin MUST be <= 319488 bytes.

If it is larger, DO NOT FLASH this layout. Tell ChatGPT the exact size.

Install
-------
Extract this ZIP and copy ALL source/build files into:

  ~/works/M0sense_BL702_example/m0sense_apps/nai_m0sense/

Keep your existing:
  nai_imu.cpp
  nai_imu.h
  io_def.h

Build:

  cd ~/works/M0sense_BL702_example
  rm -rf bl_mcu_sdk/build
  ./build.sh m0sense_apps/nai_m0sense

Flash:

  bflb-mcu-tool-uart \
    --chipname=bl702 \
    --port=/dev/ttyACM0 \
    --baudrate=2000000 \
    --firmware=bl_mcu_sdk/out/m0sense_apps/nai_m0sense/nai_m0sense_bl702.bin

Protocol
--------
Web sends:

  DEPLOY_BEGIN
  FILE_BEGIN(cfg.bin, size, crc)
  raw MODEL chunks
  FILE_END
  ...repeat...
  DEPLOY_COMMIT

Successful final status:

  MODEL_STORED

MODEL_STORED means:
- all files arrived
- each file CRC matched
- each file was read back from flash and CRC matched
- slot header was verified
- commit marker was written

It does NOT yet mean that Noodle parsed or ran the model.
That is v0.3.


v0.2.1: deployment erase is now 4 KiB sector-by-sector with BLE yielding between sectors.


v0.2.2 commit fix
-----------------
All files already transferred and CRC-verified correctly.

The final slot commit is now incremental too:

  COMMIT:START
  write <=128 header bytes
  yield to BLE
  write next <=128 bytes
  yield
  verify staged header
  yield
  write 4-byte commit word
  yield
  verify final header
  MODEL_STORED

The web app v0.3.1 can be reused unchanged.


v0.2.3 stack-safety fix
-----------------------
The upload path already reaches the final commit, so transport + file CRC are
working.

Important discovery: the current app task stack depth was only 256, not 512.
The final commit/verify path uses a ~560-byte NaiSlotHeader and previously
created additional temporary header objects on the task stack.

Changes:
- app task stack depth: 256 -> 1024
- one NaiSlotHeader scratch buffer moved to BSS/global storage
- nai_slotfs_read_header() reads directly into caller-provided storage
- lookup / active-slot / staged verify / final verify reuse the global scratch

Web app v0.3.1 remains unchanged.


v0.2.4 cache-safe header verification
-------------------------------------
ERR:COMMIT:-8 means the slot header write succeeded, but immediate header
validation failed.

The BL702 HAL's own flash_read_via_xip() explicitly flushes L1 cache before
and after cached XIP reads. Our custom raw-XIP memcpy reader did not.

That is especially dangerous for the header sector because it was just erased
and rewritten; stale cached 0xFF bytes can make the header validator reject
magic/version/CRC even though flash programming succeeded.

This build adds nai_flash_read_fresh():
- disable IRQ briefly
- L1C_Cache_Flush_Ext()
- read the tiny metadata block through the existing XIP mapping
- flush again
- re-enable IRQ

Only slot-header reads use this fresh path. Large payload CRC verification
continues using the already-proven fast reader.


Mode 2 v0.3 — FLOAT32 continuous Noodle inference
-------------------------------------------------
This version turns the existing INFERENCE MODE button into a real device-side
prediction path while preserving the proven v0.2.4 BLE + single-slot storage.

First supported inference representation:
  NAI4 REP_ACCEL_GYRO (ax, ay, az, gx, gy, gz), 50 Hz.

The model defines its own rolling-window geometry. For the current deployed
example cfg.bin this is expected to be 100 samples x 6 = 600 inputs. Nothing is
hardcoded to 75 samples anymore.

Runtime:
  continuous QMI8658 -> rolling ring -> every 5 new samples -> oldest-to-newest
  flat input -> StandardScaler -> file-backed Noodle FCN -> P:<class>:<conf>

Weights remain in the single flash slot. Only scaler vectors, Noodle activation
buffers, and two small fixed window arrays use SRAM.

Status messages added:
  MODE:T
  MODE:I
  NO_MODEL
  MODEL:READY:<window_samples>:<input_dim>:<class_count>
  P:<class_index>:<confidence>
  ERR:MODEL_LOAD
  ERR:PRED

Mode 2 v0.4.1 removes the first-proof accel+gyro restriction. The firmware
always records raw six-axis QMI8658 windows and mirrors the web trainer's NAI4
motion-representation preprocessing before Noodle inference. Supported
representations are accel, gyro, accel+gyro, relative quaternion, estimated
velocity, and velocity+quaternion.
