# nai-2
NoodleAI Mode 2 - Continuous Gesture Mode



```
cd ~/works/M0sense_BL702_example
rm -rf bl_mcu_sdk/build
./build.sh m0sense_apps/nai_m0sense
```


Then immediately check:

```bash
stat -c '%s bytes' \
  bl_mcu_sdk/out/m0sense_apps/nai_m0sense/nai_m0sense_bl702.bin
```

The limit is:

```bash
319488 bytes
```

Then flash:

```bash
cd ~/works/M0sense_BL702_example
bflb-mcu-tool-uart \
  --chipname=bl702 \
  --port=/dev/ttyACM0 \
  --baudrate=2000000 \
  --firmware=bl_mcu_sdk/out/m0sense_apps/nai_m0sense/nai_m0sense_bl702.bin
```