# nai-2
NoodleAI Mode 2 - Continuous Gesture Mode

https://student-embedded-control-and-ai-fest.github.io/nai-2/

---

<img src="https://github.com/Student-Embedded-Control-and-AI-Fest/nai-2/blob/main/works/docs/device.png" width="300">

---


This README file assumes that the working folder is in `~/works/M0sense_BL702_example/`.

0. Install `bflb-mcu-tool-uart`

```
python -m pip install bflb-mcu-tool-uart
```

1. Install the SiFive toolchain

```
cd ~/works/M0sense_BL702_example
git clone https://github.com/wonderfullook/toolchain_gcc_sifive_linux.git
```

If that repository is unavailable, the Bouffalo SDK also documents this alternative source: 

```
git clone https://gitee.com/bouffalolab/toolchain_gcc_sifive_linux.git
```

Use the alternative only if the first clone fails.

2. Set the compiler path

```
export PATH="$PWD/toolchain_gcc_sifive_linux/bin:$PATH"

which riscv64-unknown-elf-gcc
riscv64-unknown-elf-gcc --version
riscv64-unknown-elf-g++ --version
```

We expect GCC 10.2.0 from the original SiFive distribution.

3. Build the firmware

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

4. Flash the firmware

```bash
cd ~/works/M0sense_BL702_example
bflb-mcu-tool-uart \
  --chipname=bl702 \
  --port=/dev/ttyACM0 \
  --baudrate=2000000 \
  --firmware=bl_mcu_sdk/out/m0sense_apps/nai_m0sense/nai_m0sense_bl702.bin
```

5. BL702's flash layut

```
BL702 Flash (512 KiB)
┌─────────────────────────────┐ 0x00000
│ Boot / firmware area        │
│                             │
│ NoodleAI application        │
│                             │
├─────────────────────────────┤ 0x50000
│ Model directory             │
├─────────────────────────────┤ 0x51000
│ Model parameters            │
│ w00.bin, p00.bin, ...       │
│ cfg.bin, mean.bin, ...      │
└─────────────────────────────┘ 0x80000
```