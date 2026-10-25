# Packed-INT8 update for M0Sense Mode 2

This version uses Noodle `FCNFile::metadata_fn` to read `pNN.bin` records (bias/multiplier/shift, three signed int32 little-endian values per neuron) with `wNN.bin` weights. It keeps a fallback for previously stored `w/b/m/s` packages. Noodle's current packed FCN API is backported to the existing BL702-specific vendored Noodle source; the entire upstream Noodle tree has not been replaced.

The original app entry point is `main.c`. The original continuous IMU inference, filtering, confidence threshold, single-slot CRC-verified storage, and build settings are preserved.

A new harmless control command `0x25` replies `CAPS:PACKED_INT8:1`. The updated web sends this before any erase. See the combined package's `README_PACKED_INT8.md` for build, flash-size, and deployment instructions.
