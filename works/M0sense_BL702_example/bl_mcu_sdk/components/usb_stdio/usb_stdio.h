#pragma once

#if defined(M0SENSE_USE_USBSTDIO)

#include <ring_buffer.h>

extern void usb_stdio_init();
extern Ring_Buffer_Type *usb_stdio_get_tx_rb();

#endif /* M0SENSE_USE_USBSTDIO */