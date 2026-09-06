#if defined(M0SENSE_USE_USBSTDIO)

#include <hal_usb.h>
/* usb stack core */
#include <usbd_core.h>
/* usb cdc */
#include <usbd_cdc.h>
/* shell */
#include <shell.h>

#define CDC_IN_EP  0x82
#define CDC_OUT_EP 0x01
#define CDC_INT_EP 0x83

#define USBD_VID           0xFFFF
#define USBD_PID           0xFFFF
#define USBD_MAX_POWER     100
#define USBD_LANGID_STRING 1033

#define USB_CONFIG_SIZE (9 + CDC_ACM_DESCRIPTOR_LEN)

USB_DESC_SECTION static const uint8_t cdc_descriptor[] = {
    USB_DEVICE_DESCRIPTOR_INIT(USB_2_0, 0x02, 0x02, 0x01, USBD_VID, USBD_PID, 0x0100, 0x01),
    USB_CONFIG_DESCRIPTOR_INIT(USB_CONFIG_SIZE, 0x02, 0x01, USB_CONFIG_BUS_POWERED, USBD_MAX_POWER),
    CDC_ACM_DESCRIPTOR_INIT(0x00, CDC_INT_EP, CDC_OUT_EP, CDC_IN_EP, 0x02),
    ///////////////////////////////////////
    /// string0 descriptor
    ///////////////////////////////////////
    USB_LANGID_INIT(USBD_LANGID_STRING),
    ///////////////////////////////////////
    /// string1 descriptor
    ///////////////////////////////////////
    0x12,                       /* bLength */
    USB_DESCRIPTOR_TYPE_STRING, /* bDescriptorType */
    'B', 0x00,                  /* wcChar0 */
    'o', 0x00,                  /* wcChar1 */
    'u', 0x00,                  /* wcChar2 */
    'f', 0x00,                  /* wcChar3 */
    'f', 0x00,                  /* wcChar4 */
    'a', 0x00,                  /* wcChar5 */
    'l', 0x00,                  /* wcChar6 */
    'o', 0x00,                  /* wcChar7 */
    ///////////////////////////////////////
    /// string2 descriptor
    ///////////////////////////////////////
    0x24,                       /* bLength */
    USB_DESCRIPTOR_TYPE_STRING, /* bDescriptorType */
    'B', 0x00,                  /* wcChar0 */
    'o', 0x00,                  /* wcChar1 */
    'u', 0x00,                  /* wcChar2 */
    'f', 0x00,                  /* wcChar3 */
    'f', 0x00,                  /* wcChar4 */
    'a', 0x00,                  /* wcChar5 */
    'l', 0x00,                  /* wcChar6 */
    'o', 0x00,                  /* wcChar7 */
    ' ', 0x00,                  /* wcChar8 */
    'C', 0x00,                  /* wcChar9 */
    'D', 0x00,                  /* wcChar10 */
    'C', 0x00,                  /* wcChar11 */
    ' ', 0x00,                  /* wcChar13 */
    'D', 0x00,                  /* wcChar14 */
    'E', 0x00,                  /* wcChar15 */
    'M', 0x00,                  /* wcChar16 */
    'O', 0x00,                  /* wcChar17 */
    ///////////////////////////////////////
    /// string3 descriptor
    ///////////////////////////////////////
    0x16,                       /* bLength */
    USB_DESCRIPTOR_TYPE_STRING, /* bDescriptorType */
    '2', 0x00,                  /* wcChar0 */
    '0', 0x00,                  /* wcChar1 */
    '2', 0x00,                  /* wcChar2 */
    '1', 0x00,                  /* wcChar3 */
    '0', 0x00,                  /* wcChar4 */
    '3', 0x00,                  /* wcChar5 */
    '1', 0x00,                  /* wcChar6 */
    '0', 0x00,                  /* wcChar7 */
    '0', 0x00,                  /* wcChar8 */
    '0', 0x00,                  /* wcChar9 */
#ifdef CONFIG_USB_HS
    ///////////////////////////////////////
    /// device qualifier descriptor
    ///////////////////////////////////////
    0x0a, USB_DESCRIPTOR_TYPE_DEVICE_QUALIFIER, 0x00, 0x02, 0x02, 0x02, 0x01, 0x40, 0x01, 0x00,
#endif
    0x00
};

static void usbd_cdc_acm_bulk_out(uint8_t ep)
{
    uint32_t actual_read_length = 0;
    uint8_t out_buffer[64];

    if (usbd_ep_read(ep, out_buffer, 64, &actual_read_length) < 0) {
        MSG("Read DATA Packet failed\r\n");
        usbd_ep_set_stall(ep);
        return;
    }
    for (uint32_t i = 0; i < actual_read_length; i++) {
        shell_handler(out_buffer[i]);
    }
    usbd_ep_read(ep, NULL, 0, NULL);
}

static Ring_Buffer_Type usb_tx_rb;
static void usbd_cdc_acm_bulk_in(uint8_t ep)
{
    usb_dc_send_from_ringbuffer(device_find("usb"), &usb_tx_rb, ep);
}

static usbd_class_t cdc_class;
static usbd_interface_t cdc_cmd_intf;
static usbd_interface_t cdc_data_intf;

static usbd_endpoint_t cdc_out_ep = { .ep_addr = CDC_OUT_EP, .ep_cb = usbd_cdc_acm_bulk_out };

static usbd_endpoint_t cdc_in_ep = { .ep_addr = CDC_IN_EP, .ep_cb = usbd_cdc_acm_bulk_in };

void usb_stdio_init()
{
    shell_init();
    shell_set_print(printf);

#define USB_TX_RINGBUFFER_SIZE (8 * 1024)
    static uint8_t usb_tx_mem[USB_TX_RINGBUFFER_SIZE] __attribute__((section(".system_ram")));
    Ring_Buffer_Init(&usb_tx_rb, usb_tx_mem, USB_TX_RINGBUFFER_SIZE, cpu_global_irq_disable, cpu_global_irq_enable);

    usbd_desc_register(cdc_descriptor);

    usbd_cdc_add_acm_interface(&cdc_class, &cdc_cmd_intf);
    usbd_cdc_add_acm_interface(&cdc_class, &cdc_data_intf);
    usbd_interface_add_endpoint(&cdc_data_intf, &cdc_out_ep);
    usbd_interface_add_endpoint(&cdc_data_intf, &cdc_in_ep);

    extern struct device *usb_dc_init(void);
    struct device *usb_fs = usb_dc_init();

    if (usb_fs) {
        device_control(usb_fs, DEVICE_CTRL_SET_INT, (void *)(USB_EP1_DATA_OUT_IT | USB_EP2_DATA_IN_IT));
    }

    while (!usb_device_is_configured()) {
    }
}

Ring_Buffer_Type *usb_stdio_get_tx_rb()
{
    return &usb_tx_rb;
}

#endif /* M0SENSE_USE_USBSTDIO */