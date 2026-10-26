set(BSP_COMMON_DIR ${CMAKE_SOURCE_DIR}/bsp/bsp_common)

# Research-only serial timing instrumentation. Set to 0 for normal deployment.
# Does not change BLE messages, the rolling-window stride or model arithmetic.
if(NOT DEFINED NAI_BENCH_FCN)
    set(NAI_BENCH_FCN 1)
endif()
add_compile_definitions(NAI_BENCH_FCN=${NAI_BENCH_FCN})

# Mode 2 v0.4.5 = INT8 Noodle MLP + cfg-defined filtering + startup gyro bias calibration.
add_compile_definitions(
    NOODLE_USE_INT8
    NOODLE_USE_BLFLASH
    NOODLE_FILE_FORMAT=1
    NOODLE_FCN_BLOCK=64
)

set(TARGET_REQUIRED_PRIVATE_INCLUDE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/noodle
)

set(TARGET_REQUIRED_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/nai_imu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nai_imu_bridge.cpp

    ${CMAKE_CURRENT_LIST_DIR}/nai_flash_port.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nai_slotfs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/nai_storage_bridge.cpp

    ${CMAKE_CURRENT_LIST_DIR}/nai_infer_bridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/NoodleAIModelBL702.cpp

    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_buffer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_conv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_dw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_fcn.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_globals.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_int8_api.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_int8_layers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_int8_math.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_internal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_io.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_math.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_shape.cpp
    ${CMAKE_CURRENT_LIST_DIR}/noodle/noodle_tensor.cpp

    ${CMAKE_SOURCE_DIR}/examples/ble/bl_irq.c
)

set(TARGET_REQUIRED_LIBS
    freertos
    ble
    mbedtls
    shell
    usb_stack
    usb_stdio
)

set(mains main.c)

set(LINKER_SCRIPT
    ${CMAKE_SOURCE_DIR}/examples/ble/bl702_flash_ble.ld
)

generate_bin()
