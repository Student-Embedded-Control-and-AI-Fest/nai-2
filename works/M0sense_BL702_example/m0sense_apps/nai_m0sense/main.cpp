extern "C" {
#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>

#include "bflb_platform.h"
#include "bl702_glb.h"
#include "bl702_sec_eng.h"
#include "hal_gpio.h"
}

#include <math.h>
#include <new>
#include <stdio.h>
#include <string.h>

#include "io_def.h"
#include "nai_protocol.h"
#include "nai_ble.h"
#include "nai_imu.h"
#include "nai_slotfs.h"
#include "nai_motion.h"
#include "NoodleAIModelBL702.h"

/* ----------------------------- Runtime ----------------------------- */

static NoodleAIModelBL702 model;
static NaiDeviceMode mode=NaiDeviceMode::TRAINING;
static QueueHandle_t app_queue=nullptr;

static float *gesture_raw=nullptr;
static float *derived_raw=nullptr;
static float *inference_window=nullptr;

static bool gesture_active=false;
static bool gesture_truncated=false;
static uint16_t gesture_count=0;
static uint32_t gesture_start_ms=0;

static bool boot_raw=false;
static bool boot_stable=false;
static uint32_t boot_changed_ms=0;

static NaiImuPacket imu_packet{};
static uint8_t imu_batch=0;
static bool force_training=false;

static StackType_t ble_stack_mem[1024];
static StaticTask_t ble_task_tcb;
static StackType_t app_stack_mem[1536];
static StaticTask_t app_task_tcb;
static StackType_t sensor_stack_mem[1024];
static StaticTask_t sensor_task_tcb;
static StackType_t heartbeat_stack_mem[256];
static StaticTask_t heartbeat_task_tcb;

/* ----------------------------- Time ----------------------------- */

static uint32_t now_ms()
{
    return (uint32_t)(bflb_platform_get_time_us()/1000ULL);
}

/* ----------------------------- LED ----------------------------- */

static void fatal_red()
{
    gpio_write(LED_B_PIN,1);
    gpio_write(LED_G_PIN,1);
    gpio_write(LED_R_PIN,0);
}

static void heartbeat_task(void *)
{
    bool on=false;
    for (;;) {
        on=!on;
        gpio_write(LED_B_PIN,on?0:1);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

/* ----------------------------- Buffers / mode ----------------------------- */

static bool needs_derived()
{
    if (!model.loaded()) return false;
    auto r=model.representation();
    return r==NoodleAIModelBL702::REP_QUATERNION ||
           r==NoodleAIModelBL702::REP_VELOCITY ||
           r==NoodleAIModelBL702::REP_VELOCITY_QUATERNION;
}

static bool buffers_ready()
{
    return gesture_raw && inference_window && (!needs_derived() || derived_raw);
}

static void reset_gesture()
{
    gesture_active=false;
    gesture_truncated=false;
    gesture_count=0;
    gesture_start_ms=0;
    imu_batch=0;
    imu_packet.count=0;
}

static void free_inference_buffers()
{
    delete[] gesture_raw;
    delete[] derived_raw;
    delete[] inference_window;
    gesture_raw=nullptr;
    derived_raw=nullptr;
    inference_window=nullptr;
    reset_gesture();
}

static bool rebuild_inference_buffers()
{
    free_inference_buffers();
    if (!model.loaded()) return true;

    const uint16_t ch=model.channelCount();
    if (!ch || ch>7 || !model.inputDim() ||
        (uint32_t)model.windowLength()*ch != model.inputDim())
        return false;

    gesture_raw=new(std::nothrow) float[NAI_MAX_GESTURE_SAMPLES*6u];
    inference_window=new(std::nothrow) float[model.inputDim()];
    if (needs_derived())
        derived_raw=new(std::nothrow) float[NAI_MAX_GESTURE_SAMPLES*ch];

    return buffers_ready();
}

static void set_mode(NaiDeviceMode m)
{
    if (m==NaiDeviceMode::INFERENCE &&
        (!model.loaded() || !buffers_ready())) {
        mode=NaiDeviceMode::TRAINING;
        reset_gesture();
        nai_ble_status("NO_MODEL");
        return;
    }

    mode=m;
    reset_gesture();
    nai_ble_status(mode==NaiDeviceMode::TRAINING ? "MODE:T" : "MODE:I");
}

/* ----------------------------- BLE live IMU ----------------------------- */

static void flush_imu_packet()
{
    if (!imu_batch) return;
    if (!nai_slotfs_deploy_active() &&
        nai_ble_connected() &&
        nai_ble_imu_notify_enabled()) {
        imu_packet.count=imu_batch;
        nai_ble_notify_imu(&imu_packet);
    }
    imu_batch=0;
    imu_packet.count=0;
}

static void stream_sample(const float s[6],uint32_t now)
{
    if (nai_slotfs_deploy_active() ||
        !nai_ble_connected() ||
        !nai_ble_imu_notify_enabled()) return;

    if (!imu_batch) imu_packet.t0_ms=now;
    NaiImuSample &d=imu_packet.samples[imu_batch];
    d.ax=s[0];d.ay=s[1];d.az=s[2];
    d.gx=s[3];d.gy=s[4];d.gz=s[5];

    ++imu_batch;
    if (imu_batch>=NAI_BLE_BATCH_SIZE) flush_imu_packet();
}

/* ----------------------------- Gesture ----------------------------- */

static void begin_gesture(uint32_t now)
{
    if (nai_slotfs_deploy_active() || gesture_active) return;

    if (mode==NaiDeviceMode::TRAINING && !nai_ble_connected()) return;
    if (mode==NaiDeviceMode::INFERENCE &&
        (!model.loaded() || !buffers_ready())) {
        nai_ble_status("NO_MODEL");
        return;
    }

    gesture_active=true;
    gesture_truncated=false;
    gesture_count=0;
    gesture_start_ms=now;
    flush_imu_packet();
    nai_ble_status("GESTURE:START");
}

static void run_prediction(uint32_t duration)
{
    if (!model.loaded() || !buffers_ready()) return;

    if (!nai_build_inference_representation(
            model.representation(),gesture_raw,gesture_count,
            model.sampleRateHz(),model.windowLength(),
            inference_window,derived_raw)) {
        nai_ble_status("ERR:REP");
        return;
    }

    uint16_t cls=0;
    float confidence=0;
    if (!model.predict(inference_window,cls,confidence)) {
        nai_ble_status("ERR:PRED");
        return;
    }

    char msg[32];
    snprintf(msg,sizeof(msg),"P:%u:%.3f",(unsigned)cls,confidence);
    nai_ble_status(msg);

    (void)duration;
}

static void end_gesture(uint32_t now)
{
    if (!gesture_active) return;
    gesture_active=false;
    const uint32_t duration=now-gesture_start_ms;

    flush_imu_packet();

    char msg[32];
    if (gesture_count<NAI_MIN_GESTURE_SAMPLES) {
        snprintf(msg,sizeof(msg),"GESTURE:SHORT:%u",(unsigned)gesture_count);
        nai_ble_status(msg);
        return;
    }

    snprintf(msg,sizeof(msg),"GESTURE:END:%u:%lu",
             (unsigned)gesture_count,(unsigned long)duration);
    nai_ble_status(msg);

    if (mode==NaiDeviceMode::INFERENCE)
        run_prediction(duration);
}

static void update_button(uint32_t now)
{
    const bool pressed=(gpio_read(BTN_PIN)==0);

    if (pressed!=boot_raw) {
        boot_raw=pressed;
        boot_changed_ms=now;
    }

    if (pressed!=boot_stable &&
        now-boot_changed_ms>=NAI_RUNTIME_BOOT_DEBOUNCE_MS) {
        boot_stable=pressed;
        if (boot_stable) begin_gesture(now);
        else end_gesture(now);
    }
}

/* ----------------------------- Deploy ----------------------------- */

static const char *slot_result_status(NaiSlotResult r)
{
    switch (r) {
    case NAI_SLOT_ERR_NAME: return "ERR:NAME";
    case NAI_SLOT_ERR_SIZE: return "ERR:SIZE";
    case NAI_SLOT_ERR_COUNT: return "ERR:COUNT";
    case NAI_SLOT_ERR_CRC: return "ERR:CRC";
    case NAI_SLOT_ERR_FLASH: return "ERR:WRITE";
    default: return "ERR:FILE";
    }
}

static void abort_deploy(const char *status)
{
    nai_slotfs_abort(true);
    nai_ble_status(status);
}

static void begin_deploy()
{
    if (nai_slotfs_deploy_active()) nai_slotfs_abort(true);

    NaiSlotResult r=nai_slotfs_deploy_begin();
    if (r!=NAI_SLOT_OK) {
        nai_ble_status(slot_result_status(r));
        return;
    }

    set_mode(NaiDeviceMode::TRAINING);

    char msg[12];
    snprintf(msg,sizeof(msg),"DEPLOY:%c",nai_slotfs_deploy_target());
    nai_ble_status(msg);
}

static void begin_file(const uint8_t *p,size_t n)
{
    if (!nai_slotfs_deploy_active() ||
        nai_slotfs_file_active() || n<10) {
        nai_ble_status("ERR:FILE_CMD"); return;
    }

    const uint8_t nameLen=p[1];
    if (!nameLen || nameLen>=NAI_SLOT_NAME_BYTES ||
        n!=(size_t)(10u+nameLen)) {
        nai_ble_status("ERR:NAME"); return;
    }

    uint32_t size=0,crc=0;
    memcpy(&size,p+2,4);
    memcpy(&crc,p+6,4);

    char name[NAI_SLOT_NAME_BYTES]{};
    memcpy(name,p+10,nameLen);

    NaiSlotResult r=nai_slotfs_file_begin(name,size,crc);
    if (r!=NAI_SLOT_OK) {
        nai_ble_status(slot_result_status(r));
        return;
    }

    char msg[32];
    snprintf(msg,sizeof(msg),"FILE_READY:%s",name);
    nai_ble_status(msg);
}

static void finish_file()
{
    NaiSlotResult r=nai_slotfs_file_end();
    if (r!=NAI_SLOT_OK) {
        nai_ble_status(slot_result_status(r));
        return;
    }

    /*
     * The desktop only needs FILE_OK:<name>.  slotfs intentionally owns the
     * current filename, so a generic FILE_OK is not sufficient.  We recover
     * the name from the preceding FILE_BEGIN in the app event stream by
     * keeping it here.
     */
    nai_ble_status("FILE_OK");
}

static char current_file_name[NAI_SLOT_NAME_BYTES]{};

static void begin_file_tracking(const uint8_t *p,size_t n)
{
    current_file_name[0]='\0';
    if (n>=10) {
        const uint8_t len=p[1];
        if (len && len<sizeof(current_file_name) &&
            n==(size_t)(10u+len)) {
            memcpy(current_file_name,p+10,len);
            current_file_name[len]='\0';
        }
    }
    begin_file(p,n);
}

static void finish_file_tracking()
{
    NaiSlotResult r=nai_slotfs_file_end();
    if (r!=NAI_SLOT_OK) {
        nai_ble_status(slot_result_status(r));
        current_file_name[0]='\0';
        return;
    }

    char msg[32];
    snprintf(msg,sizeof(msg),"FILE_OK:%s",current_file_name);
    current_file_name[0]='\0';
    nai_ble_status(msg);
}

static void commit_deploy()
{
    if (!nai_slotfs_deploy_active() || nai_slotfs_file_active()) {
        nai_ble_status("ERR:COMMIT"); return;
    }

    const char newSlot=nai_slotfs_deploy_target();
    const char oldSlot=nai_slotfs_deploy_old_slot();

    if (nai_slotfs_stage_header()!=NAI_SLOT_OK) {
        abort_deploy("ERR:COMMIT");
        return;
    }

    model.clearRuntime();

    if (!model.loadSlot(newSlot,true)) {
        nai_slotfs_abort(true);
        if (oldSlot=='A'||oldSlot=='B') model.loadSlot(oldSlot,false);
        rebuild_inference_buffers();
        nai_ble_status("ERR:MODEL");
        return;
    }

    if (nai_slotfs_activate_staged()!=NAI_SLOT_OK) {
        model.clearRuntime();
        if (oldSlot=='A'||oldSlot=='B') model.loadSlot(oldSlot,false);
        rebuild_inference_buffers();
        nai_ble_status("ERR:ACTIVE");
        return;
    }

    /* Reload via the committed backend, then rebuild gesture buffers. */
    model.clearRuntime();
    if (!model.loadSlot(newSlot,false) || !rebuild_inference_buffers()) {
        nai_ble_status("ERR:RAM");
        set_mode(NaiDeviceMode::TRAINING);
        return;
    }

    nai_ble_status("MODEL_OK");
    set_mode(NaiDeviceMode::INFERENCE);
}

static void execute_control(const uint8_t *p,size_t n)
{
    if (!p || !n) return;

    switch (p[0]) {
    case OP_SET_TRAINING:
        set_mode(NaiDeviceMode::TRAINING);
        break;

    case OP_SET_INFERENCE:
        set_mode(NaiDeviceMode::INFERENCE);
        break;

    case OP_ERASE_MODEL:
        if (nai_slotfs_deploy_active()) nai_slotfs_abort(true);
        model.clearRuntime();
        free_inference_buffers();
        nai_slotfs_erase_all();
        set_mode(NaiDeviceMode::TRAINING);
        nai_ble_status("MODEL_ERASED");
        break;

    case OP_DEPLOY_BEGIN:
        begin_deploy();
        break;

    case OP_FILE_BEGIN:
        begin_file_tracking(p,n);
        break;

    case OP_FILE_END:
        finish_file_tracking();
        break;

    case OP_DEPLOY_COMMIT:
        commit_deploy();
        break;

    case OP_DEPLOY_ABORT:
        if (nai_slotfs_deploy_active()) abort_deploy("RX:ABORT");
        break;

    default:
        nai_ble_status("ERR:CMD");
        break;
    }
}

/* ----------------------------- App event task ----------------------------- */

static void app_task(void *)
{
    NaiAppEvent e{};

    for (;;) {
        if (xQueueReceive(app_queue,&e,portMAX_DELAY)!=pdTRUE) continue;

        switch (e.type) {
        case NAI_EVT_CONNECTED:
            nai_ble_status(model.loaded() ? "CONNECTED:M" : "CONNECTED:N");
            break;

        case NAI_EVT_DISCONNECTED:
            if (nai_slotfs_deploy_active()) nai_slotfs_abort(true);
            set_mode((model.loaded()&&buffers_ready())
                ? NaiDeviceMode::INFERENCE
                : NaiDeviceMode::TRAINING);
            break;

        case NAI_EVT_CONTROL:
            execute_control(e.data,e.len);
            break;

        case NAI_EVT_MODEL_DATA:
            if (nai_slotfs_deploy_active() && nai_slotfs_file_active()) {
                NaiSlotResult r=nai_slotfs_file_write(e.data,e.len);
                if (r!=NAI_SLOT_OK) abort_deploy(slot_result_status(r));
            }
            break;

        default:
            break;
        }
    }
}

/* ----------------------------- Sensor / button task ----------------------------- */

static void sensor_task(void *)
{
    TickType_t last=xTaskGetTickCount();

    for (;;) {
        const uint32_t now=now_ms();
        update_button(now);

        float s[6];
        if (nai_imu_read(s)) {
            stream_sample(s,now);

            if (gesture_active && !nai_slotfs_deploy_active()) {
                if (gesture_count>=NAI_MAX_GESTURE_SAMPLES) {
                    gesture_truncated=true;
                } else if (mode==NaiDeviceMode::TRAINING) {
                    ++gesture_count;
                } else if (model.loaded() && buffers_ready()) {
                    const uint16_t b=gesture_count*6u;
                    for (uint8_t i=0;i<6;++i) gesture_raw[b+i]=s[i];
                    ++gesture_count;
                }
            }
        }

        vTaskDelayUntil(&last,pdMS_TO_TICKS(NAI_SAMPLE_PERIOD_MS));
    }
}

/* ----------------------------- BLE init task ----------------------------- */

static void ble_task(void *)
{
    vTaskDelay(pdMS_TO_TICKS(500));

    nai_ble_start_stack();
    vTaskDelete(NULL);
}

/* ----------------------------- Sipeed M0Sense tickless idle hook ----------------------------- */

extern uint64_t ullNextTime;
extern const size_t uxTimerIncrementsForOneTick;

extern "C" void vApplicationSleep(uint32_t xExpectedIdleTime)
{
    uint64_t sleep_start_time=0,sleep_end_time=0,record_ullNextTime=0;
    uint32_t actual_compensate_mstime=0;
    static uint32_t actual_compensate_ustime=0;

    if (eTaskConfirmSleepModeStatus()!=eAbortSleep) {
        if (xExpectedIdleTime>60*1000) xExpectedIdleTime=60*1000;

        cpu_global_irq_disable();
        sleep_start_time=bflb_platform_get_time_us();

        ullNextTime-=uxTimerIncrementsForOneTick;
        record_ullNextTime=ullNextTime;
        ullNextTime+=uxTimerIncrementsForOneTick*(xExpectedIdleTime-1);
        *(volatile uint64_t *)(0x02000000+0x4000)=ullNextTime;
        ullNextTime+=uxTimerIncrementsForOneTick;

        __WFI();

        sleep_end_time=bflb_platform_get_time_us();

        if (xExpectedIdleTime==
            pdMS_TO_TICKS((sleep_end_time-sleep_start_time+500)/1000)) {
            vTaskStepTick(xExpectedIdleTime);
        } else {
            ullNextTime=record_ullNextTime;
            ullNextTime+=(sleep_end_time-sleep_start_time);
            *(volatile uint64_t *)(0x02000000+0x4000)=ullNextTime;
            ullNextTime+=uxTimerIncrementsForOneTick;

            actual_compensate_ustime+=(sleep_end_time-sleep_start_time);
            if (actual_compensate_ustime/1000) {
                actual_compensate_mstime=actual_compensate_ustime/1000;
                actual_compensate_ustime-=actual_compensate_mstime*1000;
            }
            vTaskStepTick(pdMS_TO_TICKS(actual_compensate_mstime));
        }
        cpu_global_irq_enable();
    }
}

/* ----------------------------- Required FreeRTOS hooks ----------------------------- */

extern uint8_t _heap_start;
extern uint8_t _heap_size;

static HeapRegion_t heap_regions[]={
    {&_heap_start,(unsigned int)&_heap_size},
    {NULL,0},{NULL,0}
};

extern "C" void user_vAssertCalled(void)
    __attribute__((weak,alias("vAssertCalled")));

extern "C" void vAssertCalled(void)
{
    fatal_red();
    for(;;){}
}

extern "C" void vApplicationTickHook(void)
{
}

extern "C" void vApplicationStackOverflowHook(TaskHandle_t,char *)
{
    fatal_red();
    for(;;){}
}

extern "C" void vApplicationMallocFailedHook(void)
{
    fatal_red();
    for(;;){}
}

extern "C" void vApplicationGetIdleTaskMemory(
    StaticTask_t **t,
    StackType_t **s,
    uint32_t *n)
{
    static StaticTask_t tcb;
    static StackType_t stack[configMINIMAL_STACK_SIZE];

    *t=&tcb;
    *s=stack;
    *n=configMINIMAL_STACK_SIZE;
}

extern "C" void vApplicationGetTimerTaskMemory(
    StaticTask_t **t,
    StackType_t **s,
    uint32_t *n)
{
    static StaticTask_t tcb;
    static StackType_t stack[configTIMER_TASK_STACK_DEPTH];

    *t=&tcb;
    *s=stack;
    *n=configTIMER_TASK_STACK_DEPTH;
}


/* ----------------------------- Startup ----------------------------- */

int main(void)
{
    bflb_platform_init(0);

    gpio_set_mode(LED_B_PIN,GPIO_OUTPUT_PP_MODE);
    gpio_set_mode(LED_G_PIN,GPIO_OUTPUT_PP_MODE);
    gpio_set_mode(LED_R_PIN,GPIO_OUTPUT_PP_MODE);
    gpio_write(LED_B_PIN,1);
    gpio_write(LED_G_PIN,1);
    gpio_write(LED_R_PIN,1);

    /* GPIO_INPUT_PP_MODE = input with pull-up in BL MCU SDK. */
    gpio_set_mode(BTN_PIN,GPIO_INPUT_PP_MODE);

    /*
     * Same crystal setup proven by Bouffalo's BLE peripheral example.
     */
    HBN_Set_XCLK_CLK_Sel(HBN_XCLK_CLK_XTAL);
    uint32_t tmp=BL_RD_REG(AON_BASE,AON_XTAL_CFG);
    tmp=BL_SET_REG_BITS_VAL(tmp,AON_XTAL_CAPCODE_IN_AON,33);
    tmp=BL_SET_REG_BITS_VAL(tmp,AON_XTAL_CAPCODE_OUT_AON,33);
    BL_WR_REG(AON_BASE,AON_XTAL_CFG,tmp);
    Sec_Eng_Trng_Enable();

    vPortDefineHeapRegions(heap_regions);

    if (!nai_slotfs_init()) {
        fatal_red();
        for(;;){}
    }

    if (!nai_imu_init()) {
        fatal_red();
        for(;;){}
    }

    /*
     * Startup BOOT choice.  GPIO2 is the software BOOT key on M0Sense; the
     * separate Boot_Strap pin is the hardware download strap.
     */
    const uint32_t t0=now_ms();
    while (now_ms()-t0<NAI_BOOT_TRAIN_WINDOW_MS) {
        if (gpio_read(BTN_PIN)==0) {
            const uint32_t p0=now_ms();
            while (gpio_read(BTN_PIN)==0 &&
                   now_ms()-p0<60) bflb_platform_delay_ms(1);
            if (now_ms()-p0>=60) {
                force_training=true;
                break;
            }
        }
        bflb_platform_delay_ms(5);
    }

    const bool model_ok=model.loadActive();
    if (model_ok && !rebuild_inference_buffers()) {
        model.clearRuntime();
        free_inference_buffers();
    }

    app_queue=xQueueCreate(10,sizeof(NaiAppEvent));
    if (!app_queue) {
        fatal_red();
        for(;;){}
    }

    nai_ble_bind_event_queue(app_queue);

    boot_raw=(gpio_read(BTN_PIN)==0);
    boot_stable=boot_raw;
    boot_changed_ms=now_ms();

    xTaskCreateStatic(heartbeat_task,"heartbeat",
        sizeof(heartbeat_stack_mem)/sizeof(StackType_t),
        NULL,3,heartbeat_stack_mem,&heartbeat_task_tcb);

    xTaskCreateStatic(sensor_task,"sensor",
        sizeof(sensor_stack_mem)/sizeof(StackType_t),
        NULL,6,sensor_stack_mem,&sensor_task_tcb);

    xTaskCreateStatic(app_task,"nai_app",
        sizeof(app_stack_mem)/sizeof(StackType_t),
        NULL,10,app_stack_mem,&app_task_tcb);

    xTaskCreateStatic(ble_task,"ble_init",
        sizeof(ble_stack_mem)/sizeof(StackType_t),
        NULL,15,ble_stack_mem,&ble_task_tcb);

    mode=(force_training || !model.loaded() || !buffers_ready())
        ? NaiDeviceMode::TRAINING
        : NaiDeviceMode::INFERENCE;

    vTaskStartScheduler();

    for (;;) __WFI();
}
