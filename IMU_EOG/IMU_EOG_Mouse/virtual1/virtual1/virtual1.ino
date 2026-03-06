/*
 * [MuMu 版] EOG AI + IMU — 纯串口输出
 * 不使用 USB HID Mouse，所有控制由 PC 端 mumu_bridge.py 通过 ADB 完成。
 * 串口输出格式: H:{ema_h}\tV:{ema_v}\tCMD:{action}\tGX:{gx}\tGY:{gy}
 */

#include <math.h>
#include <Wire.h>
#include "EOG_AI_Engine_esp32_multi_15tree.h"

// ==== 1. 硬件与引脚 ====
const int PIN_EOG_HORZ = A0;
const int PIN_EOG_VERT = A1;
const uint8_t IMU_ADDR = 0x69;

// ==== 2. 核心参数 ====
const float ALPHA = 0.2;
const int WINDOW_SIZE = 50;
const int STEP_SIZE = 10;
const String CLASSES[] = {"Rest", "Up", "Down", "Left", "Right", "Blink"};

const int DEADZONE = 250;

// ==== 3. 算法变量 ====
const int ACTION_LOCKOUT_FRAMES = 40;
int lock_counter = 0;

float buffer_h[WINDOW_SIZE], buffer_v[WINDOW_SIZE];
float filtered_ema_h = 512.0, filtered_ema_v = 512.0;
int frame_count = 0;
float raw_features[14], scaled_features[14];

void calculate_axis_features(float* buffer, int offset) {
    float sum = 0;
    for(int i=0; i<WINDOW_SIZE; i++) sum += buffer[i];
    float mean_val = sum / WINDOW_SIZE;
    float sig[WINDOW_SIZE];
    float sum_sq = 0;
    float max_val = -9999.0, min_val = 9999.0;
    float diff_abs_sum = 0, diff_abs_max = 0;

    for(int i=0; i<WINDOW_SIZE; i++) {
        sig[i] = buffer[i] - mean_val;
        sum_sq += sig[i] * sig[i];
        if(sig[i] > max_val) max_val = sig[i];
        if(sig[i] < min_val) min_val = sig[i];
        if(i > 0) {
            float d = abs(buffer[i] - buffer[i-1]);
            diff_abs_sum += d;
            if(d > diff_abs_max) diff_abs_max = d;
        }
    }
    raw_features[offset + 0] = sqrt(sum_sq / WINDOW_SIZE);
    raw_features[offset + 1] = max_val - min_val;
    raw_features[offset + 3] = max_val;
    raw_features[offset + 4] = min_val;
    raw_features[offset + 6] = diff_abs_max;
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(10);
    Wire.begin();

    // IMU 初始化
    Wire.beginTransmission(IMU_ADDR);
    Wire.write(0x6B); Wire.write(0x00);
    Wire.endTransmission(true);

    for(int i=0; i<WINDOW_SIZE; i++) {
        buffer_h[i] = 512.0; buffer_v[i] = 512.0;
    }
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.println("EOG+IMU MuMu Serial Mode Ready!");
}

void loop() {
    // --- Step A: IMU 陀螺仪读取（不控制鼠标，只输出原始值） ---
    int16_t gx = 0, gy = 0;
    Wire.beginTransmission(IMU_ADDR);
    Wire.write(0x43);
    Wire.endTransmission(false);
    Wire.requestFrom((uint16_t)IMU_ADDR, (uint8_t)6, true);
    if (Wire.available() >= 6) {
        gx = Wire.read() << 8 | Wire.read();
        gy = Wire.read() << 8 | Wire.read();
        if (abs(gx) < DEADZONE) gx = 0;
        if (abs(gy) < DEADZONE) gy = 0;
    }

    // --- Step B: EOG 采集 ---
    int rH = analogRead(PIN_EOG_HORZ);
    int rV = analogRead(PIN_EOG_VERT);
    filtered_ema_h = (ALPHA * rH) + ((1.0 - ALPHA) * filtered_ema_h);
    filtered_ema_v = (ALPHA * rV) + ((1.0 - ALPHA) * filtered_ema_v);

    for(int i = 0; i < WINDOW_SIZE - 1; i++) {
        buffer_h[i] = buffer_h[i+1]; buffer_v[i] = buffer_v[i+1];
    }
    buffer_h[WINDOW_SIZE - 1] = filtered_ema_h;
    buffer_v[WINDOW_SIZE - 1] = filtered_ema_v;
    frame_count++;

    String current_cmd = "Rest";

    // --- Step C: AI 推理 ---
    if (lock_counter > 0) {
        lock_counter--;
    }
    else if (frame_count >= WINDOW_SIZE && frame_count % STEP_SIZE == 0) {
        calculate_axis_features(buffer_h, 0);
        calculate_axis_features(buffer_v, 7);

        if (raw_features[1] > 60.0 || raw_features[8] > 60.0) {
            scale_features(raw_features, scaled_features);
            Eloquent::ML::Port::RandomForest classifier;
            int pred_idx = classifier.predict(scaled_features);

            float h_max = raw_features[3], h_min = raw_features[4];
            float v_vel = raw_features[13];
            int final_idx = pred_idx;

            if (pred_idx != 0) {
                // 极性校验
                if (pred_idx == 3 || pred_idx == 4) {
                    if (abs(h_min) > abs(h_max) * 1.2) final_idx = 3; // Left
                    else if (abs(h_max) > abs(h_min) * 1.2) final_idx = 4; // Right
                }
                if (final_idx == 5 && v_vel > 42.0) final_idx = 1;
                else if (final_idx == 1 && v_vel < 35.0) final_idx = 5;

                // 不执行任何 Mouse 操作，只更新命令字符串
                current_cmd = CLASSES[final_idx];
                if (final_idx != 0) lock_counter = ACTION_LOCKOUT_FRAMES;
            }
        }
    }

    // --- Step D: 串口输出（供 mumu_bridge.py 解析） ---
    Serial.print("H:"); Serial.print(filtered_ema_h);
    Serial.print("\tV:"); Serial.print(filtered_ema_v);
    Serial.print("\tCMD:"); Serial.print(current_cmd);
    Serial.print("\tGX:"); Serial.print(gx);
    Serial.print("\tGY:"); Serial.println(gy);

    delay(20);
}
