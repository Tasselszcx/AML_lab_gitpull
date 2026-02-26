/*
 * [Phase 7 - Polarity Priority] EOG AI Engine
 * 修复重点：1. 强制极性校验区分左右 2. 抑制水平对垂直的串扰
 */

#include <math.h>
#include "EOG_AI_Engine_esp32.h" 

const int PIN_EOG_HORZ = A0; 
const int PIN_EOG_VERT = A1; 
const float ALPHA = 0.2; 
const int WINDOW_SIZE = 50; 
const int STEP_SIZE = 10;
const String CLASSES[] = {"Rest", "Up", "Down", "Left", "Right", "Blink"};

// 稳定性参数
const int ACTION_LOCKOUT_FRAMES = 40; 
float buffer_h[WINDOW_SIZE], buffer_v[WINDOW_SIZE];
float filtered_ema_h = 512.0, filtered_ema_v = 512.0; 
int frame_count = 0;
int lock_counter = 0; 
float raw_features[14];
float scaled_features[14];

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
    raw_features[offset + 0] = sqrt(sum_sq / WINDOW_SIZE); // Std
    raw_features[offset + 1] = max_val - min_val;          // P2P
    raw_features[offset + 2] = 0;                          
    raw_features[offset + 3] = max_val;                    // 🌟 正向峰值
    raw_features[offset + 4] = min_val;                    // 🌟 负向谷值
    raw_features[offset + 5] = diff_abs_sum / (WINDOW_SIZE - 1); 
    raw_features[offset + 6] = diff_abs_max;               
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(10); 
    for(int i=0; i<WINDOW_SIZE; i++) {
        buffer_h[i] = 512.0; buffer_v[i] = 512.0;
    }
}

void loop() {
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

    if (lock_counter > 0) {
        lock_counter--;
        current_cmd = "Rest";
    } 
    else if (frame_count >= WINDOW_SIZE && frame_count % STEP_SIZE == 0) {
        calculate_axis_features(buffer_h, 0);
        calculate_axis_features(buffer_v, 7);
        
        // 门槛：只有信号强度足够大时才推理
        if (raw_features[1] > 60.0 || raw_features[8] > 60.0) {
            scale_features(raw_features, scaled_features);
            Eloquent::ML::Port::RandomForest classifier;
            int pred_idx = classifier.predict(scaled_features);
            
            // 🌟 物理纠错：极性优先逻辑 🌟
            float h_max = raw_features[3];
            float h_min = raw_features[4];
            float v_max = raw_features[10];
            float v_min = raw_features[11];
            float v_vel = raw_features[13];

            int final_idx = pred_idx;

            if (pred_idx != 0) {
                // 1. 左右区分：看 H 通道哪边绝对值大
                if (pred_idx == 3 || pred_idx == 4) { // 模型说是 Left 或 Right
                    if (abs(h_min) > abs(h_max) * 1.2) final_idx = 3; // 负向强 -> Left
                    else if (abs(h_max) > abs(h_min) * 1.2) final_idx = 4; // 正向强 -> Right
                }
                
                // 2. 抑制水平对垂直的串扰
                // 如果 H 通道的强度明显大于 V 通道，那么忽略 V 通道的 Up/Down 判定
                if ((pred_idx == 1 || pred_idx == 2) && (raw_features[1] > raw_features[8] * 1.5)) {
                    final_idx = 0; // 判定为误触，其实是左右看带出来的
                }

                // 3. 原有的 Blink/Up 修正
                if (final_idx == 5 && v_vel > 42.0) final_idx = 1; 
                else if (final_idx == 1 && v_vel < 35.0) final_idx = 5;

                current_cmd = CLASSES[final_idx];
                if (final_idx != 0) lock_counter = ACTION_LOCKOUT_FRAMES;
            }
        }
    }

    Serial.print("H:"); Serial.print(filtered_ema_h);
    Serial.print("\tV:"); Serial.print(filtered_ema_v);
    Serial.print("\tCMD:"); Serial.println(current_cmd);

    delay(20); 
}