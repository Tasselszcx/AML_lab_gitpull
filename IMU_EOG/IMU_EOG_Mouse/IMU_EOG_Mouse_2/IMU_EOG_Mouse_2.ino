/*
 * [Final Sync] EOG AI Mouse - 适配 120点长窗口与分通道阈值
 */

#include <math.h>
#include <Wire.h>
#include "USB.h"
#include "USBHIDMouse.h"
#include "EOG_AI_Engine_esp32_multi_fur.h" // 确保这里已经更新了新的 scaler 和模型

// ==== 1. 硬件与引脚 ====
USBHIDMouse Mouse;
const int PIN_EOG_HORZ = A0; 
const int PIN_EOG_VERT = A1; 
const uint8_t IMU_ADDR = 0x69; 

// ==== 2. 核心参数同步 (必须与 Python 训练一致) ====
const float ALPHA = 0.2; 
const int WINDOW_SIZE = 70;      // 🌟 70
const int STEP_SIZE = 10;        // 滑动步长，保持 10 即可（每 200ms 推理一次）
const String CLASSES[] = {"Rest", "Up", "Down", "Left", "Right", "Blink"};

// 🌟 物理阈值同步 (根据你的静态检测观察修改)
const float H_THRES_P2P = 50.0;   // 水平通道 P2P 触发门槛
const float V_THRES_P2P = 50.0;   // 垂直通道 P2P 触发门槛

const float SENSITIVITY = 0.02; 
const int DEADZONE = 250; 

// ==== 3. 算法变量 ====
const int ACTION_LOCKOUT_FRAMES = 50; // 🌟 动作后锁定时间增加到 1秒 (50*20ms)
int lock_counter = 0; 

float buffer_h[WINDOW_SIZE], buffer_v[WINDOW_SIZE];
float filtered_ema_h = 512.0, filtered_ema_v = 512.0; 
int frame_count = 0;
float raw_features[14], scaled_features[14];

// 特征提取函数 (确保与 Python 功能一致)
void calculate_axis_features(float* buffer, int offset) {
    float sum = 0;
    for(int i=0; i<WINDOW_SIZE; i++) sum += buffer[i];
    float mean_val = sum / WINDOW_SIZE;
    
    float sum_sq = 0;
    float max_val = -9999.0, min_val = 9999.0;
    float diff_abs_sum = 0, diff_abs_max = 0;

    for(int i=0; i<WINDOW_SIZE; i++) {
        float val = buffer[i] - mean_val; // 🌟 局部去直流
        sum_sq += val * val;
        if(val > max_val) max_val = val;
        if(val < min_val) min_val = val;
        if(i > 0) {
            float d = abs(buffer[i] - buffer[i-1]);
            diff_abs_sum += d;
            if(d > diff_abs_max) diff_abs_max = d;
        }
    }
    raw_features[offset + 0] = sqrt(sum_sq / WINDOW_SIZE); // Std
    raw_features[offset + 1] = max_val - min_val;          // P2P
    raw_features[offset + 2] = 0.0;                        // Mean 强制为 0
    raw_features[offset + 3] = max_val;                    // Max
    raw_features[offset + 4] = min_val;                    // Min
    raw_features[offset + 5] = diff_abs_sum / (WINDOW_SIZE - 1); // MeanDiff
    raw_features[offset + 6] = diff_abs_max;               // MaxDiff
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(10); 
    Wire.begin();
    Mouse.begin();
    USB.begin();

    // 初始化缓冲区为中值
    for(int i=0; i<WINDOW_SIZE; i++) {
        buffer_h[i] = 512.0; buffer_v[i] = 512.0;
    }
    Serial.println("🚀 EOG Mouse: 120pt Model Ready!");
}

void loop() {
    // --- Step A: IMU 移动 (逻辑不变) ---
    // ... (IMU 代码同前)

    // --- Step B: EOG 采集 ---
    int rH = analogRead(PIN_EOG_HORZ);
    int rV = analogRead(PIN_EOG_VERT);
    filtered_ema_h = (ALPHA * rH) + ((1.0 - ALPHA) * filtered_ema_h);
    filtered_ema_v = (ALPHA * rV) + ((1.0 - ALPHA) * filtered_ema_v);

    // 🌟 缓冲区滑动
    for(int i = 0; i < WINDOW_SIZE - 1; i++) {
        buffer_h[i] = buffer_h[i+1]; buffer_v[i] = buffer_v[i+1];
    }
    buffer_h[WINDOW_SIZE - 1] = filtered_ema_h;
    buffer_v[WINDOW_SIZE - 1] = filtered_ema_v;
    frame_count++;

    String current_cmd = "Rest";

    // --- Step C: AI 推理与分通道触发 (完整增强版) ---
    if (lock_counter > 0) {
        lock_counter--;
    } 
    else if (frame_count >= WINDOW_SIZE && frame_count % STEP_SIZE == 0) {
        // 1. 特征提取
        calculate_axis_features(buffer_h, 0); // H轴特征存储在 raw_features[0-6]
        calculate_axis_features(buffer_v, 7); // V轴特征存储在 raw_features[7-13]
        
        float h_p2p = raw_features[1];  // 水平峰峰值
        float v_p2p = raw_features[8];  // 垂直峰峰值 (7+1)

        // 🌟 唤醒逻辑：分通道阈值触发
        if (h_p2p > H_THRES_P2P || v_p2p > V_THRES_P2P) {
            
            // 执行标准化
            scale_features(raw_features, scaled_features);
            
            // 调用分类器
            Eloquent::ML::Port::RandomForest classifier;
            int pred_idx = classifier.predict(scaled_features);
            
            int final_idx = pred_idx;

            // --- 提取核心物理量用于极性二次校验 ---
            float h_max = raw_features[3];  // H轴 Max
            float h_min = raw_features[4];  // H轴 Min
            float v_max = raw_features[10]; // V轴 Max (7+3)
            float v_min = raw_features[11]; // V轴 Min (7+4)
            float v_vel = raw_features[13]; // V轴最大速度 (7+6)

            // --- 🌟 物理极性验证补丁 (防止 AI 误判) ---
            
            // 1. 水平校验 (Left/Right)
            if (pred_idx == 3 && h_min > -40) final_idx = 0; // Left: 必须有明显的负向俯冲
            if (pred_idx == 4 && h_max < 40)  final_idx = 0; // Right: 必须有明显的正向跳变

            // 2. 垂直校验 (Up/Down) - 🌟 新增逻辑
            if (pred_idx == 1 && v_max < 30)  final_idx = 0; // Up: 必须有明显的正向波峰
            if (pred_idx == 2 && v_min > -30) final_idx = 0; // Down: 必须有明显的负向波谷

            // 3. 眨眼校验 (Blink) - 防止轻微向上看被判为眨眼
            if (pred_idx == 5 && v_vel < 40)  final_idx = 0; // Blink 必须具有极高的瞬时速度

            // --- 最终动作执行映射 ---
            if (final_idx != 0) {
                if (final_idx == 3) {        // 向左看
                    Mouse.click(MOUSE_LEFT); 
                } 
                else if (final_idx == 4) {   // 向右看
                    Mouse.click(MOUSE_RIGHT);
                }
                else if (final_idx == 1) {   // 向上看
                    Mouse.move(0, 0, 3);    // 滚轮向上滚动
                }
                else if (final_idx == 2) {   // 向下看
                    Mouse.move(0, 0, -3);   // 滚轮向下滚动
                }
                else if (final_idx == 5) {   // 眨眼
                    // 目前设为空，可根据需要映射为双击
                }

                current_cmd = CLASSES[final_idx];
                // 🌟 动作后锁定期。120点窗口下建议锁定时间稍长，防止同一个动作被识别两次
                lock_counter = ACTION_LOCKOUT_FRAMES; 
            }
        }
    }

    // --- Step D: 串口输出调试 ---
    Serial.print("H:"); Serial.print(filtered_ema_h);
    Serial.print("\tV:"); Serial.print(filtered_ema_v);
    Serial.print("\tCMD:"); Serial.println(current_cmd);

    delay(20); // 确保 50Hz 采样
}