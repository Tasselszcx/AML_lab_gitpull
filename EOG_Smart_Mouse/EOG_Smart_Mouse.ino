/*
 * [Phase 2] ESP32 EOG 实时边缘计算引擎 (Edge AI)
 * 包含：数据采集 -> EMA滤波 -> 14维特征提取 -> StandardScaler -> 随机森林推理
 */

#include <math.h>
#include "EOG_AI_Engine.h" // 包含你们生成的 Scaler 和 模型

// ==== 1. 引脚与基础参数 ====
const int PIN_EOG_HORZ = A0; 
const int PIN_EOG_VERT = A1; 
const float ALPHA = 0.2; 
float filteredHorz = 512.0; 
float filteredVert = 512.0;

// ==== 2. 类别映射表 (对照你在 Python 里的 CLASSES 顺序) ====
// ⚠️ 注意：这里的顺序必须和你 Python 代码里的 CLASSES 列表一模一样！
const String CLASSES[] = {"Rest", "Up", "Down", "Left", "Right", "Blink"};

// ==== 3. 滑动窗口缓冲 (完美复现 Python 的 deque) ====
const int WINDOW_SIZE = 50;
float buffer_h[WINDOW_SIZE];
float buffer_v[WINDOW_SIZE];
int frame_count = 0; // 用于记录一共收集了多少帧

// 特征数组
float raw_features[14]; 
float scaled_features[14];

// 声明特征提取函数
void extract_features();
void calculate_axis_features(float* sig, int offset);

void setup() {
  Serial.begin(115200);
  
  // 初始化缓冲区为基线值
  for(int i=0; i<WINDOW_SIZE; i++){
    buffer_h[i] = 512.0;
    buffer_v[i] = 512.0;
  }
  
  Serial.println("🚀 ESP32 Edge AI Engine Ready!");
}

void loop() {
  // --- 1. 读取与滤波 ---
  int rawH = analogRead(PIN_EOG_HORZ) / 4; 
  int rawV = analogRead(PIN_EOG_VERT) / 4; 

  filteredHorz = (ALPHA * rawH) + ((1.0 - ALPHA) * filteredHorz);
  filteredVert = (ALPHA * rawV) + ((1.0 - ALPHA) * filteredVert);

  // --- 2. 滑动窗口更新 (移位操作，把新数据放在最后) ---
  for(int i = 0; i < WINDOW_SIZE - 1; i++) {
    buffer_h[i] = buffer_h[i+1];
    buffer_v[i] = buffer_v[i+1];
  }
  buffer_h[WINDOW_SIZE - 1] = filteredHorz;
  buffer_v[WINDOW_SIZE - 1] = filteredVert;
  
  frame_count++;

  // --- 3. 实时 AI 推理！(每收集满 50 帧后，每帧都预测) ---
  if (frame_count >= WINDOW_SIZE) {
    
    // A. 提取 14 维特征
    extract_features(); 
    
    // B. 数据归一化
    scale_features(raw_features, scaled_features); 
    
    // C. 实例化模型并获取初步预测结果
    Eloquent::ML::Port::RandomForest classifier;
    int pred_idx = classifier.predict(scaled_features);
    String raw_label = CLASSES[pred_idx];
    
    String final_label = "Rest";

    // =========================================================
    // 🧠 核心修改：C++ 版本的启发式规则 (速度与极性双重校验)
    // =========================================================
    if (raw_label != "Rest") {
        
        float v_max = raw_features[10];
        float v_min = raw_features[11];
        float v_velocity = raw_features[13];
        
        const float VELOCITY_THRESHOLD_HIGH = 42.0;
        const float VELOCITY_THRESHOLD_LOW = 35.0;

        // --- 规则 1: Up vs Blink (同极性，靠速度区分) ---
        if (raw_label == "Up" && v_velocity < VELOCITY_THRESHOLD_LOW) {
            // Serial.println("🐌 Up速度慢 -> 改 Blink");
            final_label = "Blink";
        } 
        else if (raw_label == "Blink" && v_velocity > VELOCITY_THRESHOLD_HIGH) {
            if (v_max > abs(v_min)) { 
                // Serial.println("🚀 Blink速度快 -> 改 Up");
                final_label = "Up";
            } else {
                final_label = "Blink";
            }
        }
        
        // --- 规则 2: Down vs Blink (异极性，靠电压正负区分) ---
        else if (raw_label == "Down") {
            if (v_max > abs(v_min) * 1.5) { 
                // Serial.println("⚠️ Down 极性不符 (正波) -> 改 Blink");
                final_label = "Blink";
            } else {
                final_label = "Down";
            }
        } 
        else if (raw_label == "Blink") {
            if (abs(v_min) > v_max * 1.5) {
                // Serial.println("⚠️ Blink 极性不符 (负波) -> 改 Down");
                final_label = "Down";
            } else {
                final_label = "Blink";
            }
        } 
        
        // --- 其他情况 (Left, Right) 相信模型 ---
        else {
            final_label = raw_label;
        }
    }
    // =========================================================

    // D. 触发最终动作并进入冷却！
    if (final_label != "Rest") {
      Serial.print("⚡ 最终判定: ");
      Serial.print(final_label);
      Serial.print(" (原判:"); Serial.print(raw_label);
      Serial.print(" | Vel:"); Serial.print(raw_features[13]);
      Serial.println(")");
      
      // 🛡️ 防回弹机制：通过物理延时来实现
      // Python 里我们是用帧数 (frames_since_last < 30) 来屏蔽回弹
      // 在单片机里，更简单粗暴的做法是直接 delay。
      // 识别到动作后，直接让单片机“睡” 400 毫秒，完美避开眼球回正产生的反向波形！
      delay(400); 
      frame_count = 0; // 重置缓冲区计数器，重新收集干净的 50 帧
    }
  }

  // --- 4. 保持 50Hz 采样率 ---
  delay(20); 
}

// ==========================================
// 核心特征提取函数实现 (不要改动)
// ==========================================
void extract_features() {
  float sig_h[WINDOW_SIZE], sig_v[WINDOW_SIZE];
  float sum_h = 0, sum_v = 0;
  
  for(int i = 0; i < WINDOW_SIZE; i++) {
    sum_h += buffer_h[i];
    sum_v += buffer_v[i];
  }
  float mean_h_raw = sum_h / WINDOW_SIZE;
  float mean_v_raw = sum_v / WINDOW_SIZE;
  
  for(int i = 0; i < WINDOW_SIZE; i++) {
    sig_h[i] = buffer_h[i] - mean_h_raw;
    sig_v[i] = buffer_v[i] - mean_v_raw;
  }
  
  calculate_axis_features(sig_h, 0);
  calculate_axis_features(sig_v, 7);
}

void calculate_axis_features(float* sig, int offset) {
  float sum = 0, sum_sq = 0;
  float max_val = -9999.0, min_val = 9999.0;
  float diff_sum = 0, diff_max = -9999.0;
  
  for(int i = 0; i < WINDOW_SIZE; i++) {
    sum += sig[i];
    sum_sq += sig[i] * sig[i];
    
    if(sig[i] > max_val) max_val = sig[i];
    if(sig[i] < min_val) min_val = sig[i];
    
    if (i > 0) {
      float diff = abs(sig[i] - sig[i-1]);
      diff_sum += diff;
      if (diff > diff_max) diff_max = diff;
    }
  }
  
  float mean_val = sum / WINDOW_SIZE;
  float variance = (sum_sq / WINDOW_SIZE) - (mean_val * mean_val);
  float std_val = variance > 0 ? sqrt(variance) : 0;
  
  raw_features[offset + 0] = std_val;                  
  raw_features[offset + 1] = max_val - min_val;        
  raw_features[offset + 2] = mean_val;                 
  raw_features[offset + 3] = max_val;                  
  raw_features[offset + 4] = min_val;                  
  raw_features[offset + 5] = diff_sum / (WINDOW_SIZE - 1); 
  raw_features[offset + 6] = diff_max;                 
}