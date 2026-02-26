#include <Wire.h>
#include "USB.h"
#include "USBHIDMouse.h"
#include "EOG_AI_Engine.h" 

// 引入你通过 micromlgen 转换的随机森林模型

Eloquent::ML::Port::RandomForest classifier; 
USBHIDMouse Mouse;

// ==========================================================
// 🛠️ 1. 独立测试开关区 (1 = 开启，0 = 关闭)
// ==========================================================
const int enable_imu   = 1; // [IMU开关] 设为0可锁定光标，专心测试EOG
const int enable_blink = 1; // [EOG开关] 眨眼 -> 鼠标左键
const int enable_up    = 0; // [EOG开关] 向上看 -> 滚轮向上
const int enable_down  = 0; // [EOG开关] 向下看 -> 滚轮向下
const int enable_left  = 0; // [EOG开关] 向左看 -> 鼠标右键
const int enable_right = 0; // [EOG开关] 向右看 -> (预留功能)

// ==========================================================
// 🕹️ 2. IMU 灵敏度与死区调参区
// ==========================================================
float sensitivity_x = 0.02; // X轴灵敏度 (默认 0.05，数值越大移动越快)
float sensitivity_y = 0.02; // Y轴灵敏度 (默认 0.05)
int deadzone = 150;          // 死区阈值 (默认 60，过滤静止时的轻微抖动)

// ==========================================================
// 🧠 3. 模型参数区 (✅ 已替换为你提供的真实数据)
// ==========================================================
const float SCALER_MEAN[14] = {20.406534678319083, 72.51687672702651, -2.191173084534968e-17, 37.24214374991456, -35.274732977112, 3.1793810004525946, 13.673141924814066, 44.85300943324361, 162.64939921599193, 2.3557781895811873e-17, 90.12877562252785, -72.52062359346442, 8.261938818383678, 28.399999470780713};

const float SCALER_SCALE[14] = {15.720781813249808, 55.92140472605455, 1.0166116940650565e-15, 31.05605915804176, 29.456410187214125, 2.156343017929521, 10.273585669451226, 40.117112856625354, 142.38487695921324, 1.5928353295539902e-15, 84.42508113315914, 64.19545225016621, 6.703125123744208, 23.89279804038172};

const float SOS_COEFFS[4][5] = {
  {0.039602662652766865, 0.07920532530553373, 0.039602662652766865, -0.4095713880801906, 0.08581465919989595},
  {1.0, 2.0, 1.0, -0.4774637949444202, 0.49701183887639755},
  {1.0, -2.0, 1.0, -1.879501037691993, 0.883839457037366},
  {1.0, -2.0, 1.0, -1.9520334563169257, 0.9559790073946604},
};

// ==========================================================
// ⚙️ 4. 硬件及全局变量配置
// ==========================================================
const uint8_t IMU_ADDR = 0x69; 
const int EOG_H_PIN = A0; 
const int EOG_V_PIN = A1; 

const int WINDOW_SIZE = 50;           
const int SAMPLE_RATE_MS = 20;        
const int COOLDOWN_MS = 1000;         

const int CLASS_REST  = 0;
const int CLASS_UP    = 1;
const int CLASS_DOWN  = 2;
const int CLASS_LEFT  = 3;
const int CLASS_RIGHT = 4;
const int CLASS_BLINK = 5;

// 滤波器状态寄存器
float filter_state_H[4][2] = {0};
float filter_state_V[4][2] = {0};

float buffer_h[WINDOW_SIZE];
float buffer_v[WINDOW_SIZE];
int sample_count = 0; 

unsigned long last_sample_time = 0;
unsigned long last_action_time = 0;
int last_pred_action = CLASS_REST;
float features[14]; 

// ==========================================================
// 🧮 5. 核心算法函数
// ==========================================================
// --- 实时数字滤波器 ---
float apply_filter(float input, float state[4][2]) {
  float out = input;
  for(int i = 0; i < 4; i++) {
    float b0 = SOS_COEFFS[i][0];
    float b1 = SOS_COEFFS[i][1];
    float b2 = SOS_COEFFS[i][2];
    float a1 = SOS_COEFFS[i][3];
    float a2 = SOS_COEFFS[i][4];

    float w0 = out - a1 * state[i][0] - a2 * state[i][1];
    float y = b0 * w0 + b1 * state[i][0] + b2 * state[i][1];

    state[i][1] = state[i][0]; 
    state[i][0] = w0;
    out = y;                   
  }
  return out;
}

// --- 特征提取 ---
void extract_axis_features(float* window_data, int start_idx) {
  float sum = 0, sq_sum = 0, std_dev = 0;
  float max_val = -999999.0, min_val = 999999.0;
  
  for (int i = 0; i < WINDOW_SIZE; i++) sum += window_data[i];
  float mean = sum / WINDOW_SIZE;
  
  for (int i = 0; i < WINDOW_SIZE; i++) {
    float val = window_data[i] - mean;
    if (val > max_val) max_val = val;
    if (val < min_val) min_val = val;
    sq_sum += val * val;
  }
  std_dev = sqrt(sq_sum / WINDOW_SIZE);
  
  float diff_sum = 0, diff_max = 0;
  for (int i = 1; i < WINDOW_SIZE; i++) {
    float diff = abs(window_data[i] - window_data[i-1]); 
    diff_sum += diff;
    if (diff > diff_max) diff_max = diff;
  }
  float diff_mean = diff_sum / (WINDOW_SIZE - 1);
  
  features[start_idx + 0] = std_dev;
  features[start_idx + 1] = max_val - min_val;
  features[start_idx + 2] = 0; 
  features[start_idx + 3] = max_val;
  features[start_idx + 4] = min_val;
  features[start_idx + 5] = diff_mean;
  features[start_idx + 6] = diff_max;
}

// ==========================================================
// 🚀 6. 初始化 (Setup)
// ==========================================================
void setup() {
  
  Serial.begin(115200);
  analogReadResolution(10);
  delay(2000); 
  pinMode(LED_BUILTIN, OUTPUT);
  Wire.begin();
  Mouse.begin();
  USB.begin();

  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x08);
  Wire.endTransmission(true);

  Serial.println("======================================");
  Serial.println("🤖 头部鼠标与眼电识别系统已启动");
  Serial.println("======================================");
}

// ==========================================================
// 🔄 7. 主循环 (Loop)
// ==========================================================
void loop() {
  unsigned long current_time = millis();

  // --------------------------------------------------
  // 模块 A: IMU 控制鼠标光标移动 
  // --------------------------------------------------
  if (enable_imu) {
    Wire.beginTransmission(IMU_ADDR);
    Wire.write(0x43);
    Wire.endTransmission(false);
    Wire.requestFrom((uint16_t)IMU_ADDR, (uint8_t)6, true);
    
    if (Wire.available() >= 6) {
      int16_t gyro_x = Wire.read() << 8 | Wire.read();
      int16_t gyro_y = Wire.read() << 8 | Wire.read();
      int16_t gyro_z = Wire.read() << 8 | Wire.read(); 

      // 死区过滤
      if (abs(gyro_x) < deadzone) gyro_x = 0;
      if (abs(gyro_y) < deadzone) gyro_y = 0;

      // 灵敏度计算映射
      int mouse_move_x = (gyro_x * sensitivity_x); 
      int mouse_move_y = (gyro_y * sensitivity_y); 

      if (mouse_move_x != 0 || mouse_move_y != 0) {
        Mouse.move(mouse_move_x, mouse_move_y, 0);
      }
    }
  }

  // --------------------------------------------------
  // 模块 B: EOG 信号处理与触发
  // --------------------------------------------------
  if (current_time - last_sample_time >= SAMPLE_RATE_MS) {
    last_sample_time = current_time;

    // 1. 采集并实时滤波
    float raw_h = analogRead(EOG_H_PIN);
    float raw_v = analogRead(EOG_V_PIN);
    
    float filtered_h = apply_filter(raw_h, filter_state_H);
    float filtered_v = apply_filter(raw_v, filter_state_V);

    // 2. 更新滑动窗口
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
      buffer_h[i] = buffer_h[i + 1];
      buffer_v[i] = buffer_v[i + 1];
    }
    buffer_h[WINDOW_SIZE - 1] = filtered_h;
    buffer_v[WINDOW_SIZE - 1] = filtered_v;

    if (sample_count < WINDOW_SIZE) {
      sample_count++;
      return; 
    }

    // 3. 提取特征
    extract_axis_features(buffer_h, 0);
    extract_axis_features(buffer_v, 7);

    

    // 💡【关键修复】：在被标准化覆盖之前，备份真实的原始速度数据！
    float raw_v_velocity = features[13]; 

    // 4. 特征标准化 (Z-Score)
    for (int i = 0; i < 14; i++) {
      features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }

    // 5. 模型预测
    int raw_label = classifier.predict(features);
    int final_label = CLASS_REST;

    // =======================================================
    // 🐞 插入调试打印代码 (用来偷看模型的想法)
    // =======================================================
    // 每隔一段时间，打印一下当前电压和模型原始判断
    static int print_counter = 0;
    if (print_counter++ % 10 == 0) { // 每隔10次(约200ms)打印一次，防止串口卡死
      Serial.print("V轴电压:"); Serial.print(raw_v);
      Serial.print(" | 原始速度:"); Serial.print(raw_v_velocity);
      Serial.print(" | 模型原始判断:"); Serial.println(raw_label);
    }
    // =======================================================



    // 6. 速度修正与防回弹 (使用刚才备份的真实速度值)
    float VELOCITY_THRESHOLD_HIGH = 42.0; 
    float VELOCITY_THRESHOLD_LOW = 35.0;

    if (raw_label != CLASS_REST) {
      if ((last_pred_action == CLASS_UP || last_pred_action == CLASS_DOWN) && 
          (current_time - last_action_time < 500)) { 
          final_label = CLASS_REST; // 忽略回弹
      } else {
          // 使用 raw_v_velocity 进行比对
          if (raw_label == CLASS_BLINK && raw_v_velocity > VELOCITY_THRESHOLD_HIGH) {
            final_label = CLASS_UP;
          } else if ((raw_label == CLASS_UP || raw_label == CLASS_DOWN) && raw_v_velocity < VELOCITY_THRESHOLD_LOW) {
            final_label = CLASS_BLINK;
          } else {
            final_label = raw_label;
          }
      }
    }

    // 7. 动作执行与开关判定
    if (final_label != CLASS_REST && (current_time - last_action_time > COOLDOWN_MS)) {
      digitalWrite(LED_BUILTIN, HIGH); 
      
      switch (final_label) {
        case CLASS_BLINK: 
          Serial.println("👀 识别到: 眨眼 (Blink) -> 触发左键");
          if (enable_blink) Mouse.click(MOUSE_LEFT); 
          break;
        case CLASS_UP:    
          Serial.println("⬆️ 识别到: 向上看 (Up) -> 向上滚轮");
          if (enable_up) Mouse.move(0, 0, 5); 
          break;
        case CLASS_DOWN:  
          Serial.println("⬇️ 识别到: 向下看 (Down) -> 向下滚轮");
          if (enable_down) Mouse.move(0, 0, -5); 
          break;
        case CLASS_LEFT:  
          Serial.println("⬅️ 识别到: 向左看 (Left) -> 触发右键");
          if (enable_left) Mouse.click(MOUSE_RIGHT); 
          break;
        case CLASS_RIGHT:
          Serial.println("➡️ 识别到: 向右看 (Right) -> (暂无动作)");
          if (enable_right) { /* 预留执行代码 */ }
          break;
      }
      
      last_action_time = current_time;
      last_pred_action = final_label;
      delay(50); 
      digitalWrite(LED_BUILTIN, LOW);
    }
  }
}