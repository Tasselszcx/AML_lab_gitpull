#include <Wire.h>

// ==== 引脚定义 ====
const int PIN_EOG_H = A0;
const int PIN_EOG_V = A1;

// IMU 的 I2C 地址 (ICM20600 通常是 0x68 或 0x69)
uint8_t imu_address = 0x69; 
bool imu_found = false;

void setup() {
  Serial.begin(115200);
  // 等待串口打开 (如果是直接插电源跑，可以注释掉这句)
  while (!Serial) { delay(10); } 
  
  Serial.println("\n--- 🚀 ESP32 硬件体检程序启动 ---");

  // 1. 设置 ADC 分辨率为 10位 (完美兼容你之前的 Python 模型!)
  // 这样 0-3.3V 就会被映射为 0-1023，2.5V 经过分压后正好在 512 左右。
  analogReadResolution(10); 
  Serial.println("✅ ADC 分辨率已设置为 10-bit (0-1023)");

  // 2. 初始化 I2C 总线并扫描 IMU
  Wire.begin(); 
  Serial.println("🔍 正在扫描 I2C 总线寻找 IMU...");
  
  Wire.beginTransmission(0x68);
  if (Wire.endTransmission() == 0) {
    imu_address = 0x68;
    imu_found = true;
  } else {
    Wire.beginTransmission(0x69);
    if (Wire.endTransmission() == 0) {
      imu_address = 0x69;
      imu_found = true;
    }
  }

  if (imu_found) {
    Serial.print("✅ 找到 IMU 设备! I2C 地址: 0x");
    Serial.println(imu_address, HEX);
    
    // 唤醒 IMU (写入电源管理寄存器 0x6B)
    Wire.beginTransmission(imu_address);
    Wire.write(0x6B); 
    Wire.write(0x00); // 唤醒
    Wire.endTransmission(true);
  } else {
    Serial.println("❌ 未找到 IMU! 请检查 SDA(A4) 和 SCL(A5) 接线。");
  }
  
  Serial.println("--- 📊 开始输出数据 (请打开'串口绘图仪') ---\n");
  delay(1000);
}

void loop() {
  // ==== 1. 读取 EOG 信号 ====
  int eog_h = analogRead(PIN_EOG_H);
  int eog_v = analogRead(PIN_EOG_V);

  // ==== 2. 读取 IMU 信号 (加速度 X, Y, Z) ====
  int16_t ax = 0, ay = 0, az = 0;
  
  if (imu_found) {
    // 指向加速度计数据寄存器首地址 0x3B
    Wire.beginTransmission(imu_address);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    
    // 连续读取 6 个字节 (X, Y, Z 各 2 字节)
    Wire.requestFrom((uint16_t)imu_address, (uint8_t)6, true);
    if (Wire.available() >= 6) {
      ax = Wire.read() << 8 | Wire.read();
      ay = Wire.read() << 8 | Wire.read();
      az = Wire.read() << 8 | Wire.read();
    }
  }

  // ==== 3. 打印数据供串口绘图仪查看 ====
  // 格式: 标签:数值 (用逗号或制表符隔开)
  Serial.print("EOG_H:"); Serial.print(eog_h); Serial.print("\t");
  Serial.print("EOG_V:"); Serial.print(eog_v); Serial.print("\t");
  
  // 为了让 IMU 数据在绘图仪里和 EOG 在同一个视野，我们把 IMU 数据缩小一点
  Serial.print("IMU_X:"); Serial.print(ax / 50); Serial.print("\t");
  Serial.print("IMU_Y:"); Serial.print(ay / 50); Serial.print("\t");
  Serial.print("IMU_Z:"); Serial.println(az / 50);

  // 采样率控制 (20ms = 50Hz)
  delay(20);
}