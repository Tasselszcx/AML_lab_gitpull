/*
 * Gaze & Nod - Interim Report IMU Data Collection
 * Hardware: ICM20600 + AK09918 (AML Lab Box)
 * Purpose: Capture Head Gestures (Nod vs Shake) for Presentation
 */

#include <Wire.h>
#include <SPI.h>
#include "AK09918.h"
#include "ICM20600.h"

// 实例化传感器对象
AK09918 ak09918;
ICM20600 icm20600(true);

// 采样率控制
const int LOOP_DELAY_MS = 10; // 100Hz 采样率

void setup() {
  // 1. 初始化串口 (用于 Serial Plotter 画图)
  Serial.begin(115200);
  while (!Serial) {};

  // 2. 初始化 I2C
  Wire.begin();

  // 3. 初始化 IMU (ICM20600)
  // 这是关键：我们需要同时读取加速度和陀螺仪
  icm20600.initialize();

  // 4. 初始化磁力计 (AK09918) - 虽然这次主要用陀螺仪，但先初始化防止报错
  ak09918.initialize();
  ak09918.switchMode(AK09918_POWER_DOWN);
  ak09918.switchMode(AK09918_CONTINUOUS_100HZ);

  Serial.println("System Ready. Open Serial Plotter (Tools > Serial Plotter).");
  delay(1000);
}

void loop() {
  // ---- 第一步：读取 6轴 原始数据 ----
  
  // A. 读取陀螺仪 (Gyroscope) - 单位：dps (degrees per second) 的原始值
  // 摇头和点头主要看这三个数据！
  int16_t gx = icm20600.getGyroscopeX();
  int16_t gy = icm20600.getGyroscopeY();
  int16_t gz = icm20600.getGyroscopeZ();

  // B. 读取加速度计 (Accelerometer) - 辅助观察
  int16_t ax = icm20600.getAccelerationX();
  int16_t ay = icm20600.getAccelerationY();
  int16_t az = icm20600.getAccelerationZ();

  // ---- 第二步：输出到 Serial Plotter ----
  // 格式必须是 "Label:Value \t Label:Value..."
  // 为了让 PPT 图表清晰，我们只输出最关键的几条线，避免杂乱

  // // 1. 输出 Gyro X (通常对应点点头/Pitch)
  // Serial.print("Gyro_X(Nod):");
  // Serial.print(gx);
  // Serial.print("\t");

  // // 2. 输出 Gyro Z (通常对应摇头/Shake)
  // Serial.print("Gyro_Z(Shake):");
  // Serial.print(gz);
  // Serial.print("\t");
  // //确保中间有空格或逗号，且每行结尾有 println
  // // 3. 输出 Accel Z (辅助参考，静止时应该是重力)
  // // 为了不让加速度的数值把陀螺仪的波形压扁，我们可以稍微放大或忽略它
  // // 这里保留它作为参考，但如果波形太乱，可以在 PPT 截图时取消勾选
  // Serial.print("Acc_Z:");
  // Serial.println(az);
  Serial.print(gx);      // 输出第一个数
  Serial.print(" ");     // 输出一个空格
  Serial.println(gz);    // 输出第二个数并换行 (println)
  // 控制采样率
  delay(LOOP_DELAY_MS);
}