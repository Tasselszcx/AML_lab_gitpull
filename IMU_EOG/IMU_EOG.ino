/*
 * Final Integration: Dual EOG + IMU
 * for Better Serial Plotter
 */

#include <Wire.h>
#include <SPI.h>
#include "ICM20600.h"

ICM20600 icm20600(true);

// ---- 引脚定义 ----
const int PIN_EOG_HORZ = A0; // 水平眼电模块
const int PIN_EOG_VERT = A1; // 垂直眼电模块

// ---- 滤波参数 (让波形更平滑) ----
// alpha 越小越平滑，但反应越慢。0.2 是个不错的平衡点。
const float ALPHA = 0.2; 
float valHorz = 0;
float valVert = 0;

void setup() {
  // 1. 开启串口 (BSP 需要 115200)
  Serial.begin(115200);
  while (!Serial) {};

  // 2. 初始化 IMU
  Wire.begin();
  icm20600.initialize();

  // 3. 预热提示
  // Serial.println("System Ready"); // BSP连接前不要打印杂乱信息
}

void loop() {
  // ---- 1. 读取 EOG 原始数据 (0-1023) ----
  int rawHorz = analogRead(PIN_EOG_HORZ);
  int rawVert = analogRead(PIN_EOG_VERT);

  // ---- 2. 简单滤波 (EMA Filter) ----
  // 模拟去噪，让 Presentation 的波形更好看
  valHorz = (ALPHA * rawHorz) + ((1.0 - ALPHA) * valHorz);
  valVert = (ALPHA * rawVert) + ((1.0 - ALPHA) * valVert);

  // ---- 3. 读取 IMU 数据 ----
  // 我们只需要陀螺仪数据来检测点头/摇头
  int16_t gx = icm20600.getGyroscopeX(); // 点头 (Nod)
  int16_t gz = icm20600.getGyroscopeZ(); // 摇头 (Shake)

  // ---- 4. 输出给 Better Serial Plotter ----
  // 格式: Label:Value \t Label:Value \n
  
  // A. 输出水平眼电
  Serial.print("EOG_H:");
  Serial.print(valHorz);
  Serial.print("\t"); // 制表符分隔

  // B. 输出垂直眼电
  Serial.print("EOG_V:");
  Serial.print(valVert);
  Serial.print("\t");

  // C. 输出陀螺仪 X (点头)
  // 除以 10 是为了让波形高度和 EOG 差不多，方便在同一个图里看
  Serial.print("Gyro_Nod:"); 
  Serial.print(gx / 10.0); 
  Serial.print("\t");

  // D. 输出陀螺仪 Z (摇头)
  Serial.print("Gyro_Shake:");
  Serial.println(gz / 10.0); // 最后用 println 换行

  Serial.print(valHorz);
  Serial.print(" "); // 制表符分隔
  Serial.print(valVert);
  Serial.print(" ");
  Serial.print(gx / 10.0); 
  Serial.print(" ");
  Serial.println(gz / 10.0); // 最后用 println 换行
  // Serial.print(gx);      // 输出第一个数
  // Serial.print(" ");     // 输出一个空格
  // Serial.println(gz);    // 输出第二个数并换行 (println)

  // ---- 5. 采样率控制 ----
  delay(20); // 50Hz，既流畅又不会太快
}