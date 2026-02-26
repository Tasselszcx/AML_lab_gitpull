#include <Wire.h>
#include "USB.h"
#include "USBHIDMouse.h"

USBHIDMouse Mouse;

const uint8_t IMU_ADDR = 0x69; // 你的正确地址

// ==========================================
// 🎮 鼠标手感调优区 (C同学专属)
// ==========================================
// 🌟 1. 灵敏度调低 (原来是 0.05，现在改成 0.02)
// 如果你觉得还是太快，可以继续降到 0.01。如果太慢，可以加到 0.03。
const float SENSITIVITY = 0.02; 

// 🌟 2. 开启死区防漂移 (滤除低于这个数值的微小抖动)
// 如果放在桌子上还是会微微移动，就把 250 改大(比如 300)
// 如果觉得头转得很累鼠标才开始动，就把 250 改小(比如 150)
const int DEADZONE = 250; 
// ==========================================

void setup() {
  delay(2000); 

  // 初始化自带的黄色 LED 灯
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW); 

  Wire.begin();
  Mouse.begin();
  USB.begin();

  // 1. 尝试唤醒 IMU
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  uint8_t error = Wire.endTransmission(true);

  if (error == 0) {
    // ✅ 成功找到 IMU：黄灯长亮 3 秒
    digitalWrite(LED_BUILTIN, HIGH);
    delay(3000);
    digitalWrite(LED_BUILTIN, LOW);
  } else {
    // ❌ 找不到 IMU：闪烁报错
    while (true) {
      digitalWrite(LED_BUILTIN, HIGH); delay(100);
      digitalWrite(LED_BUILTIN, LOW);  delay(100);
    }
  }

  // 2. 配置量程
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x08);
  Wire.endTransmission(true);
}

void loop() {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x43);
  Wire.endTransmission(false);
  Wire.requestFrom((uint16_t)IMU_ADDR, (uint8_t)6, true);

  if (Wire.available() >= 6) {
    int16_t gyro_x = Wire.read() << 8 | Wire.read();
    int16_t gyro_y = Wire.read() << 8 | Wire.read();
    int16_t gyro_z = Wire.read() << 8 | Wire.read();

    // 🌟 3. 重启死区过滤！解决缓慢漂移
    if (abs(gyro_x) < DEADZONE) gyro_x = 0; 
    if (abs(gyro_y) < DEADZONE) gyro_y = 0;

    // 🌟 4. 使用新的低灵敏度计算移动量
    int mouse_move_x = (gyro_x * SENSITIVITY); 
    int mouse_move_y = (gyro_y * SENSITIVITY); 

    if (mouse_move_x != 0 || mouse_move_y != 0) {
      Mouse.move(mouse_move_x, mouse_move_y);
      // 有动作时黄灯亮起
      digitalWrite(LED_BUILTIN, HIGH); 
    } else {
      // 没动作(或被死区过滤掉)时黄灯熄灭
      digitalWrite(LED_BUILTIN, LOW);
    }
  }
  
  delay(10); 
}