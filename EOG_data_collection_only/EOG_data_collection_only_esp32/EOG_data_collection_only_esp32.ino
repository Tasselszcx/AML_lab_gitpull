/*
 * [Phase 1 - ESP32 升级版] EOG Data Collection Code
 * 硬件主控: Arduino Nano ESP32
 * 传感器: BioAmp EXG Pill x 2
 */

// ---- 1. 引脚定义 ----
const int PIN_EOG_HORZ = A0; // 水平眼电
const int PIN_EOG_VERT = A1; // 垂直眼电

// ---- 2. 滤波参数 (Filter Settings) ----
const float ALPHA = 0.2; 

// 变量存储 (基线设为 512)
float filteredHorz = 512.0; 
float filteredVert = 512.0;

void setup() {
  Serial.begin(115200);
  
  // 删除了报错的 analogReadResolution(10); 
  // 我们在 loop 里面用除法来完美替代它！
  
  Serial.println("System Ready: ESP32 starting in software-scaled 10-bit mode.");
}

void loop() {
  // ---- 第一步: 读取并强制降维 (神级替换) ----
  // ESP32 默认读出来是 0-4095。
  // 我们直接除以 4 (或者右移2位 >> 2)，它就完美变成了 0-1023！
  // 加上分压电路，2.5V 读出来就是 2048 / 4 = 512。和 Uno 一模一样！
  
  int rawHorz = analogRead(PIN_EOG_HORZ) / 4; 
  int rawVert = analogRead(PIN_EOG_VERT) / 4; 

  // ---- 第二步: 信号平滑处理 (EMA滤波) ----
  filteredHorz = (ALPHA * rawHorz) + ((1.0 - ALPHA) * filteredHorz);
  filteredVert = (ALPHA * rawVert) + ((1.0 - ALPHA) * filteredVert);

  // ---- 第三步: 输出格式化数据 ----
  // 1. 输出水平信号
  //Serial.print("H:");          
  Serial.print(filteredHorz);  
  Serial.print("\t");          

  // 2. 输出垂直信号
  //Serial.print("V:");          
  Serial.println(filteredVert); 

  // ---- 第四步: 采样率控制 ----
  delay(20); 
}