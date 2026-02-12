/*
 * [Phase 1] EOG Data Collection Code (Corrected for Python Script)
 * 适配软件: Python Collection Script / Better Serial Plotter
 * 硬件: BioAmp EXG Pill x 2
 * * 接线:
 * - Horizontal (左右): OUT -> A0
 * - Vertical   (上下): OUT -> A1
 * - VCC -> 5V (或 3.3V)
 * - GND -> GND
 */

// ---- 1. 引脚定义 ----
const int PIN_EOG_HORZ = A0; // 水平眼电
const int PIN_EOG_VERT = A1; // 垂直眼电

// ---- 2. 滤波参数 (Filter Settings) ----
const float ALPHA = 0.2; 

// 变量存储
float filteredHorz = 512.0; 
float filteredVert = 512.0;

void setup() {
  // 必须使用 115200 以配合 Python 脚本设置
  Serial.begin(115200);
  
  // 等待串口稳定
  while (!Serial) {};
}

void loop() {
  // ---- 第一步: 读取原始数据 (0-1023) ----
  int rawHorz = analogRead(PIN_EOG_HORZ);
  int rawVert = analogRead(PIN_EOG_VERT);

  // ---- 第二步: 信号平滑处理 (EMA滤波) ----
  filteredHorz = (ALPHA * rawHorz) + ((1.0 - ALPHA) * filteredHorz);
  filteredVert = (ALPHA * rawVert) + ((1.0 - ALPHA) * filteredVert);

  // ---- 第三步: 输出格式化数据 ----
  // Python 脚本期望的格式: "Label1:Value1 \t Label2:Value2"
  // 例如: "H:512.00 \t V:515.00"
  
  // 1. 输出水平信号
  Serial.print("H:");          // 标签
  Serial.print(filteredHorz);  // 数值
  Serial.print("\t");          // 制表符分隔 (关键!)

  // 2. 输出垂直信号
  Serial.print("V:");          // 标签
  Serial.println(filteredVert); // 数值 + 换行符 (表示一帧结束)

  // ---- 第四步: 采样率控制 ----
  // 20ms 延时 ≈ 50Hz 采样率
  delay(20); 
}