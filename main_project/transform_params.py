import joblib
import scipy.signal as signal
import numpy as np

# 1. 载入你的 Scaler 模型
scaler_path = "models/models_xkp/eog_scaler_v4.joblib"
scaler = joblib.load(scaler_path)

print("============= 请将以下内容复制到 Arduino 代码的顶部 =============")
print(f"const float SCALER_MEAN[14] = {{{', '.join(map(str, scaler.mean_))}}};")
print(f"const float SCALER_SCALE[14] = {{{', '.join(map(str, scaler.scale_))}}};")

# 2. 生成 C++ 可用的滤波器系数 (SOS/Biquad 格式)
fs = 50.0
nyq = 0.5 * fs
# 4阶带通会生成 4个 二阶节(SOS)
sos = signal.butter(4, [0.5/nyq, 10.0/nyq], btype='band', output='sos')

print("\n// 4阶带通滤波器系数 (b0, b1, b2, a1, a2) - 注意略去了 a0=1.0")
print("const float SOS_COEFFS[4][5] = {")
for row in sos:
    print(f"  {{{row[0]}, {row[1]}, {row[2]}, {row[4]}, {row[5]}}},")
print("};")
print("==================================================================")