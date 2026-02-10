import serial
import numpy as np
import joblib
from collections import deque
import time

# ==== 1. 配置区域 ====
SERIAL_PORT = 'COM3'      # 请修改为你的 Arduino 端口
BAUD_RATE = 115200        # 必须与 Arduino 代码一致
MODEL_PATH = 'eog_model.joblib'

# 必须与训练时完全一致！
WINDOW_SIZE = 60          
CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]

# 冷却机制：防止重复触发
# 识别到一个动作后，暂停多少帧不识别 (比如 20帧 ≈ 0.4秒)
COOLDOWN_FRAMES = 15      

# ==== 2. 特征提取函数 (必须与训练代码一致！) ====
def extract_realtime_features(window_data):
    """
    输入: window_data (60, 2) 的 numpy 数组
    输出: (1, N_features) 的特征向量
    """
    features = []
    
    # 针对 H (idx 0) 和 V (idx 1) 两个通道
    for axis in range(2): 
        signal = window_data[:, axis]
        
        # --- 这里必须复制你在 train_model.py 里用的特征 ---
        # 如果你只用了标准差和峰峰值：
        
        features.append(np.mean(signal))       # 均值
        features.append(np.std(signal))        # 标准差
        features.append(np.max(signal))        # 最大值
        features.append(np.min(signal))        # 最小值
        features.append(np.max(signal) - np.min(signal)) # 峰峰值
        # [可选] 如果你刚才加了新特征 (diff)，请把下面取消注释
        # diff = np.diff(signal)
        # features.append(np.mean(np.abs(diff))) 
        # features.append(np.sum(np.abs(diff)))
        
    return np.array(features).reshape(1, -1) # 变成 (1, 10) 的形状

# ==== 3. 主程序 ====
def main():
    # A. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    try:
        clf = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # B. 连接串口
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"✅ Connected to {SERIAL_PORT}")
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Serial error: {e}")
        return

    # C. 初始化缓冲区 (FIFO队列)
    # deque 会自动移除旧数据，保持长度为 WINDOW_SIZE
    data_buffer = deque(maxlen=WINDOW_SIZE)
    
    print("\nSystem Ready! Waiting for data...\n")
    print("-" * 40)

    cooldown_counter = 0

    while True:
        try:
            # 1. 读取一行串口数据
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # 假设 Arduino 发送的是 "EOG_H \t EOG_V" (纯数字)
                parts = line.split('\t')
                
                if len(parts) >= 2:
                    try:
                        val_h = float(parts[0])
                        val_v = float(parts[1])
                        
                        # 2. 存入缓冲区
                        data_buffer.append([val_h, val_v])
                    except ValueError:
                        continue

                    # 3. 只有缓冲区填满后，才开始预测
                    if len(data_buffer) == WINDOW_SIZE:
                        
                        # 如果在冷却时间内，跳过预测
                        if cooldown_counter > 0:
                            cooldown_counter -= 1
                            continue
                        
                        # 4. 提取特征并预测
                        # 将 deque 转为 numpy array
                        window_np = np.array(data_buffer)
                        
                        # 提取特征
                        feat = extract_realtime_features(window_np)
                        
                        # 推理
                        prediction_idx = clf.predict(feat)[0]
                        predicted_label = CLASSES[prediction_idx]
                        
                        # 获取置信度 (概率)
                        probs = clf.predict_proba(feat)[0]
                        confidence = probs[prediction_idx]

                        # 5. 输出逻辑 (简单的阈值过滤)
                        # 只有当不是 Rest 且 置信度 > 0.6 时才触发
                        if predicted_label != "Rest" and confidence > 0.6:
                            
                            # 打印酷炫的输出
                            if predicted_label == "Left":
                                print(f"⬅️  LEFT  ({confidence:.2f})")
                            elif predicted_label == "Right":
                                print(f"➡️  RIGHT ({confidence:.2f})")
                            elif predicted_label == "Up":
                                print(f"⬆️  UP    ({confidence:.2f})")
                            elif predicted_label == "Down":
                                print(f"⬇️  DOWN  ({confidence:.2f})")
                            elif predicted_label == "Blink":
                                print(f"👁️  BLINK ({confidence:.2f})")
                            
                            # 触发一次动作后，进入冷却，防止刷屏
                            cooldown_counter = COOLDOWN_FRAMES
                            # 清空缓冲区的一半，避免同一个波形被重复切片识别
                            # (这是可选的，取决于你想要多灵敏)
                            # data_buffer.clear() 

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    ser.close()

if __name__ == "__main__":
    main()