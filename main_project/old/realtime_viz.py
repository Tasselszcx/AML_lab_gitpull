import serial
import numpy as np
import joblib
from collections import deque
import scipy.signal as signal
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import time

# ==== 1. 配置区域 (请根据你的实际情况修改) ====
SERIAL_PORT = 'COM3'       # Windows可能是 COM3/4/5，Mac是 /dev/cu.usbmodem...
BAUD_RATE = 115200         # 必须与 Arduino 一致
MODEL_PATH = 'eog_model_v2.joblib'
SCALER_PATH = 'eog_scaler.joblib'
GAIN_H = 12.0  # 水平信号放大 2 倍 (150 -> 300)
GAIN_V = 20.0  # 垂直信号放大 5 倍 (30 -> 150)

# 必须与 Jupyter Notebook 完全一致！
WINDOW_SIZE = 50           # 窗口大小
CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]
FILTER_BUFFER_SIZE = 150

# 冷却时间 (帧数)
COOLDOWN_FRAMES = 10       
CONFIDENCE_THRESHOLD = 0.7 # 置信度阈值 (0.7 = 70%)

# ==== 2. 信号处理函数 ====
def apply_realtime_filter(window_data):
    """
    对短窗口进行滤波。
    注意：在实时流中对短窗口使用 filtfilt 会有边缘效应，
    但为了匹配训练逻辑，我们这里依然使用它作为近似方案。
    """
    fs = 50.0  # 采样率
    lowcut = 0.5
    highcut = 10.0
    order = 4
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    
    # 对 H (col 0) and V (col 1) 分别滤波
    filtered_window = np.zeros_like(window_data)
    filtered_window[:, 0] = signal.filtfilt(b, a, window_data[:, 0])
    filtered_window[:, 1] = signal.filtfilt(b, a, window_data[:, 1])
    return filtered_window

def extract_features(window_data):
    """
    提取 14 个特征 (2通道 * 7特征)，必须与 Notebook Module 4 一致
    """
    features = []
    for axis in range(2): # 0=H, 1=V
        sig = window_data[:, axis]
        
        # 1. Std
        features.append(np.std(sig))
        # 2. P2P
        features.append(np.max(sig) - np.min(sig))
        # 3. Mean Velocity
        diff = np.diff(sig)
        features.append(np.mean(np.abs(diff)))
        # 4. Max Velocity
        features.append(np.max(np.abs(diff)))
        # 5. Skewness
        features.append(skew(sig))
        # 6. Kurtosis
        features.append(kurtosis(sig))
        # 7. Energy
        features.append(np.sum(sig**2))
        
    return np.array(features).reshape(1, -1)

# ==== 3. 主程序 ====
def main():
    # --- 加载资源 ---
    print("Loading model & scaler...")
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ System Ready!")

    # --- 串口连接 ---
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ Serial Error: {e}")
        return

    # --- 初始化绘图窗口 ---
    plt.ion() # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 子图 1: 波形图
    x_axis = np.arange(WINDOW_SIZE)
    line_h, = ax1.plot(x_axis, np.zeros(WINDOW_SIZE), 'b-', label='Horizontal (L/R)')
    line_v, = ax1.plot(x_axis, np.zeros(WINDOW_SIZE), 'orange', label='Vertical (U/D/B)')
    ax1.set_ylim(-200, 200) # 根据你的信号幅度调整这里！
    ax1.set_title("Real-time EOG Signal (Filtered)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 子图 2: 预测结果展示
    bar_rects = ax2.bar(CLASSES, np.zeros(len(CLASSES)), color='gray')
    ax2.set_ylim(0, 1)
    ax2.set_title("Model Confidence")
    
    # 结果文字标签
    text_pred = ax1.text(0, 150, "Waiting...", fontsize=20, color='red', fontweight='bold')

    # --- 缓冲区 ---
    # buffer_len 设长一点用于绘图流畅，但推理只取最后 WINDOW_SIZE
    plot_buffer_size = 100 
    raw_buffer = deque(maxlen=FILTER_BUFFER_SIZE) 
    
    cooldown = 0
    
    print("🚀 Starting Inference Loop... (Press Ctrl+C to stop)")

    while True:
        try:
            # 1. 串口读取 (非阻塞尝试)
            while ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        raw_h = float(parts[0])
                        raw_v = float(parts[1])
                        # 乘上各自的系数
                        val_h = raw_h * GAIN_H 
                        val_v = raw_v * GAIN_V
                        raw_buffer.append([val_h, val_v])
                    except ValueError:
                        pass
            
            # 2. 【修改】只有当数据填满 150 个点时才处理
            if len(raw_buffer) >= FILTER_BUFFER_SIZE:
                

                # A. 【核心修改】取出长窗口 (150点) 进行滤波
                long_window = np.array(list(raw_buffer))[-FILTER_BUFFER_SIZE:]
                filtered_long = apply_realtime_filter(long_window)
                
                # B. 【核心修改】截取最后 WINDOW_SIZE (50点) 给模型
                # 这样 final_window 的波形就是锐利且干净的
                final_window = filtered_long[-WINDOW_SIZE:]
                
                # C. 特征提取 & 归一化
                feats = extract_features(final_window)
                feats_scaled = scaler.transform(feats)
                
                # D. 推理
                probs = clf.predict_proba(feats_scaled)[0]
                pred_idx = np.argmax(probs)
                pred_label = CLASSES[pred_idx]
                confidence = probs[pred_idx]

                # ==== 【修正后的】上帝视角 Debug 打印 ====
                # 放在这里才是安全的！
                # 只有当不是 Rest 的时候才打印，避免刷屏，这很重要！
                
                # feats[0][1] 是水平 P2P, feats[0][2] 是水平速度
                p2p_h = feats[0][1]
                vel_h = feats[0][2]
                print(f"🔍 DEBUG: Action={pred_label} | P2P_H={p2p_h:.1f} | Velocity_H={vel_h:.2f} | Conf={confidence:.2f}")
                # =========================================

                # E. 【补全】冷却逻辑 & 状态文本生成 (你刚才漏掉的部分)
                if cooldown > 0:
                    cooldown -= 1
                    status_text = f"Cooldown... ({pred_label})"
                    text_color = 'gray'
                else:
                    if pred_label != "Rest" and confidence > CONFIDENCE_THRESHOLD:
                        status_text = f"DETECTED: {pred_label} ({confidence:.0%})"
                        text_color = 'green'
                        cooldown = COOLDOWN_FRAMES # 触发后进入冷却
                    else:
                        status_text = "Resting..."
                        text_color = 'black'

                # --- 3. 动态刷新图表 ---
                
                # 更新波形线 (只显示最后 WINDOW_SIZE 个滤波后的点)
                line_h.set_ydata(final_window[:, 0])
                line_v.set_ydata(final_window[:, 1])
                
                # 更新文字
                text_pred.set_text(status_text)
                text_pred.set_color(text_color)
                
                # 更新概率柱状图
                for rect, prob in zip(bar_rects, probs):
                    rect.set_height(prob)
                    if prob == confidence and pred_label != "Rest" and prob > CONFIDENCE_THRESHOLD:
                         rect.set_color('green')
                    else:
                         rect.set_color('gray')

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
        except KeyboardInterrupt:
            break
        except Exception as e:
            pass

    plt.ioff()
    plt.show()
    ser.close()

if __name__ == "__main__":
    main()