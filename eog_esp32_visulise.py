import sys
import time
import serial
import threading
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from scipy import signal as scipy_signal

# ================= 1. 配置区域 =================
SERIAL_PORT = 'COM5'       # ⚠️ 请根据你的设备修改端口
BAUD_RATE = 115200
MODEL_PATH = "main_project/models/models_xkp/eog_model_v4.joblib"
SCALER_PATH = "main_project/models/models_xkp/eog_scaler_v4.joblib"

SAMPLING_RATE = 50
WINDOW_SIZE = 50           
DISPLAY_LEN = 300          # 屏幕显示的点数
FILTER_BUFFER_SIZE = 150   

# 视觉增强系数
GAIN_H = 1.0  
GAIN_V = 1.0  
CONFIDENCE_THRESHOLD = 0.6 
COOLDOWN_FRAMES = 15       

CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]

# ================= 2. 核心系统类 =================

class EOGSystem:
    def __init__(self, model_path, scaler_path):
        self.running = True
        self.lock = threading.Lock()
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

        # 绘图缓冲区
        self.plot_h = deque([512]*DISPLAY_LEN, maxlen=DISPLAY_LEN)
        self.plot_v = deque([512]*DISPLAY_LEN, maxlen=DISPLAY_LEN)
        
        # 算法缓冲区
        self.raw_buffer = deque(maxlen=FILTER_BUFFER_SIZE)
        
        # 状态变量
        self.cooldown = 0
        self.last_pred = "Rest"
        self.last_conf = 0.0
        self.frame_counter = 0 
        self.detected_frame_idx = -999 

    def apply_realtime_filter(self, window_data):
        # 这里的滤波仅用于辅助推理，不破坏画出的原始波形
        fs = 50.0 
        nyq = 0.5 * fs
        b, a = scipy_signal.butter(4, [0.5/nyq, 10.0/nyq], btype='band')
        return scipy_signal.filtfilt(b, a, window_data, axis=0)

    def extract_features(self, window_data):
        features = []
        for axis in range(2):
            sig = window_data[:, axis]
            sig = sig - np.mean(sig) # 局部去直流
            
            features.append(np.std(sig))
            features.append(np.max(sig) - np.min(sig))
            features.append(np.mean(sig))
            features.append(np.max(sig))
            features.append(np.min(sig))
            diff = np.diff(sig)
            features.append(np.mean(np.abs(diff)))
            features.append(np.max(np.abs(diff)))
            
        return np.array(features).reshape(1, -1)

    def process_new_data(self, h_val, v_val):
        self.frame_counter += 1
        self.raw_buffer.append([h_val, v_val])

        with self.lock:
            self.plot_h.append(h_val)
            self.plot_v.append(v_val)
        
        if len(self.raw_buffer) < FILTER_BUFFER_SIZE:
            return

        # 推理逻辑
        if self.cooldown > 0:
            self.cooldown -= 1
        else:
            long_window = np.array(self.raw_buffer)
            filtered_long = self.apply_realtime_filter(long_window)
            final_window = filtered_long[-WINDOW_SIZE:]
            
            feats = self.extract_features(final_window)
            feats_scaled = self.scaler.transform(feats)
            probs = self.model.predict_proba(feats_scaled)[0]
            pred_idx = np.argmax(probs)
            
            if probs[pred_idx] > CONFIDENCE_THRESHOLD and CLASSES[pred_idx] != "Rest":
                with self.lock:
                    self.last_pred = CLASSES[pred_idx]
                    self.last_conf = probs[pred_idx]
                    self.cooldown = COOLDOWN_FRAMES
                    self.detected_frame_idx = self.frame_counter

# ================= 3. 串口读取线程 =================

def serial_thread(system):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"✅ 串口 {SERIAL_PORT} 已连接")
    except Exception as e:
        print(f"❌ 串口连接失败: {e}")
        system.running = False
        return

    while system.running:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line: continue
        try:
            # 兼容 "H:512 V:512" 或 "512 512" 格式
            parts = line.replace(',', '\t').replace('H:', '').replace('V:', '').split()
            if len(parts) >= 2:
                h_val = float(parts[0]) * GAIN_H
                v_val = float(parts[1]) * GAIN_V
                system.process_new_data(h_val, v_val)
        except:
            continue
    ser.close()

# ================= 4. 主程序 (绘图) =================

def main():
    system = EOGSystem(MODEL_PATH, SCALER_PATH)
    t = threading.Thread(target=serial_thread, args=(system,), daemon=True)
    t.start()

    # 创建绘图窗口
    plt.style.use('dark_background') # 使用黑背景更像示波器
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title('EOG Combined Waveform Monitor')

    # 在同一个坐标轴画两条线
    line_h, = ax.plot([], [], label='Horizontal (L/R)', color='#00d2ff', lw=2, alpha=0.9)
    line_v, = ax.plot([], [], label='Vertical (U/D)', color='#ff9d00', lw=2, alpha=0.9)
    
    # 标注文字
    text_label = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=20, fontweight='bold')
    text_info = ax.text(0.02, 0.02, "Waiting for data...", transform=ax.transAxes, color='gray')

    # 设置坐标轴
    ax.set_xlim(0, DISPLAY_LEN)
    ax.set_ylim(300, 724) # 聚焦在 512 附近的信号，如果溢出请改回 0-1024
    ax.set_title("Real-time EOG Analysis (Dual Channel Combined)", color='white', pad=20)
    ax.legend(loc='upper right', frameon=True)
    ax.set_ylabel("ADC Value")
    ax.set_xlabel("Time (Samples)")

    x_data = np.arange(DISPLAY_LEN)

    def update(frame):
        if not system.running: return line_h, line_v

        with system.lock:
            h_vals = list(system.plot_h)
            v_vals = list(system.plot_v)
            pred = system.last_pred
            conf = system.last_conf
            frames_ago = system.frame_counter - system.detected_frame_idx

        line_h.set_data(x_data, h_vals)
        line_v.set_data(x_data, v_vals)

        # 判定结果显示（触发后显示 10 帧的时间）
        if frames_ago < 10:
            text_label.set_text(f"EVENT: {pred}")
            text_label.set_color('#00ff00') # 绿色高亮
        else:
            text_label.set_text("STATUS: Monitoring...")
            text_label.set_color('#aaaaaa')

        text_info.set_text(f"Frame: {system.frame_counter} | Confidence: {conf:.2f}")
        
        return line_h, line_v, text_label

    ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    
    system.running = False
    t.join()

if __name__ == "__main__":
    main()