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
from scipy.stats import skew, kurtosis
from numpy.fft import rfft, rfftfreq

# ================= 配置区域 =================
SERIAL_PORT = 'COM6'   # 请确认端口
BAUD_RATE = 115200
MODEL_PATH = "models/rf_model.pkl"

SAMPLING_RATE = 50
WINDOW_SIZE = 75      # 窗口大小 (1秒)
HALF_WIN = WINDOW_SIZE // 2
DISPLAY_LEN = 300     # 屏幕显示长度

ENERGY_THRESHOLD = 10000  
COOLDOWN_FRAMES = 40      

CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]

# ================= 核心类定义 =================

class EOGSystem:
    def __init__(self, model_path):
        self.running = True
        self.lock = threading.Lock()
        
        try:
            self.model = joblib.load(model_path)
            print(f"模型已加载: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)

        # 数据缓冲区
        self.raw_h = deque([512]*DISPLAY_LEN, maxlen=DISPLAY_LEN)
        self.raw_v = deque([512]*DISPLAY_LEN, maxlen=DISPLAY_LEN)
        self.energy_history = deque([0]*DISPLAY_LEN, maxlen=DISPLAY_LEN)
        
        # 处理缓冲区
        self.proc_buffer_h = deque([512]*100, maxlen=100)
        self.proc_buffer_v = deque([512]*100, maxlen=100)

        # 滤波器状态
        nyq = 0.5 * SAMPLING_RATE
        b, a = scipy_signal.butter(4, [0.5 / nyq, 10.0 / nyq], btype='band')
        self.filter_b = b
        self.filter_a = a
        self.zi_h = scipy_signal.lfilter_zi(b, a) * 512
        self.zi_v = scipy_signal.lfilter_zi(b, a) * 512

        # --- 状态变量 (关键修改) ---
        self.cooldown = 0
        self.current_energy = 0
        
        # 记录上一次检测到的动作名称，用于在红框滑动时持续显示
        self.last_detected_label = "Rest" 
        
        # 计数器：距离上一次检测动作过去了多少个数据点
        # 初始化为一个大数字，表示很久没有动作
        self.frames_since_detection = 9999 

    def extract_features(self, window):
        features = []
        for axis in range(2): 
            sig = window[:, axis]
            diff = np.diff(sig)
            features.append(np.std(sig))
            features.append(np.max(sig) - np.min(sig))
            features.append(np.mean(np.abs(diff)))
            features.append(np.max(np.abs(diff)))
            features.append(skew(sig))
            features.append(kurtosis(sig))
            features.append(np.sum(sig ** 2))
            features.append(np.sqrt(np.mean(sig ** 2)))
            centered = sig - np.mean(sig)
            zcr = np.sum(np.diff(np.sign(centered)) != 0) / len(sig)
            features.append(zcr)
            fft_vals = np.abs(rfft(sig))
            freqs = rfftfreq(len(sig), d=1.0 / SAMPLING_RATE)
            features.append(np.max(fft_vals[1:]) if len(fft_vals) > 1 else 0.0)
            if len(fft_vals) > 1 and np.sum(fft_vals[1:]) > 0:
                features.append(np.sum(freqs[1:] * fft_vals[1:]) / np.sum(fft_vals[1:]))
            else:
                features.append(0.0)
        return features

    def process_frame(self, h_val, v_val):
        # 1. 每次有新数据进来，计数器+1 (代表波形整体左移了一格)
        self.frames_since_detection += 1
        
        # 2. 实时滤波
        filt_h, self.zi_h = scipy_signal.lfilter(self.filter_b, self.filter_a, [h_val], zi=self.zi_h)
        filt_v, self.zi_v = scipy_signal.lfilter(self.filter_b, self.filter_a, [v_val], zi=self.zi_v)
        
        self.proc_buffer_h.append(filt_h[0])
        self.proc_buffer_v.append(filt_v[0])
        
        energy = 0
        
        if len(self.proc_buffer_h) >= WINDOW_SIZE:
            win_h = np.array(list(self.proc_buffer_h)[-WINDOW_SIZE:])
            win_v = np.array(list(self.proc_buffer_v)[-WINDOW_SIZE:])
            window_data = np.column_stack((win_h, win_v))
            
            center_h = win_h - np.mean(win_h)
            center_v = win_v - np.mean(win_v)
            mid_idx = HALF_WIN
            energy = (center_h[mid_idx]**2 + center_v[mid_idx]**2)
        
        self.current_energy = energy
        self.energy_history.append(energy)

        # 3. 冷却与触发
        if self.cooldown > 0:
            self.cooldown -= 1
            return None

        if energy > ENERGY_THRESHOLD:
            feats = np.array([self.extract_features(window_data)])
            pred = self.model.predict(feats)[0]
            label = CLASSES[pred]
            
            print(f"!!! 检测动作: {label} (能量: {energy:.0f}) !!!")
            
            self.cooldown = COOLDOWN_FRAMES
            
            # --- 关键修改：重置检测计数器 ---
            self.last_detected_label = label
            self.frames_since_detection = 0 # 刚刚发生，距离现在0帧
            
            return label
        
        return None

# ================= 线程任务 =================

def serial_thread(system):
    print(f"正在连接串口 {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("串口已连接！等待数据...")
    except Exception as e:
        print(f"[错误] 无法打开串口: {e}")
        system.running = False
        return

    while system.running:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line: continue
            
            parts = line.split('\t')
            if len(parts) == 2:
                h_str = parts[0].split(':')
                v_str = parts[1].split(':')
                
                if len(h_str) == 2 and len(v_str) == 2:
                    h_val = float(h_str[1])
                    v_val = float(v_str[1])
                    
                    with system.lock:
                        system.raw_h.append(h_val)
                        system.raw_v.append(v_val)
                        system.process_frame(h_val, v_val)
        except ValueError:
            pass
        except Exception as e:
            print(f"串口读取错误: {e}")
            break
            
    ser.close()
    print("串口线程结束")

# ================= 主程序 (绘图) =================

def main():
    system = EOGSystem(MODEL_PATH)
    
    t = threading.Thread(target=serial_thread, args=(system,), daemon=True)
    t.start()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.2)
    
    # --- 子图1 ---
    line_h, = ax1.plot([], [], label='Horizontal', color='#1f77b4', lw=1.5)
    line_v, = ax1.plot([], [], label='Vertical', color='#ff7f0e', lw=1.5)
    
    text_status = ax1.text(0.02, 0.90, "State: Rest", transform=ax1.transAxes, 
                          fontsize=14, fontweight='bold', color='gray')
    
    rect = plt.Rectangle((0, 0), WINDOW_SIZE, 0, color='red', alpha=0.3, visible=False)
    ax1.add_patch(rect)
    
    ax1.set_ylim(0, 1024)
    ax1.set_ylabel("Raw ADC")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title("EOG Signal & Detection")

    # --- 子图2 ---
    line_energy, = ax2.plot([], [], label='Energy', color='black', lw=1)
    ax2.axhline(y=ENERGY_THRESHOLD, color='red', linestyle='--', alpha=0.8, label='Threshold')
    text_energy = ax2.text(0.02, 0.85, "Energy: 0", transform=ax2.transAxes,
                           fontsize=12, color='black', fontweight='bold')

    ax2.set_xlim(0, DISPLAY_LEN)
    ax2.set_ylim(0, ENERGY_THRESHOLD * 2) 
    ax2.set_ylabel("Signal Energy")
    ax2.set_xlabel("Time (frames)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    x_data = np.arange(DISPLAY_LEN)

    def update(frame):
        with system.lock:
            # 1. 更新波形
            line_h.set_data(x_data, list(system.raw_h))
            line_v.set_data(x_data, list(system.raw_v))
            
            # 2. 更新能量
            current_energy_list = list(system.energy_history)
            line_energy.set_data(x_data, current_energy_list)
            text_energy.set_text(f"Energy: {system.current_energy:.0f}")
            
            # ==========================================
            # 3. 核心修复：基于数据计数器计算红框位置
            # ==========================================
            
            # 计算红框应该在的位置 (从最右边减去过去了多少帧)
            # DISPLAY_LEN (300) - WINDOW_SIZE (50) - 过去了多少帧
            frames_ago = system.frames_since_detection
            rect_x = DISPLAY_LEN - WINDOW_SIZE - frames_ago
            
            # 判断红框是否还在屏幕内 (-WINDOW_SIZE 是因为它完全滑出左边界的位置)
            if rect_x > -WINDOW_SIZE:
                # --- 状态：显示动作 ---
                rect.set_x(rect_x)
                rect.set_height(1024)
                rect.set_visible(True)
                
                label = system.last_detected_label
                color = 'red' if label == 'Blink' else 'green'
                text_status.set_text(f"Detected: {label}")
                text_status.set_color(color)
            else:
                # --- 状态：Rest ---
                rect.set_visible(False)
                
                # 只有当红框看不见时，才强制变回 Rest
                text_status.set_text("State: Rest")
                text_status.set_color("#555555")

        return line_h, line_v, text_status, rect, line_energy, text_energy

    ani = animation.FuncAnimation(fig, update, interval=20, blit=False)
    plt.show()
    
    system.running = False
    t.join()

if __name__ == "__main__":
    main()