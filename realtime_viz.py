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
SERIAL_PORT = 'COM3'       # ⚠️ 请确认端口是否正确
BAUD_RATE = 115200
MODEL_PATH = "main_project/models/models_xkp/eog_model_v4.joblib"
SCALER_PATH = "main_project/models/models_xkp/eog_scaler_v4.joblib"


SAMPLING_RATE = 50
WINDOW_SIZE = 50           
DISPLAY_LEN = 300          
FILTER_BUFFER_SIZE = 150   

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

        # 绘图缓冲区 (GUI读取)
        self.plot_h = deque([0]*DISPLAY_LEN, maxlen=DISPLAY_LEN)
        self.plot_v = deque([0]*DISPLAY_LEN, maxlen=DISPLAY_LEN)
        
        # 算法缓冲区 (仅串口线程使用，不需要锁)
        self.raw_buffer = deque(maxlen=FILTER_BUFFER_SIZE)
        
        # 状态变量
        self.cooldown = 0
        self.last_pred = "Rest"
        self.last_conf = 0.0
        self.frame_counter = 0 
        self.detected_frame_idx = -999 

    def apply_realtime_filter(self, window_data):
        # 滤波器参数预计算 (为了提速，也可以放在init里)
        fs = 50.0 
        nyq = 0.5 * fs
        b, a = scipy_signal.butter(4, [0.5/nyq, 10.0/nyq], btype='band')
        
        # 沿轴0滤波
        return scipy_signal.filtfilt(b, a, window_data, axis=0)

    def extract_features(self, window_data):
        features = []
        for axis in range(2):
            sig = window_data[:, axis]
            sig = sig - np.mean(sig) # 去直流
            
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
            """
            这个函数在串口线程中运行
            """
            self.frame_counter += 1
            
            # 1. 存入算法缓冲区
            self.raw_buffer.append([h_val, v_val])
            
            if len(self.raw_buffer) < FILTER_BUFFER_SIZE:
                return

            # ==========================================
            # 耗时计算区域
            # ==========================================
            
            # A. 滤波
            long_window = np.array(self.raw_buffer)
            filtered_long = self.apply_realtime_filter(long_window)
            newest_sample = filtered_long[-1]
            
            # B. 准备特征
            final_window = filtered_long[-WINDOW_SIZE:] 
            feats = self.extract_features(final_window)
            
            # 获取垂直特征
            # Index 8: P2P (幅度)
            # Index 13: Max Velocity (速度)
            # 我们再加两个特征辅助极性判断:
            # Index 9: Mean (均值)
            # Index 10: Max (正峰值)
            # Index 11: Min (负峰值)
            v_velocity = feats[0][13]
            v_max = feats[0][10]
            v_min = feats[0][11]
            
            # C. 模型预测
            feats_scaled = self.scaler.transform(feats)
            probs = self.model.predict_proba(feats_scaled)[0]
            pred_idx = np.argmax(probs)
            raw_label = CLASSES[pred_idx] 
            conf = probs[pred_idx]

            final_label = "Rest"
            
            # 速度阈值 (仅用于 Up vs Blink)
            VELOCITY_THRESHOLD_HIGH = 42.0 
            VELOCITY_THRESHOLD_LOW  = 35.0

            if conf > CONFIDENCE_THRESHOLD and raw_label != "Rest":
                
                # --- 规则 1: 回弹抑制 (解决 Down 后半段变 Blink) ---
                frames_since_last = self.frame_counter - self.detected_frame_idx
                
                # 只要刚做完 Up/Down，0.6秒内的 Blink 全部屏蔽
                # (这里把时间稍微拉长到 30 帧，因为 Down 的回弹比较慢)
                if self.last_pred in ["Up", "Down"] and raw_label == "Blink" and frames_since_last < 30:
                    print(f"🛡️ 忽略回弹 Blink (上个动作: {self.last_pred})")
                    final_label = "Rest"
                
                else:
                    # --- 规则 2: Up vs Blink (同极性，靠速度区分) ---
                    if raw_label == "Up" and v_velocity < VELOCITY_THRESHOLD_LOW:
                        print(f"🐌 Up速度慢 ({v_velocity:.1f}) -> 改 Blink")
                        final_label = "Blink"
                        
                    elif raw_label == "Blink" and v_velocity > VELOCITY_THRESHOLD_HIGH:
                        # 只有当它真的是正波时才改 Up
                        if v_max > abs(v_min): 
                            print(f"🚀 Blink速度快 ({v_velocity:.1f}) -> 改 Up")
                            final_label = "Up"
                        else:
                            final_label = "Blink"
                    
                    # --- 规则 3: Down vs Blink (异极性，靠电压正负区分) ---
                    # Down 必须是负波 (负峰值比正峰值大)
                    # Blink 必须是正波 (正峰值比负峰值大)
                    
                    elif raw_label == "Down":
                        # 如果模型说是 Down，但波形主要是正的 -> 可能是 Blink
                        if v_max > abs(v_min) * 1.5: # 正峰是负峰的1.5倍以上
                            print(f"⚠️ Down 极性不符 (正波) -> 改 Blink")
                            final_label = "Blink"
                        else:
                            final_label = "Down" # 正常的 Down，不看速度！

                    elif raw_label == "Blink":
                        # 如果模型说是 Blink，但波形主要是负的 -> 可能是 Down
                        if abs(v_min) > v_max * 1.5:
                            print(f"⚠️ Blink 极性不符 (负波) -> 改 Down")
                            final_label = "Down"
                        else:
                            final_label = "Blink"
                            
                    # 其他情况 (Left, Right)
                    else:
                        final_label = raw_label

            # =========================================================

            if self.cooldown > 0:
                self.cooldown -= 1
                final_label = "Rest"

            with self.lock:
                self.plot_h.append(newest_sample[0])
                self.plot_v.append(newest_sample[1])
                
                if final_label != "Rest":
                    self.last_pred = final_label
                    self.last_conf = conf
                    self.cooldown = COOLDOWN_FRAMES
                    self.detected_frame_idx = self.frame_counter
                    
                    print(f"⚡ 最终: {final_label} (原:{raw_label} | Vel:{v_velocity:.1f} | Max:{v_max:.0f} Min:{v_min:.0f})")

# ================= 3. 串口线程 =================

def serial_thread(system):
    print(f"⏳ 正在连接串口 {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✅ 串口已连接！正在接收数据流...")
    except Exception as e:
        print(f"❌ 串口错误: {e}")
        system.running = False
        return

    while system.running:
        try:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                
                # 兼容逗号和制表符
                parts = line.replace(',', '\t').split('\t')
                
                if len(parts) >= 2:
                    try:
                        # ==== 修改开始: 去除标签 ====
                        # 你的 Arduino 发送的是 "H:512.00", 我们需要把 "H:" 删掉
                        str_h = parts[0].strip().replace("H:", "").replace("V:", "")
                        str_v = parts[1].strip().replace("H:", "").replace("V:", "")
                        
                        h_val = float(str_h) * GAIN_H
                        v_val = float(str_v) * GAIN_V
                        # ==== 修改结束 ====
                        
                        # 调用处理函数
                        system.process_new_data(h_val, v_val)
                            
                    except ValueError:
                        pass
            else:
                # 极其重要：给CPU一点喘息时间，防止单核跑满导致GUI卡顿
                time.sleep(0.001) 
                
        except Exception as e:
            print(f"⚠️ 线程错误: {e}")
            break
            
    if ser.is_open:
        ser.close()
    print("串口线程结束")

# ================= 4. 主程序 (GUI) =================

def main():
    system = EOGSystem(MODEL_PATH, SCALER_PATH)
    
    t = threading.Thread(target=serial_thread, args=(system,), daemon=True)
    t.start()
    
    # 设置图表风格
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'fast')
    
    # ⬇️⬇️⬇️ 修改开始：只创建一个子图 (1, 1) ⬇️⬇️⬇️
    fig, ax = plt.subplots(1, 1, figsize=(10, 6)) # 高度可以稍微改小一点，因为只有一行了
    # plt.subplots_adjust(hspace=0.2) # 不需要调整间距了，因为只有一个图
    
    # --- 在同一个 ax 上画两条线 ---
    # 1. 水平信号 (蓝色)
    line_h, = ax.plot([], [], label='Horizontal (L/R)', color='#1f77b4', lw=1.5, alpha=0.9)
    
    # 2. 垂直信号 (橙色)
    line_v, = ax.plot([], [], label='Vertical (U/D)', color='#ff7f0e', lw=1.5, alpha=0.8) # 稍微透明一点防止完全遮挡
    
    # 状态文字
    text_status = ax.text(0.02, 0.90, "Initializing...", transform=ax.transAxes, 
                          fontsize=16, fontweight='bold', color='gray')
    
    # 红框 (显示检测窗口)
    rect = plt.Rectangle((0, -200), WINDOW_SIZE, 400, color='red', alpha=0.15, visible=False)
    ax.add_patch(rect)
    
    # 设置坐标轴
    ax.set_xlim(0, DISPLAY_LEN)
    ax.set_ylim(-200, 200)       # ⚠️ 如果波形重叠太乱，可以把范围调大，比如 (-300, 300)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (Samples)")
    ax.set_title("Real-time EOG Analysis (Combined)")
    
    # 合并图例 (显示在右上角)
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    x_data = np.arange(DISPLAY_LEN)
    # ⬆️⬆️⬆️ 修改结束 ⬆️⬆️⬆️

    def update(frame):
        if not system.running: return
        
        # 快速获取锁，读取数据用于绘图
        with system.lock:
            data_h = list(system.plot_h)
            data_v = list(system.plot_v)
            
            current_frame = system.frame_counter
            detect_frame = system.detected_frame_idx
            last_pred = system.last_pred
            last_conf = system.last_conf
        
        # 1. 更新线条 (都在同一个图里更新)
        line_h.set_data(x_data, data_h)
        line_v.set_data(x_data, data_v)
        
        # 2. 更新红框和文字
        frames_ago = current_frame - detect_frame
        rect_x = DISPLAY_LEN - WINDOW_SIZE - frames_ago
        
        if rect_x > -WINDOW_SIZE and frames_ago >= 0:
            rect.set_x(rect_x)
            rect.set_visible(True)
            text_status.set_text(f"DETECTED: {last_pred} ({last_conf:.0%})")
            text_status.set_color('green')
        else:
            rect.set_visible(False)
            text_status.set_text("Monitoring...")
            text_status.set_color("gray")

        return line_h, line_v, text_status, rect

    # 启动动画
    ani = animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    
    system.running = False
    t.join()

if __name__ == "__main__":
    main()