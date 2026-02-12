import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from numpy.fft import rfft, rfftfreq

# ================= 配置区域 =================
# 模型路径 (请根据你的实际路径修改)
MODEL_PATH = "models/rf_model.pkl" 

# 信号参数
SAMPLING_RATE = 50
WINDOW_SIZE = 50  # 窗口大小 (例如 1秒)
HALF_WIN = WINDOW_SIZE // 2

# 阈值检测参数
THRESHOLD_FACTOR = 0.15  # 触发阈值系数 (0.1 ~ 0.3 通常较好)
                         # 解释: 只有超过 "最大能量 * 0.15" 的波峰才被视为动作
MIN_DISTANCE = 40        # 两个动作之间的最小距离 (点数)，避免对同一个眨眼重复切片

CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]

# ================= 核心工具函数 (复用你之前的逻辑) =================

def apply_filter(data, lowcut=0.5, highcut=10.0, fs=50, order=4):
    """ 使用 filtfilt 进行零相位滤波，适合离线分析和可视化 """
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    if padlen < 1: return data
    return scipy_signal.filtfilt(b, a, data, axis=0, padlen=padlen)

def extract_features(window):
    """ 提取特征 (保持与训练时一致) """
    features = []
    for axis in range(2): # H, V
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

def load_csv(fpath):
    df = pd.read_csv(fpath)
    # 尝试匹配常见的列名
    cols = df.columns
    if 'H' in cols and 'V' in cols: return df[['H', 'V']].values
    if 'EOG_H' in cols and 'EOG_V' in cols: return df[['EOG_H', 'EOG_V']].values
    if 'data 0' in cols and 'data 1' in cols: return df[['data 0', 'data 1']].values
    print(f"[错误] 无法识别列名: {cols}")
    return None

# ================= 新增：智能切分与预测逻辑 =================

def detect_and_predict(model, data):
    """
    基于能量阈值检测动作，并切分预测
    """
    predictions = []
    
    # 1. 预处理：滤波
    try:
        filtered = apply_filter(data)
    except:
        filtered = data
        
    # 2. 计算能量信号 (Energy)
    #    E = H^2 + V^2 (去中心化后)
    #    这能把所有方向的动作都转换成正值的波峰
    centered = filtered - np.mean(filtered, axis=0)
    energy = np.sum(centered**2, axis=1)
    
    # 3. 确定阈值
    #    这里使用动态阈值：取信号最大能量的 15% 作为触发线
    #    也可以使用 np.percentile(energy, 95)
    max_energy = np.max(energy)
    threshold = max_energy * THRESHOLD_FACTOR
    
    # 4. 寻找波峰 (find_peaks)
    #    height: 最小高度
    #    distance: 两个波峰间的最小距离 (防止同一次眨眼被切成两半)
    peaks, _ = scipy_signal.find_peaks(energy, height=threshold, distance=MIN_DISTANCE)
    
    print(f"检测到 {len(peaks)} 个潜在动作点 (阈值: {threshold:.2f})")

    # 5. 对每个波峰进行切片和预测
    valid_segments = [] # 用于绘图
    
    for peak_idx in peaks:
        # 定义窗口范围：以波峰为中心
        start = peak_idx - HALF_WIN
        end = peak_idx + HALF_WIN
        
        # 边界检查
        if start < 0 or end > len(filtered):
            continue
            
        segment = filtered[start:end]
        
        # 提取特征
        feats = np.array([extract_features(segment)])
        
        # 预测
        pred_idx = model.predict(feats)[0]
        label = CLASSES[pred_idx]
        
        # 保存结果：(中心点索引, 开始索引, 结束索引, 预测标签)
        predictions.append((peak_idx, label))
        valid_segments.append((start, end, label))
        
    return filtered, energy, threshold, valid_segments

# ================= 可视化绘图 =================

def visualize_results(filtered_data, energy, threshold, segments, filename):
    t = np.arange(len(filtered_data))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # --- 图1: 原始 EOG 信号 (H/V) ---
    ax1.plot(t, filtered_data[:, 0], label='Horizontal', color='#1f77b4', alpha=0.8)
    ax1.plot(t, filtered_data[:, 1], label='Vertical', color='#ff7f0e', alpha=0.8)
    ax1.set_ylabel('Amplitude (uV)')
    ax1.set_title(f'EOG Signal Analysis: {filename}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 绘制识别结果的背景色块
    # 颜色映射
    color_map = {
        "Rest": "gray", "Blink": "red", 
        "Left": "green", "Right": "purple", 
        "Up": "orange", "Down": "blue"
    }
    
    for start, end, label in segments:
        if label == "Rest": continue # 通常不标注 Rest，除非你想看误判
        
        # 在图上画出阴影区
        color = color_map.get(label, 'gray')
        rect = patches.Rectangle((start, np.min(filtered_data)), end-start, 
                                 np.max(filtered_data) - np.min(filtered_data),
                                 linewidth=0, edgecolor='none', facecolor=color, alpha=0.2)
        ax1.add_patch(rect)
        
        # 在中心位置写字
        center = (start + end) / 2
        ax1.text(center, np.max(filtered_data) * 0.9, label, 
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    # --- 图2: 能量与阈值 ---
    ax2.plot(t, energy, color='black', lw=1, label='Signal Energy')
    ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.1f})')
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Sample Index')
    ax2.set_title('Activity Detection (Energy-based)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 标记触发点
    for start, end, label in segments:
        peak = (start + end) // 2
        ax2.plot(peak, energy[peak], 'x', color='red')

    print("显示图表中... (关闭窗口以退出)")
    plt.show()

# ================= 主程序 =================

def main():
    if len(sys.argv) < 2:
        print("用法: python predict_continuous_viz.py <csv文件路径>")
        return

    csv_path = sys.argv[1]
    
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型文件: {MODEL_PATH}")
        print("请确认 models/rf_model.pkl 存在")
        return

    # 1. 加载模型
    print("加载模型中...")
    rf_model = joblib.load(MODEL_PATH)

    # 2. 加载数据
    data = load_csv(csv_path)
    if data is None: return

    # 3. 核心：检测与预测
    print(f"分析文件: {os.path.basename(csv_path)} ...")
    filtered, energy, thresh, segments = detect_and_predict(rf_model, data)

    # 4. 打印结果摘要
    print("\n=== 预测结果序列 ===")
    if not segments:
        print("未检测到任何动作 (能量未超过阈值)")
    else:
        for i, (s, e, lbl) in enumerate(segments):
            print(f"动作 {i+1}: 索引[{s}-{e}] -> {lbl}")

    # 5. 可视化
    visualize_results(filtered, energy, thresh, segments, os.path.basename(csv_path))

if __name__ == "__main__":
    main()