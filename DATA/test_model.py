"""
EOG 模型测试脚本
================
用保存的模型对 CSV 文件进行分类预测。

使用方法：
    # 测试单个文件
    python test_model.py path/to/file.csv

    # 测试整个目录（自动从文件名提取标签）
    python test_model.py path/to/directory/

    # 测试 IMU_EOG/output 目录
    python test_model.py ../IMU_EOG/output/
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from numpy.fft import rfft, rfftfreq
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]
SAMPLING_RATE = 50

# 文件名前缀 → 类别（用于自动标注计算准确率）
LABEL_MAP = {
    "blink": "Blink", "down": "Down", "left": "Left",
    "rest": "Rest", "right": "Right", "up": "Up",
}


def apply_filter(data, lowcut=0.5, highcut=10.0, fs=50, order=4):
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    if padlen < 1:
        return data
    return scipy_signal.filtfilt(b, a, data, axis=0, padlen=padlen)


def extract_features(window):
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


def load_csv(fpath):
    df = pd.read_csv(fpath)
    if 'H' in df.columns and 'V' in df.columns:
        return df[['H', 'V']].values
    elif 'EOG_H' in df.columns and 'EOG_V' in df.columns:
        return df[['EOG_H', 'EOG_V']].values
    elif 'data 0' in df.columns and 'data 1' in df.columns:
        return df[['data 0', 'data 1']].values
    else:
        print(f"  [跳过] {fpath}: 未找到可用的 H/V 列 (列名: {list(df.columns)})")
        return None


def predict_file(model, fpath):
    data = load_csv(fpath)
    if data is None:
        return None
    try:
        filtered = apply_filter(data)
    except Exception:
        filtered = data
    features = np.array([extract_features(filtered)])
    pred = model.predict(features)[0]
    return CLASSES[pred]


def get_label_from_filename(fname):
    fname_lower = fname.lower()
    for prefix, cls in LABEL_MAP.items():
        if fname_lower.startswith(prefix):
            return cls
    return None


def main():
    if len(sys.argv) < 2:
        print("用法: python test_model.py <csv文件或目录>")
        print("示例: python test_model.py ../IMU_EOG/output/")
        print("      python test_model.py ../IMU_EOG/output/blink01.csv")
        sys.exit(1)

    # 加载模型
    rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    if not os.path.exists(rf_path):
        print(f"[错误] 模型文件不存在: {rf_path}")
        print("请先运行 python generate_and_train_synthetic.py 生成模型。")
        sys.exit(1)

    rf = joblib.load(rf_path)
    print(f"已加载模型: {rf_path}\n")

    target = sys.argv[1]

    # 单文件测试
    if os.path.isfile(target):
        result = predict_file(rf, target)
        if result:
            print(f"  {os.path.basename(target)} -> 预测: {result}")
        return

    # 目录测试
    if not os.path.isdir(target):
        print(f"[错误] 路径不存在: {target}")
        sys.exit(1)

    csv_files = sorted(f for f in os.listdir(target) if f.endswith('.csv'))
    if not csv_files:
        print(f"目录中没有 CSV 文件: {target}")
        return

    correct = 0
    total_labeled = 0
    total = 0
    results = []

    for fname in csv_files:
        fpath = os.path.join(target, fname)
        pred = predict_file(rf, fpath)
        if pred is None:
            continue
        total += 1
        true_label = get_label_from_filename(fname)
        if true_label:
            total_labeled += 1
            is_correct = (pred == true_label)
            correct += is_correct
            mark = "OK" if is_correct else "XX"
            results.append((fname, true_label, pred, mark))
        else:
            results.append((fname, "?", pred, "--"))

    # 输出结果
    print(f"{'文件名':<35} {'真实':>6} {'预测':>6} {'结果':>4}")
    print("-" * 55)
    for fname, true_label, pred, mark in results:
        print(f"  {fname:<33} {true_label:>6} {pred:>6} {mark:>4}")

    print(f"\n共 {total} 个文件")
    if total_labeled > 0:
        print(f"有标签: {total_labeled} 个, 正确: {correct} 个, 准确率: {correct/total_labeled:.2%}")


if __name__ == "__main__":
    main()
