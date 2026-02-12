"""
合成 EOG 数据生成 + 单样本训练实验
====================================
目的：验证"每个 CSV 只含一次动作、整个 CSV 作为一个样本"的训练方式
      是否能有效区分 Left/Right/Up/Down/Rest/Blink

波形参考 IMU_EOG/demo case/ 中的真实采集数据，
使用 sigmoid 阶跃组合模拟真实 EOG 的非对称扫视波形。

生成的数据保存在 synthetic_EOG/ 目录，不影响原有 EOG_data/

使用方法：
    cd DATA
    python generate_and_train_synthetic.py
"""

import os
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from numpy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
SYNTHETIC_DIR = "synthetic_EOG"
CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]
SAMPLING_RATE = 50
WINDOW_SIZE = 150         # 每个 CSV = 150 个采样点 (3 秒)
BASELINE = 500.0          # H/V 通道基线（中心值，实际每样本会偏移）
BASELINE_JITTER = 15.0    # 基线偏移范围 ±15（每样本不同）
QUIET_NOISE = 25.0        # 静止段噪声幅度（基线 ±25）
SENSOR_NOISE = 5.0        # 全局传感器噪声 std
SAMPLES_PER_CLASS = 120   # 每类生成 120 个样本
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ============================================================
# 模块 1：合成数据生成
# ============================================================
def quiet_baseline(n_samples, baseline):
    """
    生成静止段信号：基线 + 随机波动（±QUIET_NOISE 以内）。
    使用低频随机游走模拟真实静止时的缓慢漂移，加高频噪声。
    """
    # 低频漂移（随机游走），std 更大让漂移更明显
    drift = np.cumsum(np.random.normal(0, 2.5, n_samples))
    drift = np.clip(drift, -QUIET_NOISE, QUIET_NOISE)
    # 中频波动（模拟肌电干扰等）
    mid_freq = np.random.normal(0, 4.0, n_samples)
    # 偶尔加入小毛刺（5% 概率出现幅度 10-20 的尖峰）
    spikes = np.zeros(n_samples)
    spike_mask = np.random.random(n_samples) < 0.05
    spikes[spike_mask] = np.random.uniform(-18, 18, np.sum(spike_mask))
    return baseline + drift + mid_freq + spikes


def generate_action_wave(n_action_samples, amplitude):
    """
    在动作窗口内生成一个带扰动的正弦周期波形。

    基本波形：0 → +amplitude → 0 → -amplitude → 0
    增加扰动：相位偏移、非对称包络、幅度调制、时间扭曲、噪声。
    """
    t = np.linspace(0, 2 * np.pi, n_action_samples)

    # 时间扭曲：让前半/后半周期速度不同（模拟真实扫视的非对称性）
    warp_strength = np.random.uniform(-0.3, 0.3)
    t_warped = t + warp_strength * np.sin(t)

    # 非对称包络：正弦包络加随机指数偏移，让峰值位置不在正中间
    peak_shift = np.random.uniform(-0.15, 0.15)
    envelope = np.sin(np.clip(t / 2 + peak_shift, 0, np.pi / 2 * 2)) ** 2
    # 随机让包络更尖或更平
    envelope_power = np.random.uniform(1.5, 2.5)
    envelope = envelope ** (envelope_power / 2.0)

    # 相位扰动（更大范围）
    phase = np.random.uniform(-0.5, 0.5)

    # 正负半周期幅度不对称（真实 EOG：初始扫视小，回弹/过冲大）
    # 缩小正半周期（第一个峰），保留负半周期（回弹）的完整幅度
    asym_ratio = np.random.uniform(0.4, 0.8)
    base_wave = np.sin(t_warped + phase)
    base_wave[base_wave > 0] *= asym_ratio

    sig = amplitude * base_wave * envelope

    # 加一层动作段内的高频噪声
    sig += np.random.normal(0, abs(amplitude) * 0.04, n_action_samples)

    return sig


def generate_blink_wave(n_action_samples):
    """
    生成眨眼波形（~0.5s 动作窗口内）。

    眨眼特征：快速正向尖峰 + 负向回弹，非对称。
    H 和 V 通道都有大幅波动，H 幅度更大。
    增加扰动：幅度范围更宽、形状更随机。
    返回 (h_wave, v_wave)。
    """
    t = np.linspace(0, 2 * np.pi, n_action_samples)

    # 时间扭曲
    warp = np.random.uniform(-0.25, 0.25)
    t_warped = t + warp * np.sin(t)

    # 非对称包络
    peak_shift = np.random.uniform(-0.2, 0.2)
    envelope = np.sin(np.clip(t / 2 + peak_shift, 0, np.pi / 2 * 2)) ** 2
    env_power = np.random.uniform(1.3, 2.8)
    envelope = envelope ** (env_power / 2.0)

    phase = np.random.uniform(-0.3, 0.3)

    # H 通道：+350~600, -120~300（范围更宽）
    amp_h_pos = np.random.uniform(350, 600)
    amp_h_neg = np.random.uniform(120, 300)
    asym_h = np.random.uniform(0.5, 1.0)
    h_base = np.sin(t_warped + phase)
    h_pos_part = np.where(h_base > 0, h_base, 0)
    h_neg_part = np.where(h_base < 0, h_base, 0)
    h_wave = (amp_h_pos * h_pos_part + amp_h_neg * abs(asym_h) * h_neg_part) * envelope
    h_wave += np.random.normal(0, 12, n_action_samples)

    # V 通道：幅度稍小，独立随机
    amp_v_pos = np.random.uniform(250, 500)
    amp_v_neg = np.random.uniform(80, 220)
    phase_v = phase + np.random.uniform(-0.15, 0.15)  # V 通道相位略有偏差
    asym_v = np.random.uniform(0.5, 1.0)
    v_base = np.sin(t_warped + phase_v)
    v_pos_part = np.where(v_base > 0, v_base, 0)
    v_neg_part = np.where(v_base < 0, v_base, 0)
    v_wave = (amp_v_pos * v_pos_part + amp_v_neg * abs(asym_v) * v_neg_part) * envelope
    v_wave += np.random.normal(0, 10, n_action_samples)

    return h_wave, v_wave


def generate_one_sample(class_name):
    """
    为指定类别生成一个 (WINDOW_SIZE, 2) 的合成信号（3 秒 = 150 采样点）。

    结构：[静止段] + [动作段] + [静止段]
    每个样本的基线、幅度、时间位置、波形形状都有较大随机扰动。

    列 0 = H 通道，列 1 = V 通道
    """
    n = WINDOW_SIZE  # 150

    # 每个样本的基线有随机偏移（模拟不同佩戴位置/个体差异）
    bl_h = BASELINE + np.random.uniform(-BASELINE_JITTER, BASELINE_JITTER)
    bl_v = BASELINE + np.random.uniform(-BASELINE_JITTER, BASELINE_JITTER)

    if class_name == "Rest":
        h_sig = quiet_baseline(n, bl_h)
        v_sig = quiet_baseline(n, bl_v)
        return np.column_stack([h_sig, v_sig])

    # --- 非静止类别：构造 [静止] + [动作] + [静止] ---

    if class_name == "Blink":
        # 眨眼：动作 0.3~0.7s（更宽范围）
        action_dur = int(np.random.uniform(0.3, 0.7) * SAMPLING_RATE)
    else:
        # 上下左右看：动作 1.0~2.0s（更宽范围）
        action_dur = int(np.random.uniform(1.0, 2.0) * SAMPLING_RATE)

    # 动作开始位置：0.2~1.0s（更宽范围）
    action_start_sec = np.random.uniform(0.2, 1.0)
    action_start = int(action_start_sec * SAMPLING_RATE)
    if action_start + action_dur > n:
        action_start = max(0, n - action_dur)
    action_end = action_start + action_dur

    # 初始化为静止基线
    h_sig = quiet_baseline(n, bl_h)
    v_sig = quiet_baseline(n, bl_v)

    # 动作幅度（更宽范围：100~250）
    amp = np.random.uniform(100, 250)

    # 串扰幅度也更随机
    crosstalk_small = np.random.uniform(5, 40)
    crosstalk_medium = np.random.uniform(15, 60)

    if class_name == "Right":
        # 右看：H 先正后负（初始扫视小，回弹大）
        h_sig[action_start:action_end] = bl_h + generate_action_wave(action_dur, amp)
        v_sig[action_start:action_end] = bl_v + generate_action_wave(action_dur, np.random.choice([-1, 1]) * crosstalk_small)

    elif class_name == "Left":
        # 左看：H 先负后正（初始扫视小，过冲大）
        h_sig[action_start:action_end] = bl_h + generate_action_wave(action_dur, -amp)
        v_sig[action_start:action_end] = bl_v + generate_action_wave(action_dur, np.random.choice([-1, 1]) * crosstalk_small)

    elif class_name == "Up":
        # 上看：V 先正后负
        v_sig[action_start:action_end] = bl_v + generate_action_wave(action_dur, amp)
        h_sig[action_start:action_end] = bl_h + generate_action_wave(action_dur, np.random.choice([-1, 1]) * crosstalk_small)

    elif class_name == "Down":
        # 下看：V 先负后正
        v_sig[action_start:action_end] = bl_v + generate_action_wave(action_dur, -amp)
        h_sig[action_start:action_end] = bl_h + generate_action_wave(action_dur, crosstalk_medium)

    elif class_name == "Blink":
        blink_h, blink_v = generate_blink_wave(action_dur)
        h_sig[action_start:action_end] = bl_h + blink_h
        v_sig[action_start:action_end] = bl_v + blink_v

    # 全局传感器噪声（更大）
    h_sig += np.random.normal(0, SENSOR_NOISE, n)
    v_sig += np.random.normal(0, SENSOR_NOISE, n)

    return np.column_stack([h_sig, v_sig])


def generate_all_data():
    """生成所有类别的合成数据，保存为 CSV 文件。"""
    print("=" * 60)
    print("步骤 1：生成合成 EOG 数据")
    print("=" * 60)

    total_files = 0
    for class_name in CLASSES:
        class_dir = os.path.join(SYNTHETIC_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(SAMPLES_PER_CLASS):
            data = generate_one_sample(class_name)
            # 生成时间轴
            times = np.arange(WINDOW_SIZE) / SAMPLING_RATE

            df = pd.DataFrame({
                'Program Time [s]': times,
                'data 0': data[:, 0],
                'data 1': data[:, 1],
            })
            fname = f"synthetic_{class_name}_{i+1:03d}.csv"
            df.to_csv(os.path.join(class_dir, fname), index=False)
            total_files += 1

    print(f"  已生成 {total_files} 个 CSV 文件")
    print(f"  每类 {SAMPLES_PER_CLASS} 个样本")
    print(f"  保存目录: {SYNTHETIC_DIR}/")
    for cls in CLASSES:
        print(f"    {cls}/: {SAMPLES_PER_CLASS} 个文件")


# ============================================================
# 模块 2：数据加载（每个 CSV = 一个样本）
# ============================================================
def load_synthetic_data():
    """加载合成数据，每个 CSV 整体作为一个样本。"""
    print(f"\n{'=' * 60}")
    print("步骤 2：加载合成数据（每 CSV = 一个样本）")
    print("=" * 60)

    all_data = []
    all_labels = []

    for label_id, class_name in enumerate(CLASSES):
        class_dir = os.path.join(SYNTHETIC_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"  [警告] {class_dir} 不存在，跳过")
            continue

        csv_files = sorted(f for f in os.listdir(class_dir) if f.endswith('.csv'))
        for fname in csv_files:
            fpath = os.path.join(class_dir, fname)
            df = pd.read_csv(fpath)
            data = df[['data 0', 'data 1']].values
            all_data.append(data)
            all_labels.append(label_id)

    print(f"  已加载 {len(all_data)} 个样本")
    counts = Counter(all_labels)
    for cls_id in range(len(CLASSES)):
        print(f"    {CLASSES[cls_id]:<8}: {counts.get(cls_id, 0)} 个样本")

    return all_data, np.array(all_labels)


# ============================================================
# 模块 3：特征提取（与 train_v2 一致的 22 个特征）
# ============================================================
def apply_filter(data, lowcut=0.5, highcut=10.0, fs=50, order=4):
    """带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    # 数据太短时用 padlen 参数避免报错
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    if padlen < 1:
        return data
    return scipy_signal.filtfilt(b, a, data, axis=0, padlen=padlen)


def extract_features_v2(window):
    """
    从 (N, 2) 的窗口中提取 22 个特征。
    与 train_v2.py 完全一致。
    """
    features = []
    for axis in range(2):  # 0=H, 1=V
        sig = window[:, axis]
        diff = np.diff(sig)

        # 原有 7 个特征
        features.append(np.std(sig))
        features.append(np.max(sig) - np.min(sig))
        features.append(np.mean(np.abs(diff)))
        features.append(np.max(np.abs(diff)))
        features.append(skew(sig))
        features.append(kurtosis(sig))
        features.append(np.sum(sig ** 2))

        # 新增 4 个特征
        features.append(np.sqrt(np.mean(sig ** 2)))

        centered = sig - np.mean(sig)
        zcr = np.sum(np.diff(np.sign(centered)) != 0) / len(sig)
        features.append(zcr)

        fft_vals = np.abs(rfft(sig))
        freqs = rfftfreq(len(sig), d=1.0 / SAMPLING_RATE)
        if len(fft_vals) > 1:
            features.append(np.max(fft_vals[1:]))
        else:
            features.append(0.0)

        if len(fft_vals) > 1 and np.sum(fft_vals[1:]) > 0:
            features.append(
                np.sum(freqs[1:] * fft_vals[1:]) / np.sum(fft_vals[1:])
            )
        else:
            features.append(0.0)

    return features


def build_features(all_data, all_labels):
    """对每个样本（整个 CSV）提取特征，不做滑动窗口。"""
    print(f"\n{'=' * 60}")
    print("步骤 3：特征提取（每个 CSV 整体提取 22 个特征）")
    print("=" * 60)

    X = []
    y = []

    for i in range(len(all_data)):
        data = all_data[i]

        # 带通滤波
        try:
            filtered = apply_filter(data)
        except Exception:
            filtered = data

        # 整个 CSV 作为一个窗口提取特征
        features = extract_features_v2(filtered)
        X.append(features)
        y.append(all_labels[i])

    X = np.array(X)
    y = np.array(y)
    print(f"  特征矩阵: {X.shape}")
    print(f"  标签数: {len(y)}")
    return X, y


# ============================================================
# 模块 4：训练与评估
# ============================================================
def train_and_evaluate(X, y):
    """训练多个模型并评估。"""
    print(f"\n{'=' * 60}")
    print("步骤 4：训练与评估")
    print("=" * 60)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 归一化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 模型 1: RandomForest ---
    print(f"\n  [1/2] 训练 RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"    测试准确率: {rf_acc:.4f}")
    print(f"    5 折交叉验证: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")

    # --- 模型 2: GradientBoosting ---
    print(f"\n  [2/2] 训练 GradientBoosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)

    gb_cv = cross_val_score(gb, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"    测试准确率: {gb_acc:.4f}")
    print(f"    5 折交叉验证: {gb_cv.mean():.4f} +/- {gb_cv.std():.4f}")

    # 选择更好的模型输出详细报告
    if rf_acc >= gb_acc:
        best_name, best_pred = "RandomForest", rf_pred
    else:
        best_name, best_pred = "GradientBoosting", gb_pred

    print(f"\n{'=' * 60}")
    print(f"详细评估: {best_name}")
    print("=" * 60)

    print("\n--- 分类报告 ---")
    print(classification_report(y_test, best_pred, target_names=CLASSES))

    print("--- 混淆矩阵 ---")
    cm = confusion_matrix(y_test, best_pred)
    # 文本形式打印混淆矩阵
    header = f"{'':>8}" + "".join(f"{c:>8}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{CLASSES[i]:>8}" + "".join(f"{v:>8}" for v in row)
        print(row_str)

    # Left vs Right 混淆分析
    print(f"\n--- Left/Right 区分分析 ---")
    left_idx = CLASSES.index("Left")
    right_idx = CLASSES.index("Right")

    left_mask = (y_test == left_idx)
    right_mask = (y_test == right_idx)

    left_as_right = np.sum(best_pred[left_mask] == right_idx)
    right_as_left = np.sum(best_pred[right_mask] == left_idx)
    left_correct = np.sum(best_pred[left_mask] == left_idx)
    right_correct = np.sum(best_pred[right_mask] == right_idx)

    print(f"  Left 样本数: {np.sum(left_mask)}")
    print(f"    正确识别为 Left: {left_correct}")
    print(f"    误判为 Right:    {left_as_right}")
    print(f"  Right 样本数: {np.sum(right_mask)}")
    print(f"    正确识别为 Right: {right_correct}")
    print(f"    误判为 Left:     {right_as_left}")

    # 特征重要性（RF）
    print(f"\n--- 特征重要性 (RandomForest) ---")
    feature_names = []
    for ch in ['H', 'V']:
        feature_names += [
            f'{ch}_std', f'{ch}_p2p', f'{ch}_mean_vel', f'{ch}_max_vel',
            f'{ch}_skew', f'{ch}_kurt', f'{ch}_energy',
            f'{ch}_rms', f'{ch}_zcr', f'{ch}_fft_peak', f'{ch}_spectral_centroid',
        ]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"  前 10 个重要特征:")
    for rank, idx in enumerate(indices[:10]):
        print(f"    {rank+1}. {feature_names[idx]:<25} {importances[idx]:.4f}")

    return rf_acc, gb_acc, rf, gb, scaler


# ============================================================
# 模块 5：用真实采集数据测试
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "IMU_EOG", "output")
# 文件名前缀 → 类别映射
REAL_LABEL_MAP = {
    "blink": "Blink",
    "down": "Down",
    "left": "Left",
    "rest": "Rest",
    "right": "Right",
    "up": "Up",
}


def load_real_data():
    """
    从 IMU_EOG/output/ 加载带标签的真实采集数据。
    只加载文件名以 blink/down/left/rest/right/up 开头的 CSV。
    """
    print(f"\n{'=' * 60}")
    print("步骤 5：加载真实采集数据（IMU_EOG/output/）")
    print("=" * 60)

    all_data = []
    all_labels = []
    file_names = []

    if not os.path.exists(REAL_DATA_DIR):
        print(f"  [错误] 目录不存在: {REAL_DATA_DIR}")
        return [], np.array([]), []

    csv_files = sorted(f for f in os.listdir(REAL_DATA_DIR) if f.endswith('.csv'))

    for fname in csv_files:
        prefix = None
        for key in REAL_LABEL_MAP:
            if fname.lower().startswith(key):
                prefix = key
                break
        if prefix is None:
            continue  # 跳过无标签文件（如 eog_capture_*）

        class_name = REAL_LABEL_MAP[prefix]
        label_id = CLASSES.index(class_name)

        fpath = os.path.join(REAL_DATA_DIR, fname)
        df = pd.read_csv(fpath)

        if 'H' in df.columns and 'V' in df.columns:
            data = df[['H', 'V']].values
        elif 'EOG_H' in df.columns and 'EOG_V' in df.columns:
            data = df[['EOG_H', 'EOG_V']].values
        else:
            print(f"  [跳过] {fname}: 未找到 H/V 列")
            continue

        all_data.append(data)
        all_labels.append(label_id)
        file_names.append(fname)

    print(f"  已加载 {len(all_data)} 个真实样本")
    counts = Counter(all_labels)
    for cls_id in range(len(CLASSES)):
        print(f"    {CLASSES[cls_id]:<8}: {counts.get(cls_id, 0)} 个样本")

    return all_data, np.array(all_labels), file_names


def test_on_real_data(rf, gb, scaler):
    """用训练好的模型在真实数据上测试。"""
    all_data, all_labels, file_names = load_real_data()
    if len(all_data) == 0:
        print("  没有可用的真实数据，跳过测试。")
        return

    # 提取特征
    print(f"\n  提取特征...")
    X_real = []
    valid_indices = []
    for i in range(len(all_data)):
        data = all_data[i]
        try:
            filtered = apply_filter(data)
        except Exception:
            filtered = data
        features = extract_features_v2(filtered)
        X_real.append(features)
        valid_indices.append(i)

    X_real = np.array(X_real)
    y_real = all_labels[valid_indices]
    valid_files = [file_names[i] for i in valid_indices]

    X_real_scaled = scaler.transform(X_real)

    print(f"\n{'=' * 60}")
    print("步骤 6：真实数据测试结果")
    print("=" * 60)

    # RF 用未归一化特征（训练时也是）
    rf_pred = rf.predict(X_real)
    rf_acc = accuracy_score(y_real, rf_pred)
    print(f"\n  [RandomForest] 真实数据准确率: {rf_acc:.4f} ({np.sum(rf_pred == y_real)}/{len(y_real)})")

    gb_pred = gb.predict(X_real)
    gb_acc = accuracy_score(y_real, gb_pred)
    print(f"  [GradientBoosting] 真实数据准确率: {gb_acc:.4f} ({np.sum(gb_pred == y_real)}/{len(y_real)})")

    if rf_acc >= gb_acc:
        best_name, best_pred = "RandomForest", rf_pred
    else:
        best_name, best_pred = "GradientBoosting", gb_pred

    print(f"\n  最佳模型: {best_name}")
    print(f"\n--- 分类报告（真实数据）---")
    print(classification_report(y_real, best_pred, target_names=CLASSES))

    print("--- 混淆矩阵（真实数据）---")
    cm = confusion_matrix(y_real, best_pred, labels=range(len(CLASSES)))
    header = f"{'':>8}" + "".join(f"{c:>8}" for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{CLASSES[i]:>8}" + "".join(f"{v:>8}" for v in row)
        print(row_str)

    # 逐样本误判详情
    errors = [(valid_files[i], CLASSES[y_real[i]], CLASSES[best_pred[i]])
              for i in range(len(y_real)) if y_real[i] != best_pred[i]]
    if errors:
        print(f"\n--- 误判样本详情 ({len(errors)} 个) ---")
        for fname, true_cls, pred_cls in errors:
            print(f"  {fname:<35} 真实: {true_cls:<8} -> 预测: {pred_cls}")
    else:
        print(f"\n  全部正确！")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("合成 EOG 数据实验：单 CSV 单动作训练")
    print("=" * 60)
    print(f"  每类样本数: {SAMPLES_PER_CLASS}")
    print(f"  每样本长度: {WINDOW_SIZE} 点 ({WINDOW_SIZE/SAMPLING_RATE:.1f} 秒)")
    print(f"  类别: {CLASSES}")

    # 步骤 1：生成数据
    generate_all_data()

    # 步骤 2：加载数据
    all_data, all_labels = load_synthetic_data()

    # 步骤 3：提取特征
    X, y = build_features(all_data, all_labels)

    # 步骤 4：训练与评估
    rf_acc, gb_acc, rf, gb, scaler = train_and_evaluate(X, y)

    # 步骤 5-6：用真实数据测试
    test_on_real_data(rf, gb, scaler)

    # 保存模型
    MODEL_DIR = os.path.join(_SCRIPT_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(gb, os.path.join(MODEL_DIR, "gb_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\n  模型已保存到 {MODEL_DIR}/")
    print(f"    rf_model.pkl, gb_model.pkl, scaler.pkl")

    # 总结
    print(f"\n{'=' * 60}")
    print("实验总结")
    print("=" * 60)
    print(f"  数据方式: 每个 CSV 只含一次动作，整个 CSV 作为一个样本")
    print(f"  样本总数: {len(all_labels)} ({SAMPLES_PER_CLASS} x {len(CLASSES)} 类)")
    print(f"  特征维度: 22 (与 train_v2 一致)")
    print(f"  RandomForest 准确率:      {rf_acc:.4f}")
    print(f"  GradientBoosting 准确率:  {gb_acc:.4f}")
    print(f"\n  结论: 如果 Left/Right 区分良好，说明'单 CSV 单动作'")
    print(f"        的数据采集方式能有效避免回中窗口的标签污染问题。")


if __name__ == "__main__":
    main()
