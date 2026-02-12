"""
EOG 眼动信号分类 - 优化版训练脚本 v2
=====================================
相比 v1 的改进：
  - 22 个特征（原 14 个 + 8 个频域/RMS/过零率特征）
  - 子变体追踪（Hard/Soft, Fast/Std）
  - 多模型对比：RF, GradientBoosting, XGBoost, MLP, SVM, KNN, 投票集成
  - 超参数调优（RandomizedSearchCV）
  - 类别平衡（SMOTE / class_weight）
  - 按子变体评估 & 数据充足性分析

使用方法：
    cd DATA
    python train_v2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from numpy.fft import rfft, rfftfreq
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from collections import Counter
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# 可选依赖
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================
# 配置区域
# ============================================================
DATA_PATH = "EOG_data"
CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]
SAMPLING_RATE = 50       # 采样率 (Hz)
WINDOW_SIZE = 50         # 窗口大小（1秒）
STEP_SIZE = 10           # 滑动步长
CLEANING_THRESHOLD = 60  # P2P 低于此值 -> 强制标为 Rest
TEST_RATIO = 0.2
RANDOM_STATE = 42
N_FEATURES = 22          # 每通道 11 个特征 x 2 通道


# ============================================================
# 模块 2：数据加载（含子变体追踪）
# ============================================================
def parse_variant(filename, class_name):
    """从文件名解析子变体类型（Hard/Soft/Fast/Std等）"""
    name = filename.upper()
    if class_name == "Blink":
        if "HARD" in name:
            return "Hard"
        elif "SOFT" in name:
            return "Soft"
    else:
        if "BLINK" in name:
            return "WithBlink"
        elif "FAST" in name:
            return "Fast"
        elif "STD" in name:
            return "Std"
    return "Default"


def load_raw_data(data_path):
    """从 EOG_data/ 加载所有 CSV 文件，返回原始数组 + 元数据。"""
    raw_data = []
    labels = []
    filenames = []
    variants = []

    print(f"正在从 {data_path}/ 加载数据...")

    for label_name in CLASSES:
        folder = os.path.join(data_path, label_name)
        if not os.path.exists(folder):
            print(f"  [警告] {folder} 未找到，跳过")
            continue

        csv_files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
        for fname in csv_files:
            fpath = os.path.join(folder, fname)
            try:
                df = pd.read_csv(fpath, header=0, usecols=['data 0', 'data 1'])
                df.columns = ['EOG_H', 'EOG_V']
                df = df.apply(pd.to_numeric, errors='coerce').dropna()

                if len(df) < SAMPLING_RATE:
                    print(f"  [跳过] {fname}: 数据太短 ({len(df)} 行)")
                    continue

                raw_data.append(df.values)
                labels.append(CLASSES.index(label_name))
                filenames.append(fname)
                variants.append(parse_variant(fname, label_name))

            except Exception as e:
                print(f"  [错误] {fname}: {e}")

    print(f"  已加载 {len(raw_data)} 个文件")
    return raw_data, np.array(labels), filenames, variants


def print_data_stats(raw_data, labels, filenames, variants):
    """打印每类和每子变体的数据统计信息。"""
    print("\n" + "=" * 60)
    print("数据概览")
    print("=" * 60)

    rows = []
    for i in range(len(raw_data)):
        rows.append({
            'class': CLASSES[labels[i]],
            'variant': variants[i],
            'file': filenames[i],
            'n_rows': len(raw_data[i]),
        })
    df = pd.DataFrame(rows)

    # 每类汇总
    print("\n--- 每类汇总 ---")
    summary = df.groupby('class').agg(
        files=('file', 'count'),
        total_rows=('n_rows', 'sum'),
        avg_rows=('n_rows', lambda x: int(x.mean())),
    )
    print(summary.to_string())

    # 每子变体明细
    print("\n--- 每子变体明细 ---")
    variant_df = df.groupby(['class', 'variant']).agg(
        files=('file', 'count'),
        total_rows=('n_rows', 'sum'),
    )
    print(variant_df.to_string())
    return df


# ============================================================
# 模块 3：滤波与数据增强
# ============================================================
def apply_filter(data, lowcut=0.5, highcut=10.0, fs=50, order=4):
    """带通滤波器（必须与 realtime_viz.py 完全一致）"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, data, axis=0)


def augment_jitter(data, noise_std=2.0):
    """添加高斯噪声，模拟传感器波动。"""
    return data + np.random.normal(0, noise_std, data.shape)


def augment_scale(data, scale_range=(0.9, 1.1)):
    """随机幅度缩放。"""
    scale = np.random.uniform(*scale_range)
    return data * scale


def augment_data(raw_data, labels, filenames, variants, target_classes=None):
    """
    对少数类生成增强副本（加噪声 + 缩放）。
    target_classes: 需要增强的类别索引列表（如 [0, 5] 表示 Rest 和 Blink）
    """
    if target_classes is None:
        # 自动检测：增强文件数低于中位数的类别
        counts = Counter(labels)
        median_count = np.median(list(counts.values()))
        target_classes = [c for c, n in counts.items() if n < median_count]

    aug_data, aug_labels, aug_fnames, aug_variants = [], [], [], []
    n_aug = 0

    for i in range(len(raw_data)):
        if labels[i] in target_classes:
            # 每个文件生成 1 份增强副本
            aug = augment_jitter(augment_scale(raw_data[i].copy()))
            aug_data.append(aug)
            aug_labels.append(labels[i])
            aug_fnames.append(filenames[i] + "_aug")
            aug_variants.append(variants[i])
            n_aug += 1

    if n_aug > 0:
        classes_str = [CLASSES[c] for c in target_classes]
        print(f"  已增强 {n_aug} 个文件，类别: {classes_str}")

    all_data = list(raw_data) + aug_data
    all_labels = np.concatenate([labels, np.array(aug_labels)]) if n_aug > 0 else labels
    all_fnames = list(filenames) + aug_fnames
    all_variants = list(variants) + aug_variants

    return all_data, all_labels, all_fnames, all_variants


# ============================================================
# 模块 4：特征提取（22 个特征）
# ============================================================
FEATURE_NAMES = []
for _ch in ['H', 'V']:
    FEATURE_NAMES += [
        f'{_ch}_std', f'{_ch}_p2p', f'{_ch}_mean_vel', f'{_ch}_max_vel',
        f'{_ch}_skew', f'{_ch}_kurt', f'{_ch}_energy',
        f'{_ch}_rms', f'{_ch}_zcr', f'{_ch}_fft_peak', f'{_ch}_spectral_centroid',
    ]


def extract_features_v2(window):
    """
    从 (WINDOW_SIZE, 2) 的窗口中提取 22 个特征。
    前 14 个特征与 v1 完全一致，保证向后兼容。
    """
    features = []
    for axis in range(2):  # 0=水平, 1=垂直
        sig = window[:, axis]
        diff = np.diff(sig)

        # --- 原有 7 个特征（与 v1 一致）---
        features.append(np.std(sig))                        # 标准差
        features.append(np.max(sig) - np.min(sig))          # 峰峰值
        features.append(np.mean(np.abs(diff)))              # 平均速度
        features.append(np.max(np.abs(diff)))               # 最大速度
        features.append(skew(sig))                          # 偏度
        features.append(kurtosis(sig))                      # 峰度
        features.append(np.sum(sig ** 2))                   # 能量

        # --- 新增 4 个特征 ---
        features.append(np.sqrt(np.mean(sig ** 2)))         # 均方根 (RMS)

        # 过零率
        centered = sig - np.mean(sig)
        zcr = np.sum(np.diff(np.sign(centered)) != 0) / len(sig)
        features.append(zcr)

        # FFT 主频幅值
        fft_vals = np.abs(rfft(sig))
        freqs = rfftfreq(len(sig), d=1.0 / SAMPLING_RATE)
        if len(fft_vals) > 1:
            features.append(np.max(fft_vals[1:]))           # 跳过直流分量
        else:
            features.append(0.0)

        # 频谱质心
        if len(fft_vals) > 1 and np.sum(fft_vals[1:]) > 0:
            features.append(
                np.sum(freqs[1:] * fft_vals[1:]) / np.sum(fft_vals[1:])
            )
        else:
            features.append(0.0)

    return features  # length = 22


def build_feature_matrix(filtered_data, labels, variants):
    """
    滑动窗口 -> 特征提取 -> 标签清洗。
    返回 X（特征矩阵）、y（标签）、window_variants（用于按子变体评估）。
    """
    X_feat, y_feat, w_variants = [], [], []

    file_iter = enumerate(filtered_data)
    if HAS_TQDM:
        file_iter = tqdm(list(file_iter), desc="  特征提取", unit="文件")

    for i, data in file_iter:
        original_label = labels[i]
        var = variants[i]

        for start in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
            window = data[start:start + WINDOW_SIZE]

            # 标签清洗：信号太平坦则强制标为 Rest
            ptp_h = np.max(window[:, 0]) - np.min(window[:, 0])
            ptp_v = np.max(window[:, 1]) - np.min(window[:, 1])
            max_ptp = max(ptp_h, ptp_v)

            if max_ptp < CLEANING_THRESHOLD:
                current_label = 0  # Rest
            else:
                current_label = original_label

            features = extract_features_v2(window)
            X_feat.append(features)
            y_feat.append(current_label)
            w_variants.append(var)

    return np.array(X_feat), np.array(y_feat), w_variants


# ============================================================
# 模块 5：数据充足性分析与类别平衡
# ============================================================
def analyze_class_balance(y, w_variants):
    """打印类别分布和数据充足性分析。"""
    print("\n" + "=" * 60)
    print("类别平衡分析 (滑动窗口 + 标签清洗后)")
    print("=" * 60)

    counts = Counter(y)
    total = len(y)
    print(f"\n总窗口数: {total}")
    print(f"{'类别':<10} {'数量':>7} {'比例':>8}")
    print("-" * 28)
    for cls_id in range(len(CLASSES)):
        c = counts.get(cls_id, 0)
        print(f"{CLASSES[cls_id]:<10} {c:>7} {c/total:>7.1%}")

    # 不平衡比率
    max_c = max(counts.values())
    min_c = min(counts.values()) if min(counts.values()) > 0 else 1
    print(f"\n不平衡比率 (最大/最小): {max_c/min_c:.2f}x")

    # 数据充足性启发式判断
    n_classes = len(CLASSES)
    n_features = N_FEATURES
    min_recommended = n_features * 10 * n_classes  # rule of thumb
    print(f"\n数据充足性检查:")
    print(f"  特征数: {n_features}, 类别数: {n_classes}")
    print(f"  建议最少样本数: ~{min_recommended}")
    print(f"  当前总数: {total}", end="")
    if total >= min_recommended:
        print(" (充足)")
    else:
        print(f" (不足 - 建议采集更多数据)")

    # 每子变体的窗口数分布
    print(f"\n--- 每子变体窗口数 ---")
    var_counts = {}
    for label, var in zip(y, w_variants):
        key = (CLASSES[label], var)
        var_counts[key] = var_counts.get(key, 0) + 1
    for key in sorted(var_counts.keys()):
        print(f"  {key[0]:>8} / {key[1]:<10}: {var_counts[key]:>5} 个窗口")

    return counts


def apply_balancing(X_train, y_train):
    """如果 SMOTE 可用则过采样，否则提示使用 class_weight。"""
    if HAS_SMOTE:
        print("  正在应用 SMOTE 过采样...")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_bal, y_bal = sm.fit_resample(X_train, y_train)
        print(f"  之前: {len(X_train)} -> 之后: {len(X_bal)} 个样本")
        return X_bal, y_bal
    else:
        print("  SMOTE 不可用；将在模型中使用 class_weight='balanced' 代替")
        return X_train, y_train


# ============================================================
# 模块 6：多模型训练与超参数调优
# ============================================================
def get_models(use_balanced_weights=True):
    """返回模型字典：模型名 -> (模型实例, 是否需要归一化)"""
    cw = 'balanced' if use_balanced_weights else None

    models = {
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=200, max_depth=None,
                class_weight=cw, random_state=RANDOM_STATE
            ), False  # 树模型不需要归一化
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                random_state=RANDOM_STATE
            ), False
        ),
        "SVM_RBF": (
            SVC(
                kernel='rbf', probability=True,
                class_weight=cw, random_state=RANDOM_STATE
            ), True
        ),
        "KNN_7": (
            KNeighborsClassifier(n_neighbors=7), True
        ),
        "MLP": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, validation_fraction=0.1,
                random_state=RANDOM_STATE
            ), True
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=RANDOM_STATE
            ), False
        )

    return models


def train_all_models(X_train, X_test, y_train, y_test, scaler):
    """训练所有模型，返回按准确率排序的结果。"""
    models = get_models(use_balanced_weights=not HAS_SMOTE)
    results = {}

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    n_models = len(models)
    for idx, (name, (model, needs_scaling)) in enumerate(models.items(), 1):
        print(f"\n  [{idx}/{n_models}] 正在训练 {name}...")
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test

        print(f"    -> 拟合模型...", end="", flush=True)
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        print(" 完成")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 5 折交叉验证
        print(f"    -> 5 折交叉验证...", end="", flush=True)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='accuracy')
        print(" 完成")

        results[name] = {
            'model': model,
            'needs_scaling': needs_scaling,
            'accuracy': acc,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
        }
        print(f"    准确率: {acc:.4f} | F1: {f1:.4f} | 交叉验证: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # 按准确率排序
    results = dict(sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True))
    return results


def tune_best_model(results, X_train, y_train, scaler):
    """对排名第一的模型进行 RandomizedSearchCV 超参数调优。"""
    best_name = list(results.keys())[0]
    best_info = results[best_name]

    param_grids = {
        "RandomForest": {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        "GradientBoosting": {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
        },
        "XGBoost": {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
        },
    }

    if best_name not in param_grids:
        print(f"\n  {best_name} 没有调优参数网格，跳过超参数搜索")
        return results[best_name]['model'], best_name

    print(f"\n  正在用 RandomizedSearchCV 调优 {best_name} (20 次迭代, 5 折)...")
    Xtr = X_train if not best_info['needs_scaling'] else scaler.transform(X_train)

    search = RandomizedSearchCV(
        best_info['model'].__class__(**{
            k: v for k, v in best_info['model'].get_params().items()
            if k != 'random_state'
        }, random_state=RANDOM_STATE),
        param_grids[best_name],
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(Xtr, y_train)

    print(f"  最佳参数: {search.best_params_}")
    print(f"  最佳交叉验证准确率: {search.best_score_:.4f}")
    return search.best_estimator_, best_name + "_tuned"


def build_voting_ensemble(results, top_n=3):
    """用排名前 N 的模型构建软投票集成分类器。"""
    top_names = list(results.keys())[:top_n]
    estimators = []
    for name in top_names:
        estimators.append((name, results[name]['model']))

    print(f"\n  正在用前 {top_n} 个模型构建 VotingClassifier: {top_names}")
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    return ensemble


# ============================================================
# 模块 7：深度评估
# ============================================================
def evaluate_best(model, X_test, y_test, scaler, needs_scaling,
                  w_variants_test, model_name="Best"):
    """完整评估：混淆矩阵、分类报告、按子变体准确率。"""
    Xte = scaler.transform(X_test) if needs_scaling else X_test
    y_pred = model.predict(Xte)

    print("\n" + "=" * 60)
    print(f"详细评估: {model_name}")
    print("=" * 60)

    # 分类报告
    print("\n--- 分类报告 ---")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"混淆矩阵 - {model_name}")
    plt.xlabel("预测值")
    plt.ylabel("真实值")
    plt.tight_layout()
    plt.savefig("confusion_matrix_v2.png", dpi=150)
    plt.show()
    print("  已保存: confusion_matrix_v2.png")

    # 误触分析（Rest 被误判为动作）
    print("\n--- 误触分析 (Rest -> Action) ---")
    rest_mask = (y_test == 0)
    rest_preds = y_pred[rest_mask]
    fp_count = np.sum(rest_preds != 0)
    total_rest = np.sum(rest_mask)
    print(f"  Rest 样本数: {total_rest}")
    print(f"  误触数: {fp_count} ({fp_count/max(total_rest,1):.1%})")
    if fp_count > 0:
        fp_classes = Counter(rest_preds[rest_preds != 0])
        for cls_id, cnt in fp_classes.most_common():
            print(f"    -> 被误判为 {CLASSES[cls_id]}: {cnt}")

    # 按子变体准确率
    print("\n--- 按子变体准确率 ---")
    var_correct = {}
    var_total = {}
    for i in range(len(y_test)):
        var = w_variants_test[i]
        true_cls = CLASSES[y_test[i]]
        key = f"{true_cls}/{var}"
        var_total[key] = var_total.get(key, 0) + 1
        if y_pred[i] == y_test[i]:
            var_correct[key] = var_correct.get(key, 0) + 1

    print(f"  {'类别/变体':<25} {'准确率':>7} {'正确数':>8} {'总数':>7}")
    print("  " + "-" * 50)
    for key in sorted(var_total.keys()):
        correct = var_correct.get(key, 0)
        total = var_total[key]
        acc = correct / total if total > 0 else 0
        print(f"  {key:<25} {acc:>6.1%} {correct:>8} {total:>7}")

    return y_pred


def plot_feature_importance(model, model_name):
    """绘制树模型的特征重要性排名图。"""
    if not hasattr(model, 'feature_importances_'):
        print(f"\n  {model_name} 不支持 feature_importances_，跳过")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n--- 特征重要性 ({model_name}) ---")
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(importances)), importances[indices], color='steelblue')
    plt.xticks(range(len(importances)),
               [FEATURE_NAMES[i] for i in indices], rotation=45, ha='right')
    plt.title(f"特征重要性 - {model_name}")
    plt.ylabel("重要性")
    plt.tight_layout()
    plt.savefig("feature_importance_v2.png", dpi=150)
    plt.show()
    print("  已保存: feature_importance_v2.png")

    # 打印前 10 个重要特征
    print(f"\n  前 10 个重要特征:")
    for rank, idx in enumerate(indices[:10]):
        print(f"    {rank+1}. {FEATURE_NAMES[idx]:<25} {importances[idx]:.4f}")


# ============================================================
# 模块 8：保存模型与元数据
# ============================================================
def save_model(model, scaler, model_name):
    """保存模型、归一化器和特征配置元数据。"""
    model_path = 'eog_model_v3.joblib'
    scaler_path = 'eog_scaler_v3.joblib'
    config_path = 'feature_config_v3.json'

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    config = {
        'version': 3,
        'model_name': model_name,
        'n_features': N_FEATURES,
        'feature_names': FEATURE_NAMES,
        'classes': CLASSES,
        'window_size': WINDOW_SIZE,
        'sampling_rate': SAMPLING_RATE,
        'filter': {'lowcut': 0.5, 'highcut': 10.0, 'order': 4},
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n  已保存: {model_path}")
    print(f"  已保存: {scaler_path}")
    print(f"  已保存: {config_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("EOG 眼动分类 - 训练流水线 v2")
    print("=" * 60)

    # --- 步骤 1：加载数据 ---
    raw_data, labels, fnames, variants = load_raw_data(DATA_PATH)
    if len(raw_data) == 0:
        print("未加载到数据，请检查 DATA_PATH。")
        return
    print_data_stats(raw_data, labels, fnames, variants)

    # --- 步骤 2：带通滤波 ---
    print("\n正在应用带通滤波 (0.5-10 Hz)...")
    filtered = [apply_filter(d) for d in raw_data]
    print(f"  已滤波 {len(filtered)} 个文件")

    # --- 步骤 3：少数类数据增强 ---
    print("\n正在对少数类进行数据增强...")
    aug_data, aug_labels, aug_fnames, aug_variants = augment_data(
        filtered, labels, fnames, variants
    )

    # --- 步骤 4：提取特征 ---
    print("\n正在用滑动窗口提取 22 个特征...")
    X, y, w_variants = build_feature_matrix(
        aug_data, aug_labels, aug_variants
    )
    print(f"  特征矩阵: {X.shape}")

    # --- 步骤 5：分析类别平衡 ---
    analyze_class_balance(y, w_variants)

    # --- 步骤 6：训练/测试集划分 ---
    print("\n正在划分数据 (80/20, 分层)...")
    # 需要同步划分 w_variants 与 X, y
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices, test_size=TEST_RATIO,
        random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    wv_test = [w_variants[i] for i in idx_test]
    print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # --- 步骤 7：特征归一化 ---
    scaler = StandardScaler()
    scaler.fit(X_train)

    # --- 步骤 8：类别平衡（仅对训练集做 SMOTE）---
    X_train_bal, y_train_bal = apply_balancing(
        scaler.transform(X_train), y_train
    )
    # 逆变换回原始尺度，以便树模型使用未归一化的数据
    X_train_bal_unscaled = scaler.inverse_transform(X_train_bal)

    # --- 步骤 9：训练所有模型 ---
    results = train_all_models(
        X_train_bal_unscaled, X_test, y_train_bal, y_test, scaler
    )

    # --- 步骤 10：调优最佳模型 ---
    best_model, best_name = tune_best_model(
        results, X_train_bal_unscaled, y_train_bal, scaler
    )

    # --- 步骤 11：详细评估 ---
    best_info = list(results.values())[0]
    evaluate_best(
        best_model, X_test, y_test, scaler,
        best_info['needs_scaling'], wv_test, best_name
    )
    plot_feature_importance(best_model, best_name)

    # --- 步骤 12：保存模型 ---
    save_model(best_model, scaler, best_name)

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"  最佳模型: {best_name}")
    print(f"  测试准确率: {best_info['accuracy']:.4f}")
    print(f"  依赖项: xgboost={'有' if HAS_XGB else '无'}, "
          f"smote={'有' if HAS_SMOTE else '无'}")
    print(f"\n如需在 realtime_viz.py 中使用，请将 MODEL_PATH/SCALER_PATH")
    print(f"指向 eog_model_v3.joblib / eog_scaler_v3.joblib")


if __name__ == "__main__":
    main()
