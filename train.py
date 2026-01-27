import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration Parameters ---
WINDOW_SIZE = 50  # 窗口大小 (50 samples = 1 second at 50Hz)
STEP_SIZE = 10    # 滑动步长

# --- Core: Feature Extraction Function ---
# 特征提取逻辑保持不变，因为垂直通道(V)的数据已经包含在输入中了
# Up/Down 主要影响 V 通道，Left/Right 主要影响 H 通道
def extract_features(window_data):
    # window_data shape: (50, 2) -> col 0: Vertical, col 1: Horizontal
    v_data = window_data[:, 0]
    h_data = window_data[:, 1]
    
    features = []
    
    # --- 垂直通道 (Vertical) 特征 ---
    # Up/Down 动作会显著改变这里的 Mean (均值) 和 Amplitude
    # Blink (眨眼) 会显著改变这里的 Amplitude (峰峰值)
    features.append(np.mean(v_data))       # 均值
    features.append(np.std(v_data))        # 标准差
    features.append(np.max(v_data) - np.min(v_data)) # 幅值 (Peak-to-Peak)
    
    # --- 水平通道 (Horizontal) 特征 ---
    # Left/Right 动作会显著改变这里的 Mean 和 Amplitude
    features.append(np.mean(h_data))
    features.append(np.std(h_data))
    features.append(np.max(h_data) - np.min(h_data))
    
    return features

# --- Main Training Process ---
def train():
    print("1. Loading CSV data...")
    X = [] # 特征矩阵
    y = [] # 标签向量
    
    # Map action names to ID labels
    # [修改点]: 在这里增加了 Up 和 Down
    actions = {
        "Idle": 0, 
        "Blink": 1, 
        "Left": 2, 
        "Right": 3,
        "Up": 4,    # 新增: 向上看
        "Down": 5   # 新增: 向下看
    }
    
    for action_name, label_id in actions.items():
        filename = f"{action_name}.csv"
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"⚠️  Error: File '{filename}' not found. Skipping.")
            continue
            
        print(f"   Processing: {filename} ...")
        
        try:
            # === Read CSV ===
            df = pd.read_csv(filename, header=None)
            
            # Ensure data is float type
            raw_data = df.values.astype(float)
            
            # Check column count (Need at least 2: V and H)
            if raw_data.shape[1] < 2:
                print(f"⚠️  Warning: {filename} has fewer than 2 columns. Skipping.")
                continue

            # === Sliding Window Slicing ===
            if len(raw_data) > WINDOW_SIZE:
                for i in range(0, len(raw_data) - WINDOW_SIZE, STEP_SIZE):
                    window = raw_data[i : i + WINDOW_SIZE]
                    features = extract_features(window)
                    X.append(features)
                    y.append(label_id)
            else:
                print(f"⚠️  Warning: {filename} data is too short for one window.")

        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            continue

    # Convert to Numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("\n❌ Error: No features extracted. Check your CSV files.")
        return

    print(f"\n   Dataset built: {X.shape[0]} samples with {X.shape[1]} features each")

    # Split Data
    print("2. Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and Train Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=10)
    clf.fit(X_train, y_train)

    # Validate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"3. Training complete. Test Accuracy: {acc*100:.2f}%")

    # Save Model
    joblib.dump(clf, "eog_model.pkl")
    print("4. Model saved as 'eog_model.pkl'")

if __name__ == "__main__":
    train()