import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Configuration Defaults ---
WINDOW_SIZE = 50  # samples per window
STEP_SIZE = 10
DATA_DIR = os.path.join('.', 'DATA', 'EOG_data')
MODEL_OUT = 'eog_model.pkl'

# Map folders to labels (folder names expected: Blink, Down, Left, Rest, Right, Up)
ACTION_MAP = {
    'Rest': 0,
    'Blink': 1,
    'Left': 2,
    'Right': 3,
    'Up': 4,
    'Down': 5
}

def extract_features(window_data):
    v_data = window_data[:, 0]
    h_data = window_data[:, 1]
    feats = []
    feats.append(np.mean(v_data))
    feats.append(np.std(v_data))
    feats.append(np.max(v_data) - np.min(v_data))
    feats.append(np.mean(h_data))
    feats.append(np.std(h_data))
    feats.append(np.max(h_data) - np.min(h_data))
    return feats

def gather_csv_windows_from_folder(folder_path, label_id, window_size=WINDOW_SIZE, step=STEP_SIZE):
    X = []
    y = []
    if not os.path.isdir(folder_path):
        print(f"Warning: folder not found: {folder_path}")
        return X, y

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    if not files:
        print(f"Warning: no CSV files in {folder_path}")

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            # Try reading with header (many exported CSVs include header names)
            df = pd.read_csv(fpath, header=0)
            # Convert all columns to numeric when possible, set non-convertible to NaN
            df_num = df.apply(pd.to_numeric, errors='coerce')
            # Drop columns that are entirely non-numeric
            df_num = df_num.dropna(axis=1, how='all')

            # If after dropping non-numeric columns we have fewer than 2 cols, try reading without header
            if df_num.shape[1] < 2:
                df = pd.read_csv(fpath, header=None)
                df_num = df.apply(pd.to_numeric, errors='coerce')
                df_num = df_num.dropna(axis=1, how='all')

            if df_num.shape[1] < 2:
                print(f"Skipping {fpath}: fewer than 2 numeric columns after cleaning")
                continue

            # Use first two numeric columns (assumed V, H)
            raw = df_num.iloc[:, :2].values.astype(float)
            n = len(raw)
            if n >= window_size:
                for i in range(0, n - window_size + 1, step):
                    window = raw[i:i+window_size]
                    X.append(extract_features(window))
                    y.append(label_id)
            else:
                print(f"Skipping {fpath}: too short ({n} samples)")
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

    return X, y

def build_dataset(data_dir=DATA_DIR):
    X = []
    y = []
    print(f"Scanning data directory: {data_dir}")
    for action_name, label_id in ACTION_MAP.items():
        folder = os.path.join(data_dir, action_name)
        print(f"  Processing action '{action_name}' from {folder} ...")
        X_part, y_part = gather_csv_windows_from_folder(folder, label_id)
        X.extend(X_part)
        y.extend(y_part)

    X = np.array(X)
    y = np.array(y)
    return X, y

def train(data_dir=DATA_DIR, model_out=MODEL_OUT):
    print("1) Building dataset from CSV folders...")
    X, y = build_dataset(data_dir)
    if len(X) == 0:
        print("No training samples found. Check DATA/EOG_data folder structure and CSV files.")
        return

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    print("2) Splitting and training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Training complete. Test accuracy: {acc*100:.2f}%")
    print("Classification report:\n", classification_report(y_test, y_pred))

    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EOG model from DATA/EOG_data folder structure')
    parser.add_argument('--data', type=str, default=DATA_DIR, help='Path to DATA/EOG_data folder')
    parser.add_argument('--out', type=str, default=MODEL_OUT, help='Output model filename')
    args = parser.parse_args()
    train(data_dir=args.data, model_out=args.out)