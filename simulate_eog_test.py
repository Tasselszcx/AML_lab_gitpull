import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Feature names (match extract_features order)
FEATURE_NAMES = [
    'mean_v', 'std_v', 'p2p_v',
    'mean_h', 'std_h', 'p2p_h'
]

# Configuration
WINDOW_SIZE = 50
ID_TO_ACTION = {0: "Idle", 1: "Blink", 2: "Left", 3: "Right", 4: "Up", 5: "Down"}
ACTION_TO_ID = {v:k for k,v in ID_TO_ACTION.items()}
MODEL_PATH = "eog_model.pkl"

# Feature extraction (must match training)
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
    return np.array(feats)


def explain_prediction(feats, proba, id_to_action=ID_TO_ACTION):
    # Return a short human-readable explanation for a single prediction
    top_idx = np.argmax(proba)
    top_prob = proba[top_idx]
    feat_summary = ', '.join([f"{n}:{v:.3f}" for n, v in zip(FEATURE_NAMES, feats)])
    return top_idx, top_prob, feat_summary

# Simulate a single window for a given action
def simulate_window(action, window_size=WINDOW_SIZE, noise_level=0.08):
    t = np.linspace(0, 1, window_size)
    v = np.random.normal(0, noise_level, window_size)
    h = np.random.normal(0, noise_level, window_size)

    if action == "Idle":
        # small noise only
        pass
    elif action == "Blink":
        # sharp vertical transient (Gaussian pulse)
        amp = np.random.uniform(1.0, 2.0)
        center = np.random.uniform(0.35, 0.65)
        width = np.random.uniform(0.03, 0.08)
        pulse = amp * np.exp(-0.5 * ((t - center) / width) ** 2)
        v += pulse
    elif action == "Left":
        # sustained horizontal shift to positive side with small ramp
        shift = np.random.uniform(0.4, 1.0)
        ramp = np.linspace(0, shift, window_size) * np.random.uniform(0.6, 1.0)
        h += ramp
    elif action == "Right":
        # sustained horizontal shift to negative side
        shift = np.random.uniform(0.4, 1.0)
        ramp = -np.linspace(0, shift, window_size) * np.random.uniform(0.6, 1.0)
        h += ramp
    elif action == "Up":
        # sustained vertical positive offset
        offset = np.random.uniform(0.4, 1.0)
        v += offset * np.linspace(0.2, 1.0, window_size)
    elif action == "Down":
        offset = np.random.uniform(0.4, 1.0)
        v -= offset * np.linspace(0.2, 1.0, window_size)

    # small jitter
    v += np.random.normal(0, noise_level * 0.5, window_size)
    h += np.random.normal(0, noise_level * 0.5, window_size)

    return np.column_stack((v, h))

# Build dataset of simulated windows
def build_simulated_dataset(n_per_class=80):
    X_windows = []
    y = []
    for action in ID_TO_ACTION.values():
        for i in range(n_per_class):
            w = simulate_window(action)
            X_windows.append(w)
            y.append(ACTION_TO_ID[action])
    return np.array(X_windows), np.array(y)

# Run simulation and prediction
def run_simulation(show_plots=True):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model not found at {MODEL_PATH}. Train first or place the model in the working folder.")
        return

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Simulating data...")
    X_windows, y_true = build_simulated_dataset(n_per_class=60)

    # Extract features and predict
    X_feats = np.array([extract_features(w) for w in X_windows])
    y_pred = model.predict(X_feats)
    y_proba = model.predict_proba(X_feats)

    acc = accuracy_score(y_true, y_pred)
    print(f"Simulation accuracy: {acc*100:.2f}%")
    print("Classification report:\n", classification_report(y_true, y_pred, target_names=list(ID_TO_ACTION.values())))

    # Print a few example predictions with feature summaries and probabilities
    print("\nExample predictions (feature summary, top_prob, true->pred):")
    for i in np.random.choice(len(X_feats), size=min(8, len(X_feats)), replace=False):
        feats = X_feats[i]
        proba = y_proba[i]
        top_idx, top_prob, feat_summary = explain_prediction(feats, proba)
        print(f"  [{i}] {feat_summary} | {top_prob:.2f} | {ID_TO_ACTION[y_true[i]]} -> {ID_TO_ACTION[top_idx]}")

    # Show some typical misclassifications
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) > 0:
        print(f"\nFound {len(mis_idx)} misclassified examples, showing up to 6:")
        for i in mis_idx[:6]:
            feats = X_feats[i]
            proba = y_proba[i]
            top_idx, top_prob, feat_summary = explain_prediction(feats, proba)
            print(f"  MIS [{i}] true:{ID_TO_ACTION[y_true[i]]} pred:{ID_TO_ACTION[top_idx]} p:{top_prob:.2f} | {feat_summary}")

    if show_plots:
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(ID_TO_ACTION.keys()))

        fig = plt.figure(constrained_layout=True, figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)

        ax0 = fig.add_subplot(gs[0, 0])
        im = ax0.imshow(cm, interpolation='nearest', cmap='Blues')
        ax0.set_title('Confusion Matrix (simulated)')
        ticks = np.arange(len(ID_TO_ACTION))
        ax0.set_xticks(ticks)
        ax0.set_yticks(ticks)
        ax0.set_xticklabels(list(ID_TO_ACTION.values()), rotation=45)
        ax0.set_yticklabels(list(ID_TO_ACTION.values()))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax0.text(j, i, cm[i, j], ha='center', va='center', color='black')
        fig.colorbar(im, ax=ax0)

        # Example time-series per true class (first example of each)
        ax1 = fig.add_subplot(gs[0, 1])
        for class_id, action in ID_TO_ACTION.items():
            idxs = np.where(y_true == class_id)[0]
            if len(idxs) == 0:
                continue
            idx = idxs[0]
            w = X_windows[idx]
            t = np.arange(w.shape[0])
            ax1.plot(t, w[:, 0], label=f"V {action}")
            ax1.plot(t, w[:, 1], '--', label=f"H {action}")
        ax1.set_title('Example windows (one per class)')
        ax1.legend(fontsize='small', ncol=2)

        # Predicted class distribution
        ax2 = fig.add_subplot(gs[1, :])
        counts = [np.sum(y_pred == k) for k in sorted(ID_TO_ACTION.keys())]
        ax2.bar(list(ID_TO_ACTION.values()), counts, color='C2')
        ax2.set_title('Predicted counts (simulated dataset)')

        # Additional: pairwise feature scatter (true vs predicted)
        try:
            pair_fig, pair_axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
            pairs = [ (0,3), (1,4), (2,5) ]  # mean_v vs mean_h, std_v vs std_h, p2p_v vs p2p_h
            for ax_row, (a,b) in zip(pair_axes, pairs):
                ax_true = ax_row[0]
                ax_pred = ax_row[1]
                for class_id in sorted(ID_TO_ACTION.keys()):
                    mask = (y_true == class_id)
                    ax_true.scatter(X_feats[mask, a], X_feats[mask, b], label=ID_TO_ACTION[class_id], s=8)
                ax_true.set_xlabel(FEATURE_NAMES[a]); ax_true.set_ylabel(FEATURE_NAMES[b]); ax_true.set_title('Features by TRUE label')
                for class_id in sorted(ID_TO_ACTION.keys()):
                    mask = (y_pred == class_id)
                    ax_pred.scatter(X_feats[mask, a], X_feats[mask, b], label=ID_TO_ACTION[class_id], s=8)
                ax_pred.set_xlabel(FEATURE_NAMES[a]); ax_pred.set_ylabel(FEATURE_NAMES[b]); ax_pred.set_title('Features by PRED label')
            pair_fig.suptitle('Pairwise feature scatter (true vs predicted)')
            pair_out = 'feature_pairs.png'
            pair_fig.savefig(pair_out)
            print(f"Saved feature scatter to {pair_out}")
        except Exception as e:
            print(f"Could not create feature scatter plots: {e}")

        out_png = 'simulation_results.png'
        fig.suptitle(f"Simulated EOG test — accuracy: {acc*100:.1f}%")
        fig.savefig(out_png)
        print(f"Saved visualization to {out_png}")
        plt.show()

    return y_true, y_pred, y_proba

if __name__ == '__main__':
    run_simulation()
