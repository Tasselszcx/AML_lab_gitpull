import serial
import numpy as np
import joblib
import time
from collections import deque

# --- Configuration ---
SERIAL_PORT = 'COM6'  # Windows: usually COMx; Mac/Linux: /dev/ttyUSBx
BAUD_RATE = 115200
WINDOW_SIZE = 50      # MUST match the window size used in training
ID_TO_ACTION = {0: "Idle", 1: "Blink", 2: "Left", 3: "Right", 4: "Up", 5: "Down"}

# --- Load Model ---
try:
    print("Loading model...")
    model = joblib.load("eog_model.pkl")
    print("Model loaded successfully!")
except:
    print("Error: 'eog_model.pkl' not found. Please run the training script first!")
    exit()

# --- Feature Extraction (Must match training script exactly) ---
def extract_features(window_data):
    v_data = window_data[:, 0]
    h_data = window_data[:, 1]
    features = []
    
    # Vertical features
    features.append(np.mean(v_data))
    features.append(np.std(v_data))
    features.append(np.max(v_data) - np.min(v_data))
    
    # Horizontal features
    features.append(np.mean(h_data))
    features.append(np.std(h_data))
    features.append(np.max(h_data) - np.min(h_data))
    
    return np.array(features).reshape(1, -1) # Sklearn requires a 2D array

# --- Main Loop ---
def main():
    # Initialize deque as a sliding window buffer
    data_buffer = deque(maxlen=WINDOW_SIZE)
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}, listening for eye movements...")
        
        while True:
            if ser.in_waiting > 0:
                # 添加 errors='ignore' 来跳过无法解码的乱码字节
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                
                try:
                    # Parse Arduino data format: "valV,valH"
                    parts = line.split(',')
                    if len(parts) != 2: continue
                    
                    val_v = float(parts[0])
                    val_h = float(parts[1])
                    
                    # Add to buffer
                    data_buffer.append([val_v, val_h])
                    
                    # Only predict when the buffer is full (window is complete)
                    if len(data_buffer) == WINDOW_SIZE:
                        # Convert to numpy array
                        window_array = np.array(data_buffer)
                        
                        # Extract features
                        feats = extract_features(window_array)
                        
                        # Predict
                        prediction_id = model.predict(feats)[0]
                        # Get probability (Optional: filter low confidence results)
                        probs = model.predict_proba(feats)[0]
                        confidence = np.max(probs)

                        action_name = ID_TO_ACTION.get(prediction_id, "Unknown")
                        
                        # Smoothing: Only print if confidence is high or action is not Idle
                        if prediction_id != 0 and confidence > 0.7:
                            print(f"Action Detected: {action_name}  (Confidence: {confidence:.2f})")
                        elif prediction_id == 0 and np.random.rand() > 0.95:
                            # Occasionally print status to show program is alive
                            print("...Idle...")
                            
                except ValueError:
                    pass # Ignore parsing errors
                    
    except serial.SerialException:
        print(f"Could not open port {SERIAL_PORT}. Please check connection.")
    except KeyboardInterrupt:
        print("\nProgram stopped.")
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()