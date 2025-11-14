import cv2
import torch
import numpy as np
import mediapipe as mp
import joblib
import torch.nn as nn

# ===============================
# ⚙️ Model Definition
# ===============================
class TransformerAutoencoder(nn.Module):
    def __init__(self, num_features, seq_len, d_model=128, nhead=8, num_layers=6):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, num_features)

    def forward(self, x):
        z = self.input_proj(x)
        memory = self.encoder(z)
        reconstructed = self.decoder(z, memory)
        recon_out = self.output_proj(reconstructed)
        return recon_out


# ===============================
# 📐 Helper Functions
# ===============================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def extract_angles_from_landmarks(landmarks_dict):
    angles = []
    def get(p): return [landmarks_dict[f"{p}_x"], landmarks_dict[f"{p}_y"]]

    angles.append(calculate_angle(get("LEFT_HIP"), get("LEFT_KNEE"), get("LEFT_ANKLE")))
    angles.append(calculate_angle(get("RIGHT_HIP"), get("RIGHT_KNEE"), get("RIGHT_ANKLE")))
    angles.append(calculate_angle(get("LEFT_SHOULDER"), get("LEFT_HIP"), get("LEFT_KNEE")))
    angles.append(calculate_angle(get("RIGHT_SHOULDER"), get("RIGHT_HIP"), get("RIGHT_KNEE")))
    angles.append(calculate_angle(get("LEFT_SHOULDER"), get("LEFT_HIP"), [get("LEFT_HIP")[0], get("LEFT_HIP")[1]-1]))
    angles.append(calculate_angle(get("RIGHT_SHOULDER"), get("RIGHT_HIP"), [get("RIGHT_HIP")[0], get("RIGHT_HIP")[1]-1]))
    angles.append(calculate_angle(get("LEFT_KNEE"), get("LEFT_ANKLE"), get("LEFT_FOOT_INDEX")))
    angles.append(calculate_angle(get("RIGHT_KNEE"), get("RIGHT_ANKLE"), get("RIGHT_FOOT_INDEX")))

    return angles


LANDMARK_NAMES = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20, 'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28, 'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
}


# ===============================
# 🧠 Inference Function
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_frame(frame, model, scaler, threshold, device, buffer, window_size, num_features):
    """
    Takes a video frame and returns analysis result for the exercise.
    Handles MediaPipe landmarks, feature extraction, scaling, and reconstruction error.
    """
    import numpy as np
    import torch
    import cv2

    # Convert BGR -> RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        # Append zero vector if no pose detected
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "reconstruction_error": None}

    # Extract landmarks
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = lm.x
        landmarks_dict[f"{name}_y"] = lm.y
        landmarks_dict[f"{name}_visibility"] = lm.visibility

    # Feature vector (x, y, visibility for each landmark) + angles
    feature_vector = []
    for name in LANDMARK_NAMES:
        feature_vector.extend([
            landmarks_dict[f"{name}_x"],
            landmarks_dict[f"{name}_y"],
            landmarks_dict[f"{name}_visibility"]
        ])
    feature_vector.extend(extract_angles_from_landmarks(landmarks_dict))

    # Append to buffer
    buffer.append(feature_vector)

    # Only analyze when buffer is full
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32) # Convert deque to list first
        print("Window shape:", window.shape)

        try:
            # Scale features
            scaled = scaler.transform(window)  # should be (window_size, num_features)
        except Exception as e:
            print("Scaler transform error:", e)
            return {"form_status": "Scaler Error", "reconstruction_error": None}

        # Convert to tensor with batch dimension
        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, seq_len, num_features)

        # Model inference
        with torch.no_grad():
            recon = model(tensor)
            if recon.shape != tensor.shape:
                print(f"Warning: recon shape {recon.shape} != input {tensor.shape}")
            err = torch.mean((tensor - recon) ** 2).item()

        status = "Normal" if err <= threshold else "Anomaly"
        return {"form_status": status, "reconstruction_error": err}

    # Buffer not full yet
    return {"form_status": "Analyzing...", "reconstruction_error": None}