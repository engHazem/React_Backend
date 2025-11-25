# models/lateral_raises/inference.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import onnxruntime as ort

# ===============================
# 📐 Angle Calculation
# ===============================
def calculate_angle(a, b, c):
    a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# ===============================
# 📍 MediaPipe Landmarks & Lists
# ===============================
LANDMARK_NAMES = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP'
]

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13, 'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15, 'RIGHT_WRIST': 16, 'LEFT_PINKY': 17, 'RIGHT_PINKY': 18,
    'LEFT_INDEX': 19, 'RIGHT_INDEX': 20, 'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24
}

LEFT_LANDMARKS = [
    'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
    'LEFT_PINKY', 'LEFT_INDEX', 'LEFT_THUMB', 'LEFT_HIP'
]

RIGHT_LANDMARKS = [
    'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
    'RIGHT_PINKY', 'RIGHT_INDEX', 'RIGHT_THUMB', 'RIGHT_HIP'
]


# ===============================
# 🧠 MediaPipe Pose (shared)
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# ===============================
# 🔥 ONNX Inference Function
# ===============================
def analyze_frame(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):
    """
    frame: BGR OpenCV image
    session: onnxruntime.InferenceSession for lateral_raises_model.onnx
    scaler: pretrained scaler (joblib) matching num_features
    threshold: reconstruction error threshold
    buffer: deque(maxlen=window_size) holding feature vectors
    window_size: seq length
    num_features: expected feature vector length
    rep_state: dict tracking rep state (will be mutated)
    """

    # convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    # Extract landmarks into dict (floats)
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = float(lm.x)
        landmarks_dict[f"{name}_y"] = float(lm.y)
        landmarks_dict[f"{name}_visibility"] = float(lm.visibility)

    # Build feature vector from landmarks (14 * 3 = 42) then append angles (12)
    feature_vector = []
    for name in LANDMARK_NAMES:
        feature_vector.extend([
            landmarks_dict.get(f"{name}_x", 0.0),
            landmarks_dict.get(f"{name}_y", 0.0),
            landmarks_dict.get(f"{name}_visibility", 0.0)
        ])

    # Compute angles and related features
    angles = extract_angles_from_landmarks(landmarks_dict)
    # angles returns 12 values:
    # [left_shoulder_angle, right_shoulder_angle, left_elbow_angle, right_elbow_angle,
    #  left_torso_angle, right_torso_angle, left_shoulder_elevation, right_shoulder_elevation,
    #  left_wrist_angle, right_wrist_angle, left_arm_drift, right_arm_drift]
    feature_vector.extend(angles)

    # Calculate average shoulder angle (for rep counting)
    left_shoulder_angle = angles[0]
    right_shoulder_angle = angles[1]
    angle = (left_shoulder_angle + right_shoulder_angle) / 2.0

    # Validate feature length
    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        # If mismatch, log and pad zero vector (prevents crash)
        print(f"Warning: Feature mismatch. Expected {num_features}, got {len(feature_vector)}")
        buffer.append([0.0] * num_features)

    # ---------- Rep counting logic (lateral raises) ----------
    if angle is not None:
        if rep_state.get("prev_angle") is None:
            rep_state["prev_angle"] = angle

        prev_angle = rep_state.get("prev_angle", angle)
        prev_phase = rep_state.get("prev_phase")

        # Phase detection thresholds (preserve your behavior)
        if angle <= 30 and prev_angle > 30:
            rep_state["Top_ROM_error"] = False
            rep_state["Bottom_ROM_error"] = False
            rep_state["phase"] = "LR1"  # Rest
        elif angle >= 75:
            rep_state["phase"] = "LR3"  # Top
        elif angle > prev_angle and angle < 75 and angle > 30:
            rep_state["phase"] = "LR2"  # Going up
        elif angle < prev_angle and angle < 75 and angle > 30:
            rep_state["phase"] = "LR4"  # Going down

        # ROM checks (same logic)
        if prev_phase is not None:
            if rep_state["phase"] == "LR4" and prev_phase == "LR2":
                rep_state["viable_rep"] = False
                rep_state["Top_ROM_error"] = True
        if rep_state["phase"] == "LR2" and prev_phase == "LR4":
            rep_state["viable_rep"] = False
            rep_state["Bottom_ROM_error"] = True

        # Rep count detection (down -> rest)
        if prev_phase == "LR4" and rep_state["phase"] == "LR1":
            if rep_state.get("viable_rep", True):
                rep_state["rep_counter"] = rep_state.get("rep_counter", 0) + 1
            rep_state["viable_rep"] = True

        rep_state["prev_phase"] = rep_state["phase"]
        rep_state["prev_angle"] = angle

    # ---------- Run ONNX when buffer full ----------
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32)  # shape (window_size, features)
        try:
            scaled = scaler.transform(window)
        except Exception as e:
            return {"form_status": f"Scaler Error: {e}", "rep_state": rep_state}

        # ONNX expects (batch, seq_len, features)
        onnx_input = scaled[np.newaxis, :, :].astype(np.float32)

        # Prepare input name (robust)
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: onnx_input}
        ort_outs = session.run(None, ort_inputs)
        recon = ort_outs[0]  # expected shape (1, seq_len, features)

        # compute reconstruction MSE
        err = float(np.mean((onnx_input - recon) ** 2))

        # Decide form status (preserve your messages)
        if err > threshold:
            rep_state["viable_rep"] = False
            status = "POOR FORM!"
        elif rep_state.get("Top_ROM_error"):
            status = "Raise elbows higher!"
        elif rep_state.get("Bottom_ROM_error"):
            status = "Relax arms at the end!"
        # Wrist vs elbow check when angle high
        elif angle > 45:
            # note: y smaller => higher in image coordinates
            right_wrist_higher = landmarks_dict.get("RIGHT_WRIST_y", 1.0) < landmarks_dict.get("RIGHT_ELBOW_y", 1.0)
            left_wrist_higher = landmarks_dict.get("LEFT_WRIST_y", 1.0) < landmarks_dict.get("LEFT_ELBOW_y", 1.0)
            if right_wrist_higher or left_wrist_higher:
                rep_state["viable_rep"] = False
                status = "Wrist higher than elbow!"
            else:
                status = "Good Form"
        else:
            status = "Good Form"

        return {
            "form_status": status,
            "reconstruction_error": err,
            "rep_state": rep_state
        }

    # Buffer not full yet
    return {"form_status": "Analyzing...", "rep_state": rep_state}


# ===============================
# Helper: extract_angles_from_landmarks (kept from your file)
# ===============================
def extract_angles_from_landmarks(landmarks_dict):
    # --- Left landmarks ---
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    left_elbow = [landmarks_dict['LEFT_ELBOW_x'], landmarks_dict['LEFT_ELBOW_y']]
    left_wrist = [landmarks_dict['LEFT_WRIST_x'], landmarks_dict['LEFT_WRIST_y']]
    left_hip = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']]

    # --- Right landmarks ---
    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    right_elbow = [landmarks_dict['RIGHT_ELBOW_x'], landmarks_dict['RIGHT_ELBOW_y']]
    right_wrist = [landmarks_dict['RIGHT_WRIST_x'], landmarks_dict['RIGHT_WRIST_y']]
    right_hip = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']]

    # --- Shoulder Angles (Arm Abduction) ---
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

    # --- Elbow Angles (Flexion/Extension) ---
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # --- Torso Lean Angles ---
    left_hip_vertical = [left_hip[0], left_hip[1] + 1]
    right_hip_vertical = [right_hip[0], right_hip[1] + 1]

    left_torso_angle = calculate_angle(left_shoulder, left_hip, left_hip_vertical)
    right_torso_angle = calculate_angle(right_shoulder, right_hip, right_hip_vertical)

    # --- Shoulder Elevation (shrugging detection) ---
    left_shoulder_elevation = left_shoulder[1] - left_hip[1]
    right_shoulder_elevation = right_shoulder[1] - right_hip[1]

    # --- Wrist Alignment Angles ---
    left_wrist_horizontal = [left_wrist[0] + 1, left_wrist[1]]
    right_wrist_horizontal = [right_wrist[0] + 1, right_wrist[1]]

    left_wrist_angle = calculate_angle(left_elbow, left_wrist, left_wrist_horizontal)
    right_wrist_angle = calculate_angle(right_elbow, right_wrist, right_wrist_horizontal)

    # --- Arm Drift (forward/backward) ---
    left_arm_drift = left_elbow[0] - left_shoulder[0]
    right_arm_drift = right_elbow[0] - right_shoulder[0]

    return [
        left_shoulder_angle,
        right_shoulder_angle,
        left_elbow_angle,
        right_elbow_angle,
        left_torso_angle,
        right_torso_angle,
        left_shoulder_elevation,
        right_shoulder_elevation,
        left_wrist_angle,
        right_wrist_angle,
        left_arm_drift,
        right_arm_drift
    ]


# ===============================
# Reset rep counter utility
# ===============================
def reset_rep_counter():
    return {
        'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "LR1",
        'viable_rep': True,
        'Top_ROM_error': False,
        'Bottom_ROM_error': False
    }
