# models/squat/inference.py (ONNX-ready)
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort

# ===============================
# 📐 Angle Calculation
# ===============================
def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_val = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ===============================
# Landmark configuration
# ===============================
LANDMARK_NAMES = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE','LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX',
    'LEFT_HEEL', 'RIGHT_HEEL'
]

LEFT_LANDMARKS = [
    'LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE', 'LEFT_FOOT_INDEX', 'LEFT_HEEL'
]

RIGHT_LANDMARKS = [
    'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_FOOT_INDEX', 'RIGHT_HEEL'
]

LANDMARK_INDICES = {
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28, 'LEFT_HEEL': 29, 'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
}


# ===============================
# MediaPipe pose (shared instance)
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# ===============================
# Angle extractor (keeps your original logic)
# ===============================
def extract_angles_from_landmarks(landmarks_dict):
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    left_hip = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']]
    left_knee = [landmarks_dict['LEFT_KNEE_x'], landmarks_dict['LEFT_KNEE_y']]
    left_ankle = [landmarks_dict['LEFT_ANKLE_x'], landmarks_dict['LEFT_ANKLE_y']]
    left_foot_index = [landmarks_dict['LEFT_FOOT_INDEX_x'], landmarks_dict['LEFT_FOOT_INDEX_y']]

    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    right_hip = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']]
    right_knee = [landmarks_dict['RIGHT_KNEE_x'], landmarks_dict['RIGHT_KNEE_y']]
    right_ankle = [landmarks_dict['RIGHT_ANKLE_x'], landmarks_dict['RIGHT_ANKLE_y']]
    right_foot_index = [landmarks_dict['RIGHT_FOOT_INDEX_x'], landmarks_dict['RIGHT_FOOT_INDEX_y']]

    # Use knee visibility to pick "active side" (reused logic)
    LeftKnee_vis = float(landmarks_dict.get("LEFT_KNEE_visibility", 0.0))
    RightKnee_vis = float(landmarks_dict.get("RIGHT_KNEE_visibility", 0.0))
    active_side = "left" if LeftKnee_vis > RightKnee_vis else "right"

    vertical_point_left = [left_hip[0], left_hip[1] - 1.0]
    vertical_point_right = [right_hip[0], right_hip[1] - 1.0]

    if active_side == "left":
        knee_angles = calculate_angle(left_hip, left_knee, left_ankle)
        hip_angles = calculate_angle(left_shoulder, left_hip, left_knee)
        torso_angles = calculate_angle(left_shoulder, left_hip, vertical_point_left)
        ankle_angles = calculate_angle(left_knee, left_ankle, left_foot_index)
    else:
        knee_angles = calculate_angle(right_hip, right_knee, right_ankle)
        hip_angles = calculate_angle(right_shoulder, right_hip, right_knee)
        torso_angles = calculate_angle(right_shoulder, right_hip, vertical_point_right)
        ankle_angles = calculate_angle(right_knee, right_ankle, right_foot_index)

    return [
        active_side,
        knee_angles,
        hip_angles,
        torso_angles,
        ankle_angles
    ]


# ===============================
# ONNX analyze_frame (squat)
# ===============================
def analyze_frame(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):
    """
    ONNX-ready analyze_frame for squat:
    - frame: BGR OpenCV image
    - session: onnxruntime.InferenceSession
    - scaler: sklearn scaler (joblib)
    - threshold: reconstruction error threshold
    - buffer: deque(maxlen=window_size)
    - window_size: sequence length
    - num_features: expected feature length (22)
    - rep_state: dict tracking rep info (mutated)
    """

    # convert color for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    # extract landmarks
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = float(lm.x)
        landmarks_dict[f"{name}_y"] = float(lm.y)
        landmarks_dict[f"{name}_visibility"] = float(lm.visibility)

    # compute angles & active side
    angles = extract_angles_from_landmarks(landmarks_dict)
    active_side = angles[0]
    knee_angle = angles[1]  # used for rep counting

    # build feature vector: angles (exclude active string) + selected side landmarks
    feature_vector = []
    feature_vector.extend([float(x) for x in angles[1:]])  # knee, hip, torso, ankle  -> 4 features

    selection = LEFT_LANDMARKS if active_side == "left" else RIGHT_LANDMARKS

    for name in selection:
        feature_vector.extend([
            landmarks_dict.get(f"{name}_x", 0.0),
            landmarks_dict.get(f"{name}_y", 0.0),
            landmarks_dict.get(f"{name}_visibility", 0.0)
        ])  # 6 landmarks * 3 = 18 features

    # validate and append
    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        # safe fallback: log and push zeros
        print(f"Warning: Feature mismatch. Expected {num_features}, got {len(feature_vector)}")
        buffer.append([0.0] * num_features)

    # ---- Rep counting logic (knee angle based) ----
    angle = knee_angle
    if angle is not None:
        if rep_state.get('prev_angle') is None:
            rep_state['prev_angle'] = angle

        prev_angle = rep_state.get('prev_angle', angle)
        prev_phase = rep_state.get('prev_phase')

        # Phase thresholds & transitions
        if angle >= 160 and prev_angle < 160:
            rep_state['Bottom_ROM_error'] = False
            rep_state['phase'] = "S1"  # rest / standing
        elif angle <= 90:
            rep_state['phase'] = "S3"  # bottom (squat)
        elif angle < prev_angle and angle < 160 and angle > 90:
            rep_state['phase'] = "S2"  # going down
        elif angle > prev_angle and angle < 160 and angle > 90:
            rep_state['phase'] = "S4"  # going up

        # ROM checks
        if prev_phase is not None:
            if rep_state['phase'] == "S2" and prev_phase == "S4":
                rep_state['viable_rep'] = False
                rep_state['Bottom_ROM_error'] = True

        # rep completion (going up -> standing)
        if prev_phase == "S4" and rep_state['phase'] == "S1":
            if rep_state.get('viable_rep', True):
                rep_state['rep_counter'] = rep_state.get('rep_counter', 0) + 1
            rep_state['viable_rep'] = True

        rep_state['prev_phase'] = rep_state['phase']
        rep_state['prev_angle'] = angle

    # ---- Run ONNX when buffer full ----
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32)  # (window_size, features)
        try:
            scaled = scaler.transform(window)  # same shape
        except Exception as e:
            return {"form_status": f"Scaler Error: {e}", "rep_state": rep_state}

        onnx_input = scaled[np.newaxis, :, :].astype(np.float32)  # (1, seq_len, features)
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: onnx_input}
        ort_outs = session.run(None, ort_inputs)
        recon = ort_outs[0]  # (1, seq_len, features)

        # reconstruction MSE
        err = float(np.mean((onnx_input - recon) ** 2))

        # decide form status (preserve your messages)
        if err > threshold:
            rep_state['viable_rep'] = False
            status = "BAD FORM!"
        elif rep_state.get('Bottom_ROM_error'):
            status = "Not Going Low Enough!"
        else:
            # extra checks: if torso angle too large -> arching
            torso_angle = angles[3]  # torso
            ankle_angle = angles[4]
            if torso_angle > 50:
                rep_state['viable_rep'] = False
                status = "Don't Arch Your Back!"
            else:
                status = "Good Form"

        return {
            "form_status": status,
            "reconstruction_error": err,
            "rep_state": rep_state
        }

    # buffer not full yet
    return {"form_status": "Analyzing...", "rep_state": rep_state}


# ===============================
# Utility: reset_rep_counter (returns the default rep_state dict)
# ===============================
def reset_rep_counter():
    return {
        'rep_counter': 0,
        'prev_angle': None,
        'prev_phase': None,
        'phase': "S1",
        'viable_rep': True,
        'Bottom_ROM_error': False
    }
