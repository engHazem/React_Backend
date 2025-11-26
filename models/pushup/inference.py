import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort

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
    return float(np.degrees(np.arccos(cosine)))


# ===============================
# Landmark Mapping
# ===============================
LANDMARK_NAMES = [
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE','LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX',
    'LEFT_HEEL', 'RIGHT_HEEL'
]

LEFT_LANDMARKS  = [
 'LEFT_SHOULDER',  'LEFT_ELBOW', 'LEFT_WRIST',  'LEFT_PINKY',
 'LEFT_INDEX', 'LEFT_THUMB','LEFT_HIP', 'LEFT_KNEE','LEFT_ANKLE','LEFT_FOOT_INDEX','LEFT_HEEL'
]

RIGHT_LANDMARKS = [
 'RIGHT_SHOULDER',  'RIGHT_ELBOW','RIGHT_WRIST',  'RIGHT_PINKY',
 'RIGHT_INDEX',  'RIGHT_THUMB','RIGHT_HIP', 'RIGHT_KNEE','RIGHT_ANKLE','RIGHT_FOOT_INDEX','RIGHT_HEEL'
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
# Angle Extraction (unchanged)
# ===============================
def extract_angles_from_landmarks(landmarks_dict):
    left_shoulder = [landmarks_dict['LEFT_SHOULDER_x'], landmarks_dict['LEFT_SHOULDER_y']]
    left_elbow = [landmarks_dict['LEFT_ELBOW_x'], landmarks_dict['LEFT_ELBOW_y']]
    left_wrist = [landmarks_dict['LEFT_WRIST_x'], landmarks_dict['LEFT_WRIST_y']]
    left_hip = [landmarks_dict['LEFT_HIP_x'], landmarks_dict['LEFT_HIP_y']]
    left_knee = [landmarks_dict['LEFT_KNEE_x'], landmarks_dict['LEFT_KNEE_y']]

    right_shoulder = [landmarks_dict['RIGHT_SHOULDER_x'], landmarks_dict['RIGHT_SHOULDER_y']]
    right_elbow = [landmarks_dict['RIGHT_ELBOW_x'], landmarks_dict['RIGHT_ELBOW_y']]
    right_wrist = [landmarks_dict['RIGHT_WRIST_x'], landmarks_dict['RIGHT_WRIST_y']]
    right_hip = [landmarks_dict['RIGHT_HIP_x'], landmarks_dict['RIGHT_HIP_y']]
    right_knee = [landmarks_dict['RIGHT_KNEE_x'], landmarks_dict['RIGHT_KNEE_y']]

    LeftElbow_vis = landmarks_dict.get("LEFT_ELBOW_visibility", 0)
    RightElbow_vis = landmarks_dict.get("RIGHT_ELBOW_visibility", 0)
    active_arm = "left" if LeftElbow_vis > RightElbow_vis else "right"

    vertical_point_left = [left_hip[0], left_hip[1] + 1]
    vertical_point_right = [right_hip[0], right_hip[1] + 1]

    horizontal_point_left = [left_wrist[0] + 1, left_wrist[1]]
    horizontal_point_right = [right_wrist[0] + 1, right_wrist[1]]

    if active_arm == "left":
        elbow_flexion_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        torso_angle = calculate_angle(left_shoulder, left_hip, vertical_point_left)
        wrist_angle = calculate_angle(left_elbow, left_wrist, horizontal_point_left)
        shoulder_elev = (left_shoulder[1] - left_hip[1])
        torso_hip_drift = (left_hip[0] - left_shoulder[0])
    else:
        elbow_flexion_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        torso_angle = calculate_angle(right_shoulder, right_hip, vertical_point_right)
        wrist_angle = calculate_angle(right_elbow, right_wrist, horizontal_point_right)
        shoulder_elev = (right_shoulder[1] - right_hip[1])
        torso_hip_drift = (right_hip[0] - right_shoulder[0])

    return [
        active_arm,
        elbow_flexion_angle,
        shoulder_angle,
        hip_angle,
        torso_angle,
        wrist_angle,
        shoulder_elev,
        torso_hip_drift
    ]


# ===============================
# MediaPipe Pose
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# ===============================
# ONNX analyze_frame (push-ups)
# ===============================
def analyze_frame(frame, session, scaler, threshold, buffer, window_size, num_features, rep_state):
    """
    ONNX-ready push-ups inference.
    - session: onnxruntime.InferenceSession
    - scaler: sklearn scaler
    - returns form_status, reconstruction_error, rep_state
    """

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        buffer.append([0.0] * num_features)
        return {"form_status": "No Pose Detected", "rep_state": rep_state}

    # Extract landmarks into dict
    landmarks_dict = {}
    for name, idx in LANDMARK_INDICES.items():
        lm = results.pose_landmarks.landmark[idx]
        landmarks_dict[f"{name}_x"] = float(lm.x)
        landmarks_dict[f"{name}_y"] = float(lm.y)
        landmarks_dict[f"{name}_visibility"] = float(lm.visibility)

    # Build feature vector (angles + active-side landmarks)
    angles = extract_angles_from_landmarks(landmarks_dict)
    active_arm = angles[0]
    elbow_flexion_angle = angles[1]

    feature_vector = [float(x) for x in angles[1:]]  # add angles excluding active string

    selection = LEFT_LANDMARKS if active_arm == 'left' else RIGHT_LANDMARKS
    for name in selection:
        feature_vector.extend([
            float(landmarks_dict.get(f'{name}_x', 0.0)),
            float(landmarks_dict.get(f'{name}_y', 0.0)),
            float(landmarks_dict.get(f'{name}_visibility', 0.0))
        ])

    # Append to buffer (defensive)
    if len(feature_vector) == num_features:
        buffer.append(feature_vector)
    else:
        print(f"Warning: Feature mismatch. Expected {num_features}, got {len(feature_vector)}")
        buffer.append([0.0] * num_features)

    # --------------------------
    # REP STATE defaults (include good_form_flag)
    # --------------------------
    rep_state.setdefault('prev_angle', None)
    rep_state.setdefault('prev_phase', None)
    rep_state.setdefault('phase', "P1")
    rep_state.setdefault('viable_rep', True)
    rep_state.setdefault('Top_ROM_error', False)
    rep_state.setdefault('Bottom_ROM_error', False)
    rep_state.setdefault('good_form_flag', False)
    rep_state.setdefault('rep_counter', 0)

    # --------------------------
    # REP LOGIC (unchanged except counting only if good_form_flag)
    # --------------------------
    angle = elbow_flexion_angle

    if angle is not None:
        if rep_state['prev_angle'] is None:
            rep_state['prev_angle'] = angle

        prev_angle = rep_state['prev_angle']
        prev_phase = rep_state['prev_phase']

        if angle >= 150 and prev_angle < 150:
            rep_state['Top_ROM_error'] = False
            rep_state['Bottom_ROM_error'] = False
            rep_state['phase'] = "P1"
        elif angle <= 65:
            rep_state['phase'] = "P3"
        elif angle < prev_angle and 65 < angle < 150:
            rep_state['phase'] = "P2"
        elif angle > prev_angle and 65 < angle < 150:
            rep_state['phase'] = "P4"

        if prev_phase is not None and rep_state['phase'] == "P2" and prev_phase == "P4":
            rep_state['viable_rep'] = False
            rep_state['Top_ROM_error'] = True

        if rep_state['phase'] == "P4" and prev_phase == "P2":
            rep_state['viable_rep'] = False
            rep_state['Bottom_ROM_error'] = True

        # Count rep only if good_form_flag was True
        if prev_phase == "P4" and rep_state['phase'] == "P1":
            if rep_state.get('good_form_flag', False):
                rep_state['rep_counter'] += 1
            rep_state['good_form_flag'] = False
            rep_state['viable_rep'] = True

        rep_state['prev_phase'] = rep_state['phase']
        rep_state['prev_angle'] = angle

    # --------------------------
    # ONNX inference when window full
    # --------------------------
    if len(buffer) >= window_size:
        window = np.array(list(buffer), dtype=np.float32)
        try:
            scaled = scaler.transform(window)
        except Exception as e:
            return {"form_status": f"Scaler Error: {e}", "rep_state": rep_state}

        onnx_input = scaled[np.newaxis, :, :].astype(np.float32)
        input_name = session.get_inputs()[0].name
        recon = session.run(None, {input_name: onnx_input})[0]

        err = float(np.mean((onnx_input - recon) ** 2))

        # Form classification (preserve order/messages)
        if rep_state.get('Top_ROM_error'):
            status = "Go up more!"
            rep_state['good_form_flag'] = False
        elif rep_state.get('Bottom_ROM_error'):
            status = "Go down more!"
            rep_state['good_form_flag'] = False
        elif err > threshold:
            rep_state['viable_rep'] = False
            status = "POOR FORM!"
            rep_state['good_form_flag'] = False
        else:
            status = "Good Form"
            rep_state['good_form_flag'] = True

        return {
            "form_status": status,
            "rep_state": rep_state
        }

    return {"form_status": "Analyzing...", "rep_state": rep_state}


# --------------------------
# Reset helper
# --------------------------
def reset_rep_counter(rep_state):
    """Reset the rep counter in the given rep_state dictionary"""
    rep_state['rep_counter'] = 0
    rep_state['prev_angle'] = None
    rep_state['prev_phase'] = None
    rep_state['phase'] = "P1"
    rep_state['viable_rep'] = True
    rep_state['Top_ROM_error'] = False
    rep_state['Bottom_ROM_error'] = False
    rep_state['good_form_flag'] = False
    return rep_state
